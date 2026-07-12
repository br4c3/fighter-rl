"""Torch state-machine port of the flight-control system in supplied f16.xml."""

from collections import namedtuple
import torch

FCSState = namedtuple(
    "FCSState",
    (
        "roll_i",
        "pitch_i",
        "yaw_i",
        "roll_prev",
        "pitch_prev",
        "yaw_prev",
        "aileron",
        "elevator",
        "rudder",
        "lef",
        "tef",
    ),
)


def initial_fcs_state(batch, device=None, dtype=torch.float32):
    z = torch.zeros(batch, device=device, dtype=dtype)
    return FCSState(*(z.clone() for _ in range(11)))


def interp1(x, xp, fp):
    """GPU-friendly clamped piecewise-linear lookup."""
    xp = x.new_tensor(xp)
    fp = x.new_tensor(fp)
    index = torch.searchsorted(xp, x.contiguous()).clamp(1, len(xp) - 1)
    x0, x1 = xp[index - 1], xp[index]
    y0, y1 = fp[index - 1], fp[index]
    return y0 + (x.clamp(xp[0], xp[-1]) - x0) * (y1 - y0) / (x1 - x0)


def rate_limit(current, target, traverse_seconds, dt):
    # XML traverse time is from -1 to +1.
    delta = 2.0 * dt / traverse_seconds
    return current + (target - current).clamp(-delta, delta)


class CompetitionF16FCSTorch(torch.nn.Module):
    """Exact constants/signs from stock_runtime/aircraft/f16/f16.xml."""

    def __init__(self, hz=60):
        super().__init__()
        self.dt = 1.0 / float(hz)

    def forward(
        self,
        state,
        action,
        aircraft_state,
        angular_accel=None,
        pilot_loads=None,
        gravity_attitude=None,
    ):
        """Return new FCS state and [elevator, aileron, rudder, LEF, TEF] deg.

        action: [roll, pitch, rudder, throttle], with first three in [-1,1].
        aircraft_state uses the competition 41-field schema.
        """
        a = action.clamp(-1, 1)
        roll, pitch, rudder = a[:, 0], a[:, 1], a[:, 2]
        phi = torch.deg2rad(aircraft_state[:, 3])
        theta = torch.deg2rad(aircraft_state[:, 4])
        gravity_phi, gravity_theta = (
            (phi, theta)
            if gravity_attitude is None
            else (gravity_attitude[:, 0], gravity_attitude[:, 1])
        )
        alpha = torch.deg2rad(aircraft_state[:, 13])
        p = torch.deg2rad(aircraft_state[:, 9])
        q = torch.deg2rad(aircraft_state[:, 10])
        r = torch.deg2rad(aircraft_state[:, 11])
        mach = aircraft_state[:, 29]
        vg_fps = aircraft_state[:, 28] / 0.3048
        ny, nz = aircraft_state[:, 32], aircraft_state[:, 31]
        # XML feedback uses acceleration at the pilot station, not CG Nz/Ny:
        # a_pilot = a_CG + wdot x R + w x (w x R).
        if pilot_loads is not None:
            ny, nz = pilot_loads[:, 0], pilot_loads[:, 1]
        elif angular_accel is not None:
            omega = torch.stack((p, q, r), 1)
            lever = omega.new_tensor([12.26102756, 0.0, -2.64083882]).expand_as(omega)
            rotational = torch.linalg.cross(angular_accel, lever) + torch.linalg.cross(
                omega, torch.linalg.cross(omega, lever)
            )
            ny = ny + rotational[:, 1] / 32.174
            nz = -nz + rotational[:, 2] / 32.174
        self.last_pilot_ny = ny
        self.last_pilot_nz = nz

        # Roll PID: kp=3, ki=.0005, kd=-.00125.
        re = roll - 0.31821 * p
        # In this XML the PID trigger is 1 at flying speed.  FGPID interprets
        # non-zero as "freeze integrator" (P/D remain active).
        roll_integrate = aircraft_state[:, 12] * 1.94384449 < 20.0
        ri_candidate = state.roll_i + (1.5 * re - 0.5 * state.roll_prev) * self.dt
        ri = torch.where(roll_integrate, ri_candidate, state.roll_i)
        rd = (re - state.roll_prev) / self.dt
        roll_pid = 3.0 * re + 0.00050 * ri - 0.00125 * rd
        roll_target = (roll_pid + roll).clamp(-1, 1)
        an = rate_limit(state.aileron, roll_target, 0.3, self.dt)
        mach_gain = interp1(mach, [0.0, 1.0], [1.0, 0.15])
        aileron_deg = an * mach_gain * torch.rad2deg(an.new_tensor(0.375))

        # Pitch alpha/g/q feedback and actuator.
        limited = pitch.clamp(-1, 0.44)
        alpha_gain = interp1(alpha, [-0.5236, -0.5, 0.0, 0.5, 0.5236], [0.0, 0.11, 1.0, 0.11, 0.0])
        scheduled = limited * alpha_gain
        g_corrected = nz - torch.cos(gravity_theta) * torch.cos(gravity_phi)
        pe = scheduled + 6.2 * q - 0.020 * g_corrected
        self.last_pitch_scheduled = scheduled
        self.last_pitch_rate = 6.2 * q
        self.last_g_corrected = g_corrected
        pitch_integrate = aircraft_state[:, 12] * 1.94384449 < 5.0
        pi_candidate = state.pitch_i + (1.5 * pe - 0.5 * state.pitch_prev) * self.dt
        pi = torch.where(pitch_integrate, pi_candidate, state.pitch_i)
        pitch_pid = (0.3000 * pe + 0.0250 * pi).clamp(-1, 1)
        pitch_target = (scheduled + 1.0472 * alpha + pitch_pid).clamp(-1, 1)
        self.last_pitch_error = pe
        self.last_pitch_pid = pitch_pid
        self.last_pitch_target = pitch_target
        en = rate_limit(state.elevator, pitch_target, 0.3, self.dt)
        elevator_deg = en * torch.rad2deg(en.new_tensor(0.436))

        # Yaw r/ny feedback.
        yaw_gain = interp1(vg_fps, [80.0, 100.0, 150.0], [0.0, 15.0, 100.0])
        ye = rudder + yaw_gain * r + 0.25 * ny
        yaw_integrate = aircraft_state[:, 12] * 1.94384449 < 10.0
        yi_candidate = state.yaw_i + (1.5 * ye - 0.5 * state.yaw_prev) * self.dt
        yi = torch.where(yaw_integrate, yi_candidate, state.yaw_i)
        yd = (ye - state.yaw_prev) / self.dt
        yaw_pid = (0.1055 * ye + 0.000010 * yi + 0.00005 * yd).clamp(-1, 1)
        yaw_target = (rudder + yaw_pid).clamp(-1, 1)
        self.last_yaw_error = ye
        self.last_yaw_pid = yaw_pid
        self.last_yaw_target = yaw_target
        # Both the PID and kinematic component write fcs/rudder-pos-norm.
        # FGPID runs first, so FGKinemat reads yaw_pid as its current output
        # every frame rather than retaining the previous physical deflection.
        rn = rate_limit(yaw_pid, yaw_target, 0.4, self.dt)
        rudder_deg = rn * torch.rad2deg(rn.new_tensor(0.524))

        # Airborne gear-up LEF/TEF switches from XML, including Mach override.
        lef_rad = torch.where(
            alpha > 0.2618,
            alpha.new_tensor(0.436),
            torch.where(alpha > 0.0873, alpha.new_tensor(0.262), alpha.new_zeros(())),
        )
        lef_rad = torch.where(mach > 0.9, alpha.new_tensor(-0.0349), lef_rad)
        lef = rate_limit(state.lef, lef_rad * 2.293578, 3.0, self.dt)
        # Important JSBSim convention: the aero XML functions use
        # fcs/lef-pos-rad, which is the switch command above, not the
        # kinematic/aerosurface output fcs/lef-pos-deg.  The kinematic output
        # is still tracked in FCSState for diagnostics/reset continuity.
        lef_command_deg = torch.rad2deg(lef_rad)
        vc_kts = aircraft_state[:, 12] * 1.94384449
        tef_rad = torch.where(vc_kts < 250.0, alpha.new_tensor(0.349), alpha.new_zeros(()))
        tef_rad = torch.where(mach > 0.9, alpha.new_tensor(-0.0349), tef_rad)
        tef = rate_limit(state.tef, tef_rad * 2.864789, 3.0, self.dt)
        tef_deg = tef * 20.0

        new_state = FCSState(ri, pi, yi, re, pe, ye, an, en, rn, lef, tef)
        # f16.xml intentionally exposes two roll signals: telemetry/physical
        # left aileron after actuator+Mach compensation, and aero/aileron-pos
        # directly from roll-rate-command before that lag.
        aero_aileron_deg = roll_target * torch.rad2deg(an.new_tensor(0.375))
        # Column 6 is the DLL-compatible telemetry value.  The distributed
        # wrapper reports 25*pitch-scheduler, not the physical elevator.
        reported_elevator_deg = 25.0 * pitch_target
        surfaces = torch.stack(
            (
                elevator_deg,
                aileron_deg,
                rudder_deg,
                lef_command_deg,
                tef_deg,
                aero_aileron_deg,
                reported_elevator_deg,
            ),
            1,
        )
        return new_state, surfaces

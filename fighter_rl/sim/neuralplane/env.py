"""End-to-end GPU batched competition F-16 environment."""

from pathlib import Path
import torch

from .fcs import CompetitionF16FCSTorch, FCSState, initial_fcs_state
from .engine import CompetitionF100, EngineState
from .xml_aero import CompetitionXMLAero, airborne_properties
from .dynamics import CompetitionDynamics
from .atmosphere import standard_atmosphere, calibrated_airspeed
from .eci import (
    OMEGA,
    body_to_ned_matrix,
    earth_rotation,
    ecef_to_geocentric,
    ecef_to_geodetic,
    geocentric_to_ecef,
    geodetic_to_ecef,
    gravity_j2,
    matrix_to_euler,
    matrix_to_quaternion,
    ned_to_ecef_matrix,
    quaternion_to_matrix,
)

ROOT = Path(__file__).resolve().parents[3]
FT_TO_M = 0.3048


def euler_to_quaternion(euler):
    roll, pitch, yaw = (euler[:, i] * 0.5 for i in range(3))
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        1,
    )


def quaternion_to_euler(q):
    w, x, y, z = q.unbind(1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return torch.stack((roll, pitch, yaw), 1)


class CompetitionNeuralPlane(torch.nn.Module):
    """All aircraft advance in a single sequence of batched Torch operations."""

    def __init__(self, num_envs, device="cpu", hz=60, dtype=torch.float32, use_eci=True):
        super().__init__()
        self.num_envs = int(num_envs)
        self.hz = int(hz)
        self.dt = 1.0 / self.hz
        self.fcs = CompetitionF16FCSTorch(hz)
        self.engine = CompetitionF100(ROOT / "stock_runtime/engine/F100-PW-229.xml", hz)
        self.aero = CompetitionXMLAero(ROOT / "stock_runtime/aircraft/f16/f16.xml")
        self.dynamics = CompetitionDynamics()
        self.to(device=device, dtype=dtype)
        self.device = torch.device(device)
        self.dtype = dtype
        self.use_eci = bool(use_eci)
        self.state = torch.zeros(self.num_envs, 12, device=self.device, dtype=dtype)
        self.fcs_state = initial_fcs_state(self.num_envs, self.device, dtype=dtype)
        self.engine_state = self.engine.initial(self.num_envs, self.device)
        self.surfaces = torch.zeros(self.num_envs, 7, device=self.device, dtype=dtype)
        self.ny = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self.nz = torch.ones(self.num_envs, device=self.device, dtype=dtype)
        self.angular_accel = torch.zeros(self.num_envs, 3, device=self.device, dtype=dtype)
        self.pilot_ny = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self.pilot_nz = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self.pending_aero = torch.zeros(self.num_envs, 6, device=self.device, dtype=dtype)
        self.pending_thrust = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self.gear_pos = torch.ones(self.num_envs, device=self.device, dtype=dtype)
        self.body_accel_history = torch.zeros(self.num_envs, 2, 3, device=self.device, dtype=dtype)
        self.position_rate_history = torch.zeros(
            self.num_envs, 2, 3, device=self.device, dtype=dtype
        )
        self.quaternion = torch.zeros(self.num_envs, 4, device=self.device, dtype=dtype)
        self.fuel_lbs = torch.full(
            (self.num_envs,), 6971.963548061171, device=self.device, dtype=dtype
        )
        self.frame_index = 0
        self.aero_scale = torch.ones(self.num_envs, 6, device=self.device, dtype=dtype)
        self.thrust_scale = torch.ones(self.num_envs, device=self.device, dtype=dtype)
        # Absolute ECI coordinates are O(2e7 ft); float32 has ~2 ft ULP there.
        # Keep only the small propagation state in float64 while aero/FCS and
        # policy tensors retain the requested training dtype.
        self.eci_position = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float64)
        self.eci_velocity = torch.zeros_like(self.eci_position)
        self.eci_quaternion = torch.zeros(self.num_envs, 4, device=self.device, dtype=dtype)
        self.pqri = torch.zeros(self.num_envs, 3, device=self.device, dtype=dtype)
        self.epa = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        self.eci_accel_history = torch.zeros(
            self.num_envs, 2, 3, device=self.device, dtype=torch.float64
        )
        self.eci_velocity_history = torch.zeros(
            self.num_envs, 2, 3, device=self.device, dtype=torch.float64
        )
        self.release_position_m = torch.zeros(self.num_envs, 3, device=self.device, dtype=dtype)

    def reset(
        self, altitude_m=6000.0, speed_mps=250.0, roll_deg=0.0, pitch_deg=0.0, heading_deg=0.0
    ):
        def column(value):
            t = torch.as_tensor(value, dtype=self.dtype, device=self.device)
            return t.expand(self.num_envs) if t.ndim == 0 else t

        alt, speed, roll, pitch, heading = map(
            column, (altitude_m, speed_mps, roll_deg, pitch_deg, heading_deg)
        )
        s = torch.zeros(self.num_envs, 51, device=self.device, dtype=self.dtype)
        s[:, 2] = -alt
        s[:, 3] = roll
        s[:, 4] = pitch
        s[:, 5] = heading
        s[:, 6] = speed
        s[:, 12] = speed
        s[:, 27] = speed
        s[:, 28] = speed
        s[:, 29] = speed / 340.0
        s[:, 31] = 1.0
        s[:, 22] = 100.0
        s[:, 33] = 100.0
        s[:, 42] = 37.91455691666667
        s[:, 43] = 128.18188127777776
        s[:, 44] = alt
        return self.reset_from_exact41(s)

    def reset_from_exact41(self, state41):
        s = torch.as_tensor(state41, dtype=self.dtype, device=self.device)
        if s.ndim == 1:
            s = s.expand(self.num_envs, -1)
        if self.use_eci:
            self.release_position_m = s[:, :3]
        u, v, w = s[:, 6], s[:, 7], s[:, 8]
        vt = torch.sqrt(u * u + v * v + w * w).clamp_min(1e-3)
        self.state = torch.stack(
            (
                s[:, 0] / FT_TO_M,
                s[:, 1] / FT_TO_M,
                -s[:, 2] / FT_TO_M,
                torch.deg2rad(s[:, 3]),
                torch.deg2rad(s[:, 4]),
                torch.deg2rad(s[:, 5]),
                vt / FT_TO_M,
                torch.atan2(w, u),
                torch.asin((v / vt).clamp(-0.999, 0.999)),
                torch.deg2rad(s[:, 9]),
                torch.deg2rad(s[:, 10]),
                torch.deg2rad(s[:, 11]),
            ),
            1,
        )
        self.quaternion = euler_to_quaternion(self.state[:, 3:6])
        self.fcs_state = initial_fcs_state(self.num_envs, self.device, dtype=self.dtype)
        mach = s[:, 29].clamp_min(0.01)
        # 18 is DLL telemetry (pre-actuator); 17/19/21 carry physical surfaces
        # when supplied by the Exact adapter.  Fall back for older recordings.
        physical_elevator = torch.where(s[:, 19].abs() > 1e-8, s[:, 19], s[:, 18])
        physical_rudder = torch.where(s[:, 21].abs() > 1e-8, s[:, 21], s[:, 20])
        physical_aileron = torch.where(s[:, 17].abs() > 1e-8, s[:, 17], s[:, 16])
        en = (physical_elevator / torch.rad2deg(s.new_tensor(0.436))).clamp(-1, 1)
        rn = (physical_rudder / torch.rad2deg(s.new_tensor(0.524))).clamp(-1, 1)
        an = (physical_aileron / (torch.rad2deg(s.new_tensor(0.375)) * interp_mach(mach))).clamp(
            -1, 1
        )
        fs = list(self.fcs_state)
        fs[6], fs[7], fs[8] = an, en, rn
        self.fcs_state = FCSState(*fs)
        n1 = torch.where(s[:, 22] > 1, s[:, 22], torch.full_like(s[:, 22], 100.0))
        n2 = torch.where(s[:, 33] > 1, s[:, 33], torch.full_like(s[:, 33], 100.0))
        initial_flow = self.engine.steady_dry_fuel_flow(n2, mach, self.state[:, 2])
        self.engine_state = EngineState(n1, n2, torch.zeros_like(n1), initial_flow)
        self.fuel_lbs = torch.where(s[:, 23] > 1, s[:, 23], torch.full_like(n1, 6971.963548061171))
        self.ny, self.nz = s[:, 32], s[:, 31]
        # Exact/JSBSim reset starts with gear_pos=1 and retracts to 0 over
        # roughly five seconds.  Teacher-forced diagnostics that reset from a
        # mid-trajectory Exact state must overwrite this with the Exact gear
        # property, because the public 41-state vector does not carry gear.
        self.gear_pos.fill_(1.0)
        self.frame_index = 0
        self.surfaces[:, 0] = physical_elevator
        self.surfaces[:, 1] = physical_aileron
        self.surfaces[:, 2] = physical_rudder
        alpha = self.state[:, 7]
        lef_rad = torch.where(
            alpha > 0.2618,
            alpha.new_tensor(0.436),
            torch.where(alpha > 0.0873, alpha.new_tensor(0.262), alpha.new_zeros(())),
        )
        lef_rad = torch.where(mach > 0.9, alpha.new_tensor(-0.0349), lef_rad)
        self.surfaces[:, 3] = torch.rad2deg(lef_rad)
        self.surfaces[:, 6] = s[:, 18]
        self.pending_aero = (
            self.aero(airborne_properties(self.state, self.surfaces, gear_pos=self.gear_pos))
            * self.aero_scale
        )
        # DLL reset performs one full-throttle warm-start. The pending force
        # therefore belongs to throttle=1.0 even before the first policy step.
        self.engine_state = self.engine(
            self.engine_state, torch.ones_like(n1), mach, self.state[:, 2]
        )
        self.pending_thrust = self.engine_state.thrust_lbf * self.thrust_scale
        derivative = self.dynamics(
            self.state, self.pending_aero, self.pending_thrust, self.fuel_lbs
        )
        body_accel = self._body_acceleration(self.state, derivative)
        # RunIC initializes AB histories before InitRunning and the single
        # full-throttle warm frame.  Therefore the history consumed by the
        # first policy step is the same aerodynamic frame with zero engine
        # thrust, not the current post-warm acceleration.
        prewarm_derivative = self.dynamics(
            self.state, self.pending_aero, torch.zeros_like(self.pending_thrust), self.fuel_lbs
        )
        prewarm_nonrotating = self._nonrotating_acceleration(self.state, prewarm_derivative)
        self.body_accel_history[:] = prewarm_nonrotating[:, None, :]
        prewarm_specific = self.dynamics.specific_force(
            self.state, self.pending_aero, torch.zeros_like(self.pending_thrust), self.fuel_lbs
        )
        omega = self.state[:, 9:12]
        lever = omega.new_tensor([12.26102756, 0.0, -2.64083882]).expand_as(omega)
        prewarm_rot = torch.linalg.cross(prewarm_derivative[:, 9:12], lever) + torch.linalg.cross(
            omega, torch.linalg.cross(omega, lever)
        )
        estimated_pilot_y = prewarm_specific[:, 1] / 32.174 + prewarm_rot[:, 1] / 32.174
        estimated_pilot_z = prewarm_specific[:, 2] / 32.174 + prewarm_rot[:, 2] / 32.174
        # Exact adapter carries native delayed Auxiliary values in spare slots.
        has_native = (s[:, 34].abs() + s[:, 35].abs()) > 1e-9
        self.pilot_ny = torch.where(has_native, s[:, 34], estimated_pilot_y)
        self.pilot_nz = torch.where(has_native, s[:, 35], estimated_pilot_z)
        g_corrected = self.pilot_nz - torch.cos(self.state[:, 4]) * torch.cos(self.state[:, 3])
        estimated_pitch_i = (-0.020 * g_corrected) * self.dt
        pitch_i = torch.where(s[:, 36].abs() > 1e-12, s[:, 36] / 0.025, estimated_pitch_i)
        fs = list(self.fcs_state)
        fs[1] = pitch_i
        self.fcs_state = FCSState(*fs)
        self.position_rate_history[:] = derivative[:, :3, None].transpose(1, 2)
        self._reset_eci(s, prewarm_specific)
        return self.observation41()

    def reset_from_release51(self, state51):
        """Reset from the public Release_260708 ``FighterSim.JSBSim`` state.

        The competition wrapper and the Exact adapter use different spare-slot
        layouts after column 16.  In particular Release columns 17/19/21 are
        stick/throttle commands, while ``reset_from_exact41`` expects physical
        actuator and native auxiliary values there.  Keep this conversion
        explicit so release-vs-fast validation targets the actual competition
        state contract.
        """
        s = torch.as_tensor(state51, dtype=self.dtype, device=self.device)
        if s.ndim == 1:
            s = s.expand(self.num_envs, -1)
        mapped = torch.zeros(
            (self.num_envs, max(51, s.shape[1])), dtype=self.dtype, device=self.device
        )
        mapped[:, :16] = s[:, :16]
        if self.use_eci and s.shape[1] > 44:
            mapped[:, 2] = -s[:, 44]
        mapped[:, 16] = s[:, 16]
        mapped[:, 17] = s[:, 16]  # physical aileron, not release pitch command
        mapped[:, 18] = s[:, 18]
        mapped[:, 19] = s[:, 18]  # physical elevator, not release rudder command
        mapped[:, 20] = s[:, 20]
        mapped[:, 21] = s[:, 20]  # physical rudder, not release throttle command
        mapped[:, 22:34] = s[:, 22:34]
        mapped[:, 41:46] = s[:, 41:46]
        obs = self.reset_from_exact41(mapped)
        if self.use_eci:
            alt_m = (-s[:, 2]).clamp_min(0.0)
            if s.shape[1] > 44:
                alt_m = torch.where(alt_m > 1.0, alt_m, s[:, 44].clamp_min(0.0))
            obs = self.set_local_position_m(s[:, 0], s[:, 1], alt_m)
        thrust = self.engine.thrust_from_n2(
            self.engine_state.n2,
            s[:, 21].clamp(0, 1),
            mapped[:, 29].clamp_min(0.01),
            self.state[:, 2],
        )
        self.engine_state = EngineState(
            self.engine_state.n1, self.engine_state.n2, thrust, self.engine_state.fuel_flow_pph
        )
        self.pending_thrust = thrust * self.thrust_scale
        return obs

    def randomize_model(self, force_fraction=0.0, moment_fraction=0.0, thrust_fraction=0.0):
        """Apply independent per-environment uncertainty around the XML model."""
        ff, mf, tf = map(float, (force_fraction, moment_fraction, thrust_fraction))
        self.aero_scale[:, :3].uniform_(1 - ff, 1 + ff)
        self.aero_scale[:, 3:].uniform_(1 - mf, 1 + mf)
        self.thrust_scale.uniform_(1 - tf, 1 + tf)
        self.pending_aero = (
            self.aero(airborne_properties(self.state, self.surfaces, gear_pos=self.gear_pos))
            * self.aero_scale
        )
        self.pending_thrust = self.engine_state.thrust_lbf * self.thrust_scale

    def _reset_eci(self, s, prewarm_specific):
        lat0 = self.state.new_tensor(37.91455691666667 * torch.pi / 180)
        lon0 = self.state.new_tensor(128.18188127777776 * torch.pi / 180)
        if s.shape[1] > 44:
            lat = torch.where(
                s[:, 42].abs() > 1, torch.deg2rad(s[:, 42]), lat0.expand(self.num_envs)
            )
            lon = torch.where(
                s[:, 43].abs() > 1, torch.deg2rad(s[:, 43]), lon0.expand(self.num_envs)
            )
        else:
            lat = lat0.expand(self.num_envs)
            lon = lon0.expand(self.num_envs)
        # The release DLL's propagation behaves like a radial/geocentric local
        # frame, while FighterSim later feeds the exported Lat/Lon/Alt directly
        # to pymap3d.geodetic2ned.  Keep that split: radial ECI internally,
        # release-style pseudo-geodetic projection at observation time.
        ecef = geocentric_to_ecef(lat.double(), lon.double(), self.state[:, 2].double())
        r_n2e = ned_to_ecef_matrix(lat, lon)
        r_b2e = r_n2e @ body_to_ned_matrix(self.state[:, 3:6])
        self.epa.zero_()
        self.eci_position = ecef
        self.eci_quaternion = matrix_to_quaternion(r_b2e)
        uvw = self._uvw(self.state)
        omega_i = self.state.new_tensor([0.0, 0.0, OMEGA]).expand_as(uvw)
        self.eci_velocity = (r_b2e.double() @ uvw.double().unsqueeze(2)).squeeze(
            2
        ) + torch.linalg.cross(omega_i.double(), self.eci_position)
        omega_b = (r_b2e.transpose(1, 2) @ omega_i.unsqueeze(2)).squeeze(2)
        self.pqri = self.state[:, 9:12] + omega_b
        accel = (r_b2e.double() @ prewarm_specific.double().unsqueeze(2)).squeeze(2) + gravity_j2(
            ecef
        )
        self.eci_accel_history[:] = accel[:, None, :]
        self.eci_velocity_history[:] = self.eci_velocity[:, None, :]

    def set_local_position_m(self, north_m, east_m, alt_m):
        """Move an already-reset aircraft without desynchronizing ECI state."""

        def column(value):
            t = torch.as_tensor(value, dtype=self.dtype, device=self.device)
            return t.expand(self.num_envs) if t.ndim == 0 else t

        north_m, east_m, alt_m = map(column, (north_m, east_m, alt_m))
        self.release_position_m[:, 0] = north_m
        self.release_position_m[:, 1] = east_m
        self.release_position_m[:, 2] = -alt_m
        north_ft = north_m.double() / FT_TO_M
        east_ft = east_m.double() / FT_TO_M
        alt_ft = alt_m.double() / FT_TO_M
        self.state[:, 0] = north_ft.to(self.dtype)
        self.state[:, 1] = east_ft.to(self.dtype)
        self.state[:, 2] = alt_ft.to(self.dtype)
        if not self.use_eci:
            return self.observation41()
        origin_lat = torch.full(
            (self.num_envs,),
            37.91455691666667 * torch.pi / 180,
            device=self.device,
            dtype=torch.float64,
        )
        origin_lon = torch.full(
            (self.num_envs,),
            128.18188127777776 * torch.pi / 180,
            device=self.device,
            dtype=torch.float64,
        )
        origin = geodetic_to_ecef(origin_lat, origin_lon, torch.zeros_like(origin_lat))
        ned = torch.stack((north_ft, east_ft, -alt_ft), 1)
        obs_ecef = origin + (ned_to_ecef_matrix(origin_lat, origin_lon) @ ned.unsqueeze(2)).squeeze(
            2
        )
        lat, lon, radial_alt = ecef_to_geodetic(obs_ecef)
        ecef = geocentric_to_ecef(lat, lon, radial_alt)
        r_n2e = ned_to_ecef_matrix(lat.to(self.dtype), lon.to(self.dtype))
        r_b2e = r_n2e @ body_to_ned_matrix(self.state[:, 3:6])
        self.epa.zero_()
        self.eci_position = ecef
        self.eci_quaternion = matrix_to_quaternion(r_b2e)
        uvw = self._uvw(self.state)
        omega_i = self.state.new_tensor([0.0, 0.0, OMEGA]).expand_as(uvw)
        self.eci_velocity = (r_b2e.double() @ uvw.double().unsqueeze(2)).squeeze(
            2
        ) + torch.linalg.cross(omega_i.double(), self.eci_position)
        omega_b = (r_b2e.transpose(1, 2) @ omega_i.unsqueeze(2)).squeeze(2)
        self.pqri = self.state[:, 9:12] + omega_b
        self.pending_aero = (
            self.aero(airborne_properties(self.state, self.surfaces, gear_pos=self.gear_pos))
            * self.aero_scale
        )
        self.pending_thrust = self.engine_state.thrust_lbf * self.thrust_scale
        prewarm_specific = self.dynamics.specific_force(
            self.state, self.pending_aero, torch.zeros_like(self.pending_thrust), self.fuel_lbs
        )
        accel = (r_b2e.double() @ prewarm_specific.double().unsqueeze(2)).squeeze(2) + gravity_j2(
            ecef
        )
        self.eci_accel_history[:] = accel[:, None, :]
        self.eci_velocity_history[:] = self.eci_velocity[:, None, :]
        return self.observation41()

    @staticmethod
    def _uvw(state):
        vt, alpha, beta = state[:, 6], state[:, 7], state[:, 8]
        return torch.stack(
            (
                vt * torch.cos(beta) * torch.cos(alpha),
                vt * torch.sin(beta),
                vt * torch.cos(beta) * torch.sin(alpha),
            ),
            1,
        )

    def _eci_attitude(self, q, epa, position):
        r_i2e = earth_rotation(epa)
        ecef = (r_i2e @ position.unsqueeze(2)).squeeze(2)
        lat, lon, _ = ecef_to_geocentric(ecef)
        r_b2e = r_i2e @ quaternion_to_matrix(q).double()
        r_b2n = ned_to_ecef_matrix(lat, lon).transpose(1, 2) @ r_b2e
        return matrix_to_euler(r_b2n).to(q.dtype)

    def observation41(self):
        x = self.state
        out = torch.zeros(self.num_envs, 41, device=self.device, dtype=self.dtype)
        if self.use_eci:
            out[:, 0:3] = self.release_position_m
        else:
            out[:, 0] = x[:, 0] * FT_TO_M
            out[:, 1] = x[:, 1] * FT_TO_M
            out[:, 2] = -x[:, 2] * FT_TO_M
        out[:, 3:6] = torch.rad2deg(x[:, 3:6])
        ca, sa, cb, sb = (
            torch.cos(x[:, 7]),
            torch.sin(x[:, 7]),
            torch.cos(x[:, 8]),
            torch.sin(x[:, 8]),
        )
        out[:, 6] = x[:, 6] * ca * cb * FT_TO_M
        out[:, 7] = x[:, 6] * sb * FT_TO_M
        out[:, 8] = x[:, 6] * sa * cb * FT_TO_M
        out[:, 9:12] = torch.rad2deg(x[:, 9:12])
        out[:, 13] = torch.rad2deg(x[:, 7])
        out[:, 14] = torch.rad2deg(x[:, 8])
        out[:, 16] = self.surfaces[:, 1]
        out[:, 18] = self.surfaces[:, 6]
        out[:, 20] = self.surfaces[:, 2]
        out[:, 22] = self.engine_state.n1
        out[:, 27] = x[:, 6] * FT_TO_M
        out[:, 28] = out[:, 27]
        # Exact JSBSim pitot/calibrated speed drives PID and TEF switches.
        out[:, 12] = calibrated_airspeed(x[:, 6], x[:, 2]) * FT_TO_M
        out[:, 23] = self.fuel_lbs
        _, sound_speed = standard_atmosphere(x[:, 2])
        out[:, 29] = x[:, 6] / sound_speed
        out[:, 31] = self.nz
        out[:, 32] = self.ny
        out[:, 33] = self.engine_state.n2
        return out

    def _derivative(self, state, surfaces, thrust):
        props = airborne_properties(state, surfaces)
        forces = self.aero(props)
        return self.dynamics(state, forces, thrust, self.fuel_lbs)

    @staticmethod
    def _body_acceleration(state, derivative):
        vt, alpha, beta = state[:, 6], state[:, 7], state[:, 8]
        sa, ca, sb, cb = torch.sin(alpha), torch.cos(alpha), torch.sin(beta), torch.cos(beta)
        vd, ad, bd = derivative[:, 6], derivative[:, 7], derivative[:, 8]
        ud = cb * ca * vd - vt * sb * ca * bd - vt * cb * sa * ad
        vv = sb * vd + vt * cb * bd
        wd = cb * sa * vd - vt * sb * sa * bd + vt * cb * ca * ad
        return torch.stack((ud, vv, wd), 1)

    @staticmethod
    def _body_to_ned(vector, attitude):
        phi, theta, psi = attitude.unbind(1)
        u, v, w = vector.unbind(1)
        sp, cp, st, ct, ss, cs = (
            torch.sin(phi),
            torch.cos(phi),
            torch.sin(theta),
            torch.cos(theta),
            torch.sin(psi),
            torch.cos(psi),
        )
        return torch.stack(
            (
                u * ct * cs + v * (sp * st * cs - cp * ss) + w * (cp * st * cs + sp * ss),
                u * ct * ss + v * (sp * st * ss + cp * cs) + w * (cp * st * ss - sp * cs),
                -u * st + v * sp * ct + w * cp * ct,
            ),
            1,
        )

    @staticmethod
    def _ned_to_body(vector, attitude):
        phi, theta, psi = attitude.unbind(1)
        n, e, d = vector.unbind(1)
        sp, cp, st, ct, ss, cs = (
            torch.sin(phi),
            torch.cos(phi),
            torch.sin(theta),
            torch.cos(theta),
            torch.sin(psi),
            torch.cos(psi),
        )
        return torch.stack(
            (
                ct * cs * n + ct * ss * e - st * d,
                (sp * st * cs - cp * ss) * n + (sp * st * ss + cp * cs) * e + sp * ct * d,
                (cp * st * cs + sp * ss) * n + (cp * st * ss - sp * cs) * e + cp * ct * d,
            ),
            1,
        )

    def _nonrotating_acceleration(self, state, derivative):
        uvwdot = self._body_acceleration(state, derivative)
        vt, alpha, beta = state[:, 6], state[:, 7], state[:, 8]
        uvw = torch.stack(
            (
                vt * torch.cos(beta) * torch.cos(alpha),
                vt * torch.sin(beta),
                vt * torch.cos(beta) * torch.sin(alpha),
            ),
            1,
        )
        transport = torch.linalg.cross(state[:, 9:12], uvw)
        return self._body_to_ned(uvwdot + transport, state[:, 3:6])

    def step(self, action):
        action = torch.as_tensor(action, dtype=self.dtype, device=self.device)
        # JSBSim integrates the already-computed forces, then executes FCS/aero
        # for the following frame. FCS feedback samples the frame-start state.
        x = self.state
        aero = self.pending_aero
        thrust = self.pending_thrust
        k1 = self.dynamics(x, aero, thrust, self.fuel_lbs)
        body_accel = self._body_acceleration(x, k1)
        nonrotating_accel = self._nonrotating_acceleration(x, k1)
        pre_obs = self.observation41()
        attitude_quaternion = self.eci_quaternion if self.use_eci else self.quaternion
        attitude_rates = self.pqri if self.use_eci else x[:, 9:12]
        qw, qx, qy, qz = attitude_quaternion.unbind(1)
        p_rate, q_rate, r_rate = attitude_rates.unbind(1)
        qdot = 0.5 * torch.stack(
            (
                -qx * p_rate - qy * q_rate - qz * r_rate,
                qw * p_rate + qy * r_rate - qz * q_rate,
                qw * q_rate - qx * r_rate + qz * p_rate,
                qw * r_rate + qx * q_rate - qy * p_rate,
            ),
            1,
        )
        propagated_quaternion = attitude_quaternion + self.dt * qdot
        propagated_quaternion = propagated_quaternion / torch.linalg.vector_norm(
            propagated_quaternion, dim=1, keepdim=True
        ).clamp_min(1e-9)
        propagated_attitude = (
            self._eci_attitude(propagated_quaternion, self.epa + self.dt * OMEGA, self.eci_position)
            if self.use_eci
            else quaternion_to_euler(propagated_quaternion)
        )
        delayed_pilot = torch.stack((self.pilot_ny, self.pilot_nz), 1)
        self.fcs_state, self.surfaces = self.fcs(
            self.fcs_state,
            action,
            pre_obs,
            pilot_loads=delayed_pilot,
            gravity_attitude=propagated_attitude,
        )
        self.engine_state = self.engine(self.engine_state, action[:, 3], pre_obs[:, 29], x[:, 2])
        self.fuel_lbs = (
            self.fuel_lbs - self.engine_state.fuel_flow_pph * self.dt / 3600.0
        ).clamp_min(0.0)
        if self.use_eci:
            # Exact FGPropagate sequence: ECI attitude/rate RectEuler, ECI
            # velocity AB2 and position AB3, then reconstruct ECEF/local state.
            r_b2i = quaternion_to_matrix(self.eci_quaternion)
            r_i2e = earth_rotation(self.epa)
            ecef = (r_i2e @ self.eci_position.unsqueeze(2)).squeeze(2)
            specific = self.dynamics.specific_force(x, aero, thrust, self.fuel_lbs)
            accel_i = (r_b2i.double() @ specific.double().unsqueeze(2)).squeeze(2) + (
                r_i2e.transpose(1, 2) @ gravity_j2(ecef).unsqueeze(2)
            ).squeeze(2)
            new_eci_position = (
                self.eci_position
                + self.dt
                * (
                    23 * self.eci_velocity
                    - 16 * self.eci_velocity_history[:, 0]
                    + 5 * self.eci_velocity_history[:, 1]
                )
                / 12
            )
            new_eci_velocity = self.eci_velocity + self.dt * (
                1.5 * accel_i - 0.5 * self.eci_accel_history[:, 0]
            )
            omega_i = x.new_tensor([0.0, 0.0, OMEGA]).expand_as(self.pqri)
            omega_b = (r_b2i.transpose(1, 2) @ omega_i.unsqueeze(2)).squeeze(2)
            pqri_dot = k1[:, 9:12] - torch.linalg.cross(self.pqri, omega_b)
            new_pqri = self.pqri + self.dt * pqri_dot
            new_epa = self.epa + self.dt * OMEGA
            r_i2e_new = earth_rotation(new_epa)
            new_ecef = (r_i2e_new @ new_eci_position.unsqueeze(2)).squeeze(2)
            lat, lon, alt = ecef_to_geocentric(new_ecef)
            r_b2i_new = quaternion_to_matrix(propagated_quaternion)
            r_b2e = r_i2e_new @ r_b2i_new.double()
            new_attitude = matrix_to_euler(ned_to_ecef_matrix(lat, lon).transpose(1, 2) @ r_b2e).to(
                x.dtype
            )
            relative_i = new_eci_velocity - torch.linalg.cross(omega_i.double(), new_eci_position)
            uvw = (
                (r_b2i_new.double().transpose(1, 2) @ relative_i.unsqueeze(2))
                .squeeze(2)
                .to(x.dtype)
            )
            new_vt = torch.linalg.vector_norm(uvw, dim=1).clamp_min(1.0)
            new_alpha = torch.atan2(uvw[:, 2], uvw[:, 0])
            new_beta = torch.asin((uvw[:, 1] / new_vt).clamp(-0.999, 0.999))
            new_rates = new_pqri - (r_b2i_new.transpose(1, 2) @ omega_i.unsqueeze(2)).squeeze(2)
            origin_lat = torch.full(
                (self.num_envs,),
                37.91455691666667 * torch.pi / 180,
                device=x.device,
                dtype=torch.float64,
            )
            origin_lon = torch.full(
                (self.num_envs,),
                128.18188127777776 * torch.pi / 180,
                device=x.device,
                dtype=torch.float64,
            )
            lat_obs, lon_obs, alt_obs = ecef_to_geocentric(new_ecef)
            obs_ecef = geodetic_to_ecef(lat_obs, lon_obs, alt_obs)
            origin = geodetic_to_ecef(origin_lat, origin_lon, torch.zeros_like(origin_lat))
            ned = (
                ned_to_ecef_matrix(origin_lat, origin_lon).transpose(1, 2)
                @ (obs_ecef - origin).unsqueeze(2)
            ).squeeze(2)
            self.release_position_m = (ned * FT_TO_M).to(x.dtype)
            new_position = torch.stack((ned[:, 0], ned[:, 1], alt), 1).to(x.dtype)
            self.eci_velocity_history = torch.stack(
                (self.eci_velocity, self.eci_velocity_history[:, 0]), 1
            )
            self.eci_accel_history = torch.stack((accel_i, self.eci_accel_history[:, 0]), 1)
            self.eci_position, self.eci_velocity, self.eci_quaternion, self.pqri, self.epa = (
                new_eci_position,
                new_eci_velocity,
                propagated_quaternion,
                new_pqri,
                new_epa,
            )
            self.quaternion = euler_to_quaternion(new_attitude)
        else:
            # Legacy local approximation retained for controlled A/B tests.
            pos_rate = (
                23 * k1[:, :3]
                - 16 * self.position_rate_history[:, 0]
                + 5 * self.position_rate_history[:, 1]
            ) / 12
            new_position = x[:, :3] + self.dt * pos_rate
            old_uvw = self._uvw(x)
            old_ned_velocity = self._body_to_ned(old_uvw, x[:, 3:6])
            ab2 = 1.5 * nonrotating_accel - 0.5 * self.body_accel_history[:, 0]
            new_ned_velocity = old_ned_velocity + self.dt * ab2
            uvw = self._ned_to_body(new_ned_velocity, propagated_attitude)
            new_vt = torch.linalg.vector_norm(uvw, dim=1).clamp_min(1.0)
            new_alpha = torch.atan2(uvw[:, 2], uvw[:, 0])
            new_beta = torch.asin((uvw[:, 1] / new_vt).clamp(-0.999, 0.999))
            self.quaternion = propagated_quaternion
            new_attitude = quaternion_to_euler(self.quaternion)
            new_rates = x[:, 9:12] + self.dt * k1[:, 9:12]
        self.state = torch.cat(
            (
                new_position,
                new_attitude,
                new_vt[:, None],
                new_alpha[:, None],
                new_beta[:, None],
                new_rates,
            ),
            1,
        )
        self.position_rate_history = torch.stack((k1[:, :3], self.position_rate_history[:, 0]), 1)
        self.body_accel_history = torch.stack((nonrotating_accel, self.body_accel_history[:, 0]), 1)
        self.state[:, 3:6] = torch.remainder(self.state[:, 3:6] + torch.pi, 2 * torch.pi) - torch.pi
        # Pilot load approximation used only as next-frame FCS feedback.
        # Convert (V,alpha,beta) derivatives back to body acceleration before
        # calculating accelerometer load factors at the CG.
        # FGAuxiliary Nx/Ny/Nz use vBodyAccel=Force/Mass, not UVWdot.
        # The latter also contains gravity, Earth-rate and centrifugal terms.
        specific = self.dynamics.specific_force(x, aero, thrust, self.fuel_lbs)
        self.ny = specific[:, 1] / 32.174
        self.nz = -specific[:, 2] / 32.174
        omega = x[:, 9:12]
        lever = omega.new_tensor([12.26102756, 0.0, -2.64083882]).expand_as(omega)
        rotational = torch.linalg.cross(k1[:, 9:12], lever) + torch.linalg.cross(
            omega, torch.linalg.cross(omega, lever)
        )
        self.pilot_ny = self.ny + rotational[:, 1] / 32.174
        self.pilot_nz = -self.nz + rotational[:, 2] / 32.174
        self.angular_accel = k1[:, 9:12]
        self.gear_pos = (self.gear_pos - self.dt / 5.0).clamp_min(0.0)
        self.pending_aero = (
            self.aero(airborne_properties(self.state, self.surfaces, gear_pos=self.gear_pos))
            * self.aero_scale
        )
        self.pending_thrust = self.engine_state.thrust_lbf * self.thrust_scale
        self.frame_index += 1
        return self.observation41()


def interp_mach(mach):
    return 1.0 - 0.85 * mach.clamp(0, 1)

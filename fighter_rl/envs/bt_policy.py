import torch

MODE_PURE = 0
MODE_LEAD = 1
MODE_BREAK = 2
MODE_RECOVER = 3
MODE_EMPTY = 4
MODE_LAG = 5
MODE_HIGH_YOYO = 6
MODE_BARREL_ROLL = 7
MODE_EXTEND = 8
MODE_LOW_YOYO = 9
MODE_CLIMB_EMERGENCY = 10
MODE_LAG_CLOSE = 11
MODE_NONE = 12

MODE_LABELS = {
    MODE_PURE: "pure",
    MODE_LEAD: "lead",
    MODE_BREAK: "break",
    MODE_RECOVER: "climb_recovery",
    MODE_EMPTY: "empty",
    MODE_LAG: "lag",
    MODE_HIGH_YOYO: "high_yoyo",
    MODE_BARREL_ROLL: "barrel_roll",
    MODE_EXTEND: "unloaded_extension",
    MODE_LOW_YOYO: "low_yoyo",
    MODE_CLIMB_EMERGENCY: "climb_emergency",
    MODE_LAG_CLOSE: "lag_close",
    MODE_NONE: "none",
}


def _unit(v, eps=1e-6):
    return v / v.norm(dim=1, keepdim=True).clamp_min(eps)


def _angle_deg(a, b):
    denom = a.norm(dim=1).clamp_min(1e-6) * b.norm(dim=1).clamp_min(1e-6)
    c = (a * b).sum(1) / denom

    return torch.rad2deg(torch.acos(c.clamp(-1.0, 1.0)))


def _cross(a, b):
    return torch.linalg.cross(a, b, dim=1)


def _pos_alt_up(obs41):
    return torch.stack((obs41[:, 0], obs41[:, 1], -obs41[:, 2]), 1)


def _uvw_ned(obs41):
    """Convert body-axis velocity columns [u, v, w] to local NED m/s."""
    phi = torch.deg2rad(obs41[:, 3])
    theta = torch.deg2rad(obs41[:, 4])
    psi = torch.deg2rad(obs41[:, 5])
    u, v, w = obs41[:, 6], obs41[:, 7], obs41[:, 8]
    sp, cp = torch.sin(phi), torch.cos(phi)
    st, ct = torch.sin(theta), torch.cos(theta)
    ss, cs = torch.sin(psi), torch.cos(psi)

    # fmt: off
    return torch.stack((
        u*ct*cs + v*(sp*st*cs - cp*ss) + w*(cp*st*cs + sp*ss),
        u*ct*ss + v*(sp*st*ss + cp*cs) + w*(cp*st*ss - sp*cs),
        -u*st + v*sp*ct + w*cp*ct,
    ), 1)
    # fmt: on


def _bt_forward_up_right(obs41):
    """Return the C++ BT forward/up/right vectors in [north, east, altitude-up].

    This intentionally mirrors ``DirectionVectorUpdate.cpp`` and
    ``Controller_CY.cpp`` instead of using a cleaner rotation-matrix helper.
    """
    yaw = torch.deg2rad(obs41[:, 5])
    pitch = torch.deg2rad(obs41[:, 4])
    roll = torch.deg2rad(obs41[:, 3])

    c1, s1 = torch.cos(yaw / 2.0), torch.sin(yaw / 2.0)
    c2, s2 = torch.cos(pitch / 2.0), torch.sin(pitch / 2.0)
    c3, s3 = torch.cos(roll / 2.0), torch.sin(roll / 2.0)
    c1c2 = c1 * c2
    s1s2 = s1 * s2

    qw = c1c2 * c3 + s1s2 * s3
    qx = c1 * s2 * c3 + s1 * c2 * s3
    qy = s1 * c2 * c3 - c1 * s2 * s3
    qz = c1 * c2 * s3 - s1 * s2 * c3
    qn = torch.sqrt(qw * qw + qx * qx + qy * qy + qz * qz).clamp_min(1e-9)
    qw, qx, qy, qz = qw / qn, qx / qn, qy / qn, qz / qn

    forward = torch.stack(
        (
            1.0 - 2.0 * (qx * qx + qy * qy),
            2.0 * (qx * qz + qw * qy),
            -2.0 * (qy * qz - qw * qx),
        ),
        1,
    )
    up = torch.stack(
        (
            -2.0 * (qy * qz + qw * qx),
            -2.0 * (qx * qy - qw * qz),
            1.0 - 2.0 * (qx * qx + qz * qz),
        ),
        1,
    )
    right = torch.stack(
        (
            2.0 * (qx * qz - qw * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
            -2.0 * (qx * qy + qw * qz),
        ),
        1,
    )

    return _unit(forward), _unit(up), _unit(right)


def make_bt_state(num_envs, device, dtype=torch.float32):
    """Create mutable BT state for cooldown/current-task behavior."""
    device = torch.device(device)

    return {
        "time": torch.zeros(num_envs, device=device, dtype=dtype),
        "current_mode": torch.full((num_envs,), MODE_NONE, device=device, dtype=torch.long),
        "last_high_yoyo": torch.full((num_envs,), -1.0e9, device=device, dtype=dtype),
        "last_barrel_roll": torch.full((num_envs,), -1.0e9, device=device, dtype=dtype),
        "last_low_yoyo": torch.full((num_envs,), -1.0e9, device=device, dtype=dtype),
    }


def _geometry(my_obs41, target_obs41):
    my_pos = _pos_alt_up(my_obs41)
    target_pos = _pos_alt_up(target_obs41)
    my_fwd, my_up, my_right = _bt_forward_up_right(my_obs41)
    target_fwd, target_up, target_right = _bt_forward_up_right(target_obs41)

    to_target = target_pos - my_pos
    distance = to_target.norm(dim=1).clamp_min(1e-6)
    to_target_u = to_target / distance[:, None]
    los = _angle_deg(my_fwd, to_target_u)
    target_los = _angle_deg(target_fwd, -to_target_u)
    angle_off = _angle_deg(my_fwd, target_fwd)

    target_to_my = my_pos - target_pos
    projection_h = (target_to_my * target_up).sum(1, keepdim=True)
    projected_my = my_pos - projection_h * target_up
    aspect_vec = projected_my - target_pos
    aspect = _angle_deg(aspect_vec, target_fwd)

    return {
        "my_pos": my_pos,
        "target_pos": target_pos,
        "my_fwd": my_fwd,
        "my_up": my_up,
        "my_right": my_right,
        "target_fwd": target_fwd,
        "target_up": target_up,
        "target_right": target_right,
        "to_target": to_target,
        "distance": distance,
        "los": los,
        "target_los": target_los,
        "angle_off": angle_off,
        "aspect": aspect,
        "my_speed": my_obs41[:, 27],
        "target_speed": target_obs41[:, 27],
        "altitude": my_pos[:, 2],
    }


def _cooldown_ok(
    state,
    previous_mode,
    now,
    maneuver_mode,
    last_key,
    cooldown_s,
    urgent=None,
):
    if state is None:
        return torch.ones_like(previous_mode, dtype=torch.bool)

    current = previous_mode == int(maneuver_mode)

    if urgent is None:
        urgent = torch.zeros_like(current)
    elapsed = now - state[last_key]
    never_used = state[last_key] < -1.0e8

    return current | urgent | never_used | (elapsed >= float(cooldown_s))


def _lead_vp(g):
    lead_distance = 0.25 * g["distance"] + 0.4 * g["target_speed"]
    lead_distance = torch.where(g["los"] > 20.0, lead_distance * 1.3, lead_distance)
    lead_distance = torch.where(g["los"] < 5.0, lead_distance * 0.8, lead_distance)
    lead_distance = lead_distance.clamp(500.0, 2000.0)

    return g["target_pos"] + g["target_fwd"] * lead_distance[:, None], torch.full_like(
        g["distance"], MODE_LEAD, dtype=torch.long
    )


def _lag_vp(g):
    lag_distance = 0.20 * g["distance"] + 0.30 * g["target_speed"]
    close = g["distance"] < 500.0
    lag_distance = torch.where(close, torch.full_like(lag_distance, 1500.0), lag_distance)
    lag_distance = torch.where(g["los"] > 20.0, lag_distance * 1.2, lag_distance)
    lag_distance = torch.where(g["los"] < 5.0, lag_distance * 0.8, lag_distance)
    lag_distance = lag_distance.clamp(300.0, 1800.0)
    mode = torch.where(
        close,
        torch.full_like(g["distance"], MODE_LAG_CLOSE, dtype=torch.long),
        torch.full_like(g["distance"], MODE_LAG, dtype=torch.long),
    )

    return g["target_pos"] - g["target_fwd"] * lag_distance[:, None], mode


def _pure_vp(g):
    return g["target_pos"], torch.full_like(g["distance"], MODE_PURE, dtype=torch.long)


def _climb_vp(g):
    altitude_error = (900.0 - g["altitude"]).clamp(300.0, 1200.0)
    emergency = g["altitude"] < 400.0
    forward_distance = torch.where(
        emergency, torch.full_like(g["distance"], 1200.0), torch.full_like(g["distance"], 1800.0)
    )
    vp = (
        g["my_pos"]
        + g["my_fwd"] * forward_distance[:, None]
        + torch.stack(
            (torch.zeros_like(altitude_error), torch.zeros_like(altitude_error), altitude_error),
            1,
        )
    )
    mode = torch.where(
        emergency,
        torch.full_like(g["distance"], MODE_CLIMB_EMERGENCY, dtype=torch.long),
        torch.full_like(g["distance"], MODE_RECOVER, dtype=torch.long),
    )

    return vp, mode


def _break_vp(g):
    direction = g["to_target"] / g["distance"][:, None].clamp_min(1e-6)
    right_comp = (direction * g["my_right"]).sum(1)
    break_sign = torch.where(
        right_comp >= 0.0, torch.ones_like(right_comp), -torch.ones_like(right_comp)
    )
    vp = (
        g["my_pos"]
        + g["my_fwd"] * 500.0
        + g["my_right"] * (break_sign * 6000.0)[:, None]
        + g["my_up"] * 300.0
    )

    return vp, torch.full_like(g["distance"], MODE_BREAK, dtype=torch.long)


def _high_yoyo_vp(g):
    vp = g["target_pos"] + g["target_up"] * 500.0 - g["target_fwd"] * 100.0

    return vp, torch.full_like(g["distance"], MODE_HIGH_YOYO, dtype=torch.long)


def _barrel_roll_vp(g):
    rel = g["my_pos"] - g["target_pos"]
    fwd_comp = (rel * g["target_fwd"]).sum(1, keepdim=True)
    perp = rel - g["target_fwd"] * fwd_comp
    perp_len = perp.norm(dim=1)
    fallback = g["my_up"] * 50.0
    perp = torch.where((perp_len < 50.0)[:, None], fallback, perp)
    perp_norm = _unit(perp)
    orbit_tangent = _cross(g["target_fwd"], perp_norm)
    vp = g["target_pos"] - g["target_fwd"] * 200.0 + perp_norm * 350.0 + orbit_tangent * 350.0

    return vp, torch.full_like(g["distance"], MODE_BARREL_ROLL, dtype=torch.long)


def _extend_vp(g):
    return g["my_pos"] + g["my_fwd"] * 10000.0, torch.full_like(
        g["distance"], MODE_EXTEND, dtype=torch.long
    )


def _low_yoyo_vp(g):
    dive_depth = torch.minimum(g["distance"] * 0.25, torch.full_like(g["distance"], 800.0))
    lead_dist = torch.minimum(g["distance"] * 0.25, torch.full_like(g["distance"], 1200.0))
    vp = (
        g["target_pos"]
        - g["target_up"] * dive_depth[:, None]
        + g["target_fwd"] * lead_dist[:, None]
    )

    return vp, torch.full_like(g["distance"], MODE_LOW_YOYO, dtype=torch.long)


def _apply_choice(
    selected,
    vp,
    mode,
    condition,
    candidate_vp,
    candidate_mode,
):
    take = (~selected) & condition
    vp = torch.where(take[:, None], candidate_vp, vp)
    mode = torch.where(take, candidate_mode, mode)

    return selected | take, vp, mode


def inha_viper_virtual_point(
    my_obs41,
    target_obs41,
    *,
    bt_state=None,
    dt=1.0 / 60.0,
    active_mask=None,
    **_,
):
    """Evaluate the INHA_VIPER selector and return VP plus diagnostic info."""
    g = _geometry(my_obs41, target_obs41)
    lag_vp, lag_mode = _lag_vp(g)
    vp = lag_vp
    mode = lag_mode
    selected = torch.zeros_like(g["distance"], dtype=torch.bool)

    if active_mask is None:
        active = torch.ones_like(selected)
    else:
        active = active_mask.to(device=my_obs41.device, dtype=torch.bool)

    previous_mode = (
        bt_state["current_mode"].to(device=my_obs41.device)
        if bt_state is not None
        else torch.full_like(mode, MODE_NONE)
    )
    previous_time = (
        bt_state["time"].to(device=my_obs41.device)
        if bt_state is not None
        else torch.zeros_like(g["distance"])
    )
    now = previous_time + float(dt)

    climb_vp, climb_mode = _climb_vp(g)
    selected, vp, mode = _apply_choice(
        selected, vp, mode, g["altitude"] < 600.0, climb_vp, climb_mode
    )

    break_cond = (
        (g["distance"] <= 4000.0)
        & (g["los"] >= 120.0)
        & (g["angle_off"] >= 60.0)
        & (g["aspect"] >= 100.0)
    )
    break_vp, break_mode = _break_vp(g)
    selected, vp, mode = _apply_choice(selected, vp, mode, break_cond, break_vp, break_mode)

    pure_vp, pure_mode = _pure_vp(g)
    weapon_cond = (
        (g["distance"] <= 914.0)
        & (g["los"] <= 5.0)
        & (g["angle_off"] <= 20.0)
        & (g["aspect"] <= 80.0)
    )
    selected, vp, mode = _apply_choice(selected, vp, mode, weapon_cond, pure_vp, pure_mode)
    selected, vp, mode = _apply_choice(
        selected, vp, mode, g["distance"] <= 914.0, pure_vp, pure_mode
    )

    high_ok = _cooldown_ok(
        bt_state,
        previous_mode,
        now,
        MODE_HIGH_YOYO,
        "last_high_yoyo",
        5.0,
        urgent=g["distance"] < 1000.0,
    )
    high_cond = high_ok & (g["distance"] <= 1500.0) & (g["distance"] >= 914.0) & (g["los"] >= 8.0)
    high_vp, high_mode = _high_yoyo_vp(g)
    selected, vp, mode = _apply_choice(selected, vp, mode, high_cond, high_vp, high_mode)

    barrel_ok = _cooldown_ok(
        bt_state, previous_mode, now, MODE_BARREL_ROLL, "last_barrel_roll", 8.0
    )
    barrel_cond = (
        barrel_ok & (g["distance"] <= 2000.0) & (g["distance"] >= 914.0) & (g["los"] >= 10.0)
    )
    barrel_vp, barrel_mode = _barrel_roll_vp(g)
    selected, vp, mode = _apply_choice(selected, vp, mode, barrel_cond, barrel_vp, barrel_mode)

    extend_vp, extend_mode = _extend_vp(g)
    extend_cond = (g["my_speed"] <= 150.0) & (g["distance"] >= 3000.0)
    selected, vp, mode = _apply_choice(selected, vp, mode, extend_cond, extend_vp, extend_mode)

    low_ok = _cooldown_ok(bt_state, previous_mode, now, MODE_LOW_YOYO, "last_low_yoyo", 5.0)
    low_cond = low_ok & (g["distance"] >= 2500.0) & (g["los"] <= 25.0) & (g["angle_off"] <= 45.0)
    low_vp, low_mode = _low_yoyo_vp(g)
    selected, vp, mode = _apply_choice(selected, vp, mode, low_cond, low_vp, low_mode)

    lead_vp, lead_mode = _lead_vp(g)
    selected, vp, mode = _apply_choice(
        selected, vp, mode, g["distance"] >= 3000.0, lead_vp, lead_mode
    )
    selected, vp, mode = _apply_choice(selected, vp, mode, g["los"] >= 15.0, lead_vp, lead_mode)
    selected, vp, mode = _apply_choice(
        selected, vp, mode, g["angle_off"] >= 50.0, lead_vp, lead_mode
    )

    if bt_state is not None:
        changed = active & (mode != previous_mode)
        bt_state["last_high_yoyo"] = torch.where(
            changed & (previous_mode == MODE_HIGH_YOYO),
            now,
            bt_state["last_high_yoyo"],
        )
        bt_state["last_barrel_roll"] = torch.where(
            changed & (previous_mode == MODE_BARREL_ROLL),
            now,
            bt_state["last_barrel_roll"],
        )
        bt_state["last_low_yoyo"] = torch.where(
            changed & (previous_mode == MODE_LOW_YOYO),
            now,
            bt_state["last_low_yoyo"],
        )
        bt_state["time"] = torch.where(active, now, bt_state["time"])
        bt_state["current_mode"] = torch.where(active, mode, bt_state["current_mode"])

    return vp, {
        "distance": g["distance"],
        "los": g["los"],
        "target_los": g["target_los"],
        "angle_off": g["angle_off"],
        "aspect": g["aspect"],
        "mode": mode,
        "vp": vp,
        "throttle_cmd": torch.ones_like(g["distance"]),
    }


def bt_virtual_point(
    my_obs41,
    target_obs41,
    **kwargs,
):
    return inha_viper_virtual_point(my_obs41, target_obs41, **kwargs)


def bt_empty_virtual_point(my_obs41):
    my_pos = _pos_alt_up(my_obs41)
    my_fwd, _, _ = _bt_forward_up_right(my_obs41)
    vp = my_pos + my_fwd * 10000.0
    mode = torch.full((my_obs41.shape[0],), MODE_EMPTY, device=my_obs41.device, dtype=torch.long)

    return vp, {"mode": mode, "vp": vp}


def vp_to_action(
    my_obs41,
    vp,
    *,
    target_speed_mps=280.0,
    throttle=1.0,
):
    """Convert a BT virtual point to roll/pitch/rudder/throttle.

    Roll, pitch, and rudder mirror the projection geometry in
    ``Geometry/Controller_CY.cpp``.  The C++ moving filters are intentionally
    omitted to keep batched GPU execution stateless except for BT cooldowns.
    """
    del target_speed_mps
    my_pos = _pos_alt_up(my_obs41)
    my_fwd, my_up, my_right = _bt_forward_up_right(my_obs41)
    target_location = vp
    to_vp = target_location - my_pos
    distance = to_vp.norm(dim=1).clamp_min(1e-6)
    los = torch.rad2deg(torch.acos((my_fwd * (to_vp / distance[:, None])).sum(1).clamp(-1.0, 1.0)))

    forward_point = my_fwd * 1000.0 + my_pos
    forward_point_to_vp = target_location - forward_point
    proj_v = (forward_point_to_vp * my_fwd).sum(1, keepdim=True) * my_fwd
    proj_p = target_location - proj_v
    proj_tv = proj_p - forward_point
    proj_len = proj_tv.norm(dim=1).clamp_min(1e-6)
    proj_u = proj_tv / proj_len[:, None]

    up_angle = torch.acos((my_up * proj_u).sum(1).clamp(-1.0, 1.0))
    side = torch.where((my_right * proj_u).sum(1) >= 0.0, 1.0, -1.0)
    ut_angle = up_angle * side
    ut_deg = torch.rad2deg(ut_angle)
    sin_ut = torch.sin(ut_angle)

    roll_close = torch.where(los > 3.0, sin_ut.clamp(-1.0, 1.0), sin_ut * los * -0.1)
    roll_near = sin_ut.clamp(-1.0, 1.0)
    roll_near = roll_near * roll_near.abs()
    roll = torch.where(ut_deg.abs() > 90.0, roll_close, roll_near)
    roll = torch.where(roll < 0.1, roll * 3.0, roll)
    roll = roll * los.clamp(0.0, 1.0)

    rudder = -sin_ut * los.clamp(0.0, 6.0)
    error_effect = (los / 6.0).clamp(0.0, 1.5)
    roll_effect = 1.0 - (ut_deg.abs() / 90.0).clamp(0.0, 1.0)
    horizon_effect = torch.where(ut_deg.abs() <= 90.0, 1.0, 0.5)
    pitch = torch.where(
        los < 90.0, -error_effect * roll_effect * horizon_effect, torch.full_like(los, -1.0)
    )

    a = torch.zeros(my_obs41.shape[0], 4, device=my_obs41.device, dtype=my_obs41.dtype)
    a[:, 0] = roll.clamp(-1.0, 1.0)
    a[:, 1] = pitch.clamp(-1.0, 1.0)
    a[:, 2] = rudder.clamp(-1.0, 1.0)
    throttle_t = torch.as_tensor(throttle, device=my_obs41.device, dtype=my_obs41.dtype)

    if throttle_t.ndim == 0:
        throttle_t = throttle_t.expand(my_obs41.shape[0])
    a[:, 3] = throttle_t.clamp(0.0, 1.0)

    return a


def point_los_to_action(my_obs41, vp_alt_up, *, target_speed_mps=270.0):
    """Compatibility wrapper; INHA_VIPER uses VP-to-stick control."""

    return vp_to_action(my_obs41, vp_alt_up, target_speed_mps=target_speed_mps)


def los_to_action(my_obs41, target_obs41, *, target_speed_mps=270.0):
    return vp_to_action(my_obs41, _pos_alt_up(target_obs41), target_speed_mps=target_speed_mps)


def inha_viper_action(
    my_obs41,
    target_obs41,
    *,
    bt_state=None,
    dt=1.0 / 60.0,
    active_mask=None,
):
    vp, info = inha_viper_virtual_point(
        my_obs41, target_obs41, bt_state=bt_state, dt=dt, active_mask=active_mask
    )
    action = vp_to_action(my_obs41, vp, throttle=1.0)
    info["vp"] = vp

    return action, info


def bt_action(
    my_obs41,
    target_obs41,
    *,
    bt_state=None,
    dt=1.0 / 60.0,
    active_mask=None,
):
    return inha_viper_action(
        my_obs41, target_obs41, bt_state=bt_state, dt=dt, active_mask=active_mask
    )


def bt_empty_action(my_obs41, target_obs41=None):
    del target_obs41
    vp, info = bt_empty_virtual_point(my_obs41)
    action = vp_to_action(my_obs41, vp, throttle=1.0)

    return action, info


__all__ = [
    "MODE_PURE",
    "MODE_LEAD",
    "MODE_BREAK",
    "MODE_RECOVER",
    "MODE_EMPTY",
    "MODE_LAG",
    "MODE_HIGH_YOYO",
    "MODE_BARREL_ROLL",
    "MODE_EXTEND",
    "MODE_LOW_YOYO",
    "MODE_CLIMB_EMERGENCY",
    "MODE_LAG_CLOSE",
    "MODE_NONE",
    "MODE_LABELS",
    "make_bt_state",
    "inha_viper_virtual_point",
    "inha_viper_action",
    "bt_virtual_point",
    "bt_empty_virtual_point",
    "bt_action",
    "bt_empty_action",
    "los_to_action",
    "point_los_to_action",
    "vp_to_action",
]

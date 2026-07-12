import torch
from fighter_rl.envs.batch import CompetitionBatchDogfight
from fighter_rl.envs.bt_policy import bt_action, bt_empty_action, make_bt_state
from fighter_rl.sim.neuralplane import CompetitionNeuralPlane
from fighter_rl.training.stages import LoiterStage


class CompetitionLoiterCurriculumEnv:
    base_observation_dim = 16
    action_dim = 4
    frame_dim = 20  # tactical16 + previous action4
    frames = 4
    observation_dim = 80

    def __init__(
        self,
        stage,
        num_envs=4096,
        device="cuda",
        hz=60,
        domain_randomization=True,
        target_maneuver="random_loiter",
        temporal_frames=4,
        include_previous_action=True,
    ):
        self.stage = stage
        self.n = int(num_envs)
        self.device = torch.device(device)
        self.hz = int(hz)
        self.domain_randomization = bool(domain_randomization)
        self.target_maneuver = str(target_maneuver)
        self.frames = max(1, int(temporal_frames))
        self.include_previous_action = bool(include_previous_action)
        self.frame_dim = self.base_observation_dim + (
            self.action_dim if self.include_previous_action else 0
        )
        self.observation_dim = self.frame_dim * self.frames
        self.own = CompetitionNeuralPlane(self.n, self.device, self.hz)
        self.target = CompetitionNeuralPlane(self.n, self.device, self.hz)
        self.steps = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.active = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self.own_health = torch.ones(self.n, device=self.device)
        self.target_health = torch.ones(self.n, device=self.device)
        self.history = torch.zeros(self.n, self.frames, self.frame_dim, device=self.device)
        self.cached_observation = torch.zeros(self.n, self.observation_dim, device=self.device)
        self.completed = []
        self.reset()

    def _uniform(self, values, default=0.0):
        if values is None:
            return torch.full((self.n,), float(default), device=self.device)

        if isinstance(values, (int, float)):
            return torch.full((self.n,), float(values), device=self.device)

        lo, hi = map(float, values)

        return torch.empty(self.n, device=self.device).uniform_(min(lo, hi), max(lo, hi))

    def _is_gun_curriculum(self):
        return (
            str(self.stage.reward.get("mode", "")).strip().lower() == "gun_curriculum"
            or str(self.stage.target_randomization.get("geometry_mode", "")).strip().lower()
            == "gun_curriculum"
        )

    def _sample_progressive_abs(self, values, default=0.0):
        """Sample positive ranges/angles with center and boundary emphasis.

        For the new gun curriculum, pure uniform sampling hides hard boundary
        cases.  This keeps 60% uniform, 20% easy-center, 20% hard-boundary by
        default, using per-stage fractions from target_randomization.
        """

        if values is None:
            return torch.full((self.n,), float(default), device=self.device)

        if isinstance(values, (int, float)):
            return torch.full((self.n,), abs(float(values)), device=self.device)

        lo, hi = sorted(map(float, values))

        if hi <= lo:
            return torch.full((self.n,), lo, device=self.device)

        cfg = self.stage.target_randomization

        easy = float(cfg.get("easy_fraction", 0.20))
        boundary = float(cfg.get("boundary_fraction", 0.20))

        u = torch.rand(self.n, device=self.device)

        uniform = torch.empty(self.n, device=self.device).uniform_(lo, hi)

        easy_hi = lo + (hi - lo) * 0.35
        easy_sample = torch.empty(self.n, device=self.device).uniform_(lo, easy_hi)

        hard_lo = hi - (hi - lo) * 0.25
        hard_sample = torch.empty(self.n, device=self.device).uniform_(hard_lo, hi)

        return torch.where(
            u < easy, easy_sample, torch.where(u < easy + boundary, hard_sample, uniform)
        )

    def _sample_progressive_abs_cfg(self, cfg, values, default=0.0, axis=None):
        if values is None:
            return torch.full((self.n,), float(default), device=self.device)

        if isinstance(values, (int, float)):
            return torch.full((self.n,), abs(float(values)), device=self.device)

        lo, hi = sorted(map(float, values))

        if hi <= lo:
            return torch.full((self.n,), lo, device=self.device)

        sampling = cfg.get("sampling", {}) if isinstance(cfg, dict) else {}
        axis_cfg = sampling.get(axis or "", {}) if isinstance(sampling, dict) else {}

        easy = float(axis_cfg.get("easy", cfg.get("easy_fraction", 0.20)))
        boundary = float(axis_cfg.get("boundary", cfg.get("boundary_fraction", 0.20)))

        easy = max(0.0, min(1.0, easy))
        boundary = max(0.0, min(1.0 - easy, boundary))

        u = torch.rand(self.n, device=self.device)

        uniform = torch.empty(self.n, device=self.device).uniform_(lo, hi)

        easy_hi = lo + (hi - lo) * float(axis_cfg.get("easy_span", 0.35))
        easy_sample = torch.empty(self.n, device=self.device).uniform_(lo, easy_hi)

        hard_span = float(axis_cfg.get("boundary_span", 0.25))
        hard_lo = hi - (hi - lo) * hard_span
        hard_sample = torch.empty(self.n, device=self.device).uniform_(hard_lo, hi)

        return torch.where(
            u < easy, easy_sample, torch.where(u < easy + boundary, hard_sample, uniform)
        )

    @staticmethod
    def _policy_mix_from_cfg(cfg):
        if not cfg:
            return None

        if "target_policy_mix" in cfg:
            return cfg.get("target_policy_mix")

        name = cfg.get("target_policy", cfg.get("policy", None))

        if name is None:
            return None

        return [{"policy": str(name), "weight": 1.0}]

    @staticmethod
    def _safe_bucket_name(name):
        text = str(name or "bucket").strip().lower()
        out = "".join(ch if ch.isalnum() else "_" for ch in text)
        out = "_".join(part for part in out.split("_") if part)

        return out or "bucket"

    def _bucket_mix(self, target_cfg):
        raw = target_cfg.get("bucket_mix") or []
        buckets = [dict(item) for item in raw if float(item.get("weight", 1.0)) > 0]

        if not buckets:
            return []

        total = sum(float(item.get("weight", 1.0)) for item in buckets)

        for i, item in enumerate(buckets):
            item.setdefault("name", f"bucket_{i}")
            item["weight"] = float(item.get("weight", 1.0)) / max(total, 1e-9)
        return buckets

    def _sample_bucket_ids(self, buckets):
        if not buckets:
            self.bucket_names = ["default"]

            return torch.zeros(self.n, dtype=torch.long, device=self.device)

        probs = torch.as_tensor(
            [float(item.get("weight", 1.0)) for item in buckets],
            dtype=torch.float32,
            device=self.device,
        )
        probs = probs / probs.sum()
        self.bucket_names = [
            self._safe_bucket_name(item.get("name", f"bucket_{i}"))
            for i, item in enumerate(buckets)
        ]

        return torch.multinomial(probs, self.n, replacement=True)

    def _bucket_uniform(self, target_cfg, buckets, bucket_ids, key, default=0.0):
        out = self._uniform(target_cfg.get(key), default)

        for i, bucket in enumerate(buckets):
            if key not in bucket:
                continue

            mask = bucket_ids == i

            if bool(mask.any()):
                out = torch.where(mask, self._uniform(bucket.get(key), default), out)
        return out

    def _bucket_abs(self, target_cfg, buckets, bucket_ids, key, default=0.0, axis=None):
        out = self._sample_progressive_abs_cfg(target_cfg, target_cfg.get(key), default, axis=axis)

        for i, bucket in enumerate(buckets):
            if key not in bucket:
                continue

            mask = bucket_ids == i

            if bool(mask.any()):
                out = torch.where(
                    mask,
                    self._sample_progressive_abs_cfg(bucket, bucket.get(key), default, axis=axis),
                    out,
                )
        return out

    def _bucket_bool(self, target_cfg, buckets, bucket_ids, key, default=True):
        out = torch.full(
            (self.n,), bool(target_cfg.get(key, default)), dtype=torch.bool, device=self.device
        )

        for i, bucket in enumerate(buckets):
            if key not in bucket:
                continue

            mask = bucket_ids == i

            if bool(mask.any()):
                out = torch.where(mask, torch.full_like(out, bool(bucket.get(key))), out)
        return out

    def _bucket_sign(self, buckets, bucket_ids, key):
        out = torch.where(torch.rand(self.n, device=self.device) < 0.5, -1.0, 1.0)

        for i, bucket in enumerate(buckets):
            if key not in bucket:
                continue

            mask = bucket_ids == i

            if bool(mask.any()):
                out = torch.where(mask, torch.full_like(out, float(bucket.get(key))), out)
        return out

    def _bucket_policy_ids(self, target_cfg, buckets, bucket_ids):
        ids = self._sample_policy_ids(
            self._policy_mix_from_cfg(target_cfg) or target_cfg.get("target_policy_mix")
        )

        for i, bucket in enumerate(buckets):
            mix = self._policy_mix_from_cfg(bucket)

            if not mix:
                continue

            mask = bucket_ids == i

            if bool(mask.any()):
                ids = torch.where(mask, self._sample_policy_ids(mix), ids)
        return ids

    def _apply_bucket_dv(self, target_cfg, buckets, bucket_ids, target_speed, own_speed):
        out = own_speed
        base_dv = target_cfg.get("dv_mps")

        if base_dv is not None:
            out = target_speed + self._uniform(base_dv, 0.0)
        for i, bucket in enumerate(buckets):
            if "dv_mps" not in bucket:
                continue

            mask = bucket_ids == i

            if bool(mask.any()):
                out = torch.where(
                    mask, target_speed + self._uniform(bucket.get("dv_mps"), 0.0), out
                )
        return out.clamp(180.0, 380.0)

    @staticmethod
    def _sync_release_position(plane, north_m, east_m, alt_m):
        if hasattr(plane, "set_local_position_m"):
            plane.set_local_position_m(north_m, east_m, alt_m)

            return

        if getattr(plane, "use_eci", False) and hasattr(plane, "release_position_m"):
            # fmt: off
            plane.release_position_m[:,0] = north_m
            plane.release_position_m[:,1] = east_m
            plane.release_position_m[:,2] = -alt_m
            # fmt: on

    def _sample_policy_ids(self, mix):
        mapping = {
            "straight": 0,
            "weak_turn": 1,
            "constant_turn": 2,
            "jink": 3,
            "defensive": 4,
            "shooter": 5,
            "bt": 6,
        }

        if not mix:
            return torch.zeros(self.n, dtype=torch.long, device=self.device)

        weights = []
        ids = []

        for item in mix:
            name = str(item.get("policy", "straight"))
            weight = float(item.get("weight", 1.0))

            if weight <= 0:
                continue

            ids.append(mapping.get(name, 0))
            weights.append(weight)
        if not weights:
            return torch.zeros(self.n, dtype=torch.long, device=self.device)

        probs = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        probs = probs / probs.sum()
        choice = torch.multinomial(probs, self.n, replacement=True)
        ids_t = torch.as_tensor(ids, dtype=torch.long, device=self.device)

        return ids_t[choice]

    def _initial_feasibility(self, distance, ata_abs, aa_abs, own_speed, target_speed):
        """Cheap initial-geometry sanity check for tail-chase reachability.

        This is not a full BFM solver.  It only answers whether the sampled
        initial geometry has at least a plausible radial closure into the
        competition WEZ within the stage time budget.  Without this guard the
        late curriculum can contain many long-range tail-chase samples where
        the target simply cannot be reached before timeout.
        """
        cfg = self.stage.target_randomization
        wez = self.stage.wez
        maximum = float(wez.get("max_range_m", 0.0) or 0.0)
        ata = torch.deg2rad(ata_abs)
        aa = torch.deg2rad(aa_abs)
        closing = own_speed * torch.cos(ata) - target_speed * torch.cos(aa)
        min_closing = float(cfg.get("min_initial_closing_mps", 8.0))
        time_budget = float(self.stage.max_engage_time) * float(
            cfg.get("max_time_to_wez_fraction", 0.80)
        )
        outside = (
            (distance > maximum) if maximum > 0 else torch.ones_like(distance, dtype=torch.bool)
        )
        safe_closing = torch.clamp(closing, min=1e-6)
        time_to = torch.where(
            outside, (distance - maximum).clamp_min(0.0) / safe_closing, torch.zeros_like(distance)
        )
        feasible = (~outside) | ((closing >= min_closing) & (time_to <= time_budget))
        opening = closing <= 0.0

        return closing, time_to, feasible, opening

    def reset(self):
        own_cfg = self.stage.ownship_randomization
        target_cfg = self.stage.target_randomization

        if self._is_gun_curriculum():
            buckets = self._bucket_mix(target_cfg)
            bucket_ids = self._sample_bucket_ids(buckets)

            self.init_bucket_id = bucket_ids

            def bucket_uniform(name, default):
                return self._bucket_uniform(target_cfg, buckets, bucket_ids, name, default)

            def bucket_abs(name, default, axis=None):
                return self._bucket_abs(target_cfg, buckets, bucket_ids, name, default, axis=axis)

            def bucket_bool(name, default):
                return self._bucket_bool(target_cfg, buckets, bucket_ids, name, default)

            def apply_bucket_dv(target_speed, own_speed):
                return self._apply_bucket_dv(
                    target_cfg, buckets, bucket_ids, target_speed, own_speed
                )

            self.init_bucket_require_feasible = bucket_bool("ensure_initial_feasible", True)

            own_alt = bucket_uniform("altitude_m", 7000.0)
            own_speed = bucket_uniform("own_speed_mps", 285.0)
            own_heading = bucket_uniform("own_heading_deg", 0.0)

            roll_limit = float(own_cfg.get("r_roll", 0))
            pitch_limit = float(own_cfg.get("r_pitch", 0))

            own_roll = self._uniform([-roll_limit, roll_limit])
            own_pitch = self._uniform([-pitch_limit, pitch_limit])

            distance = bucket_abs("distance_m", 700.0, axis="distance")
            ata_abs = bucket_abs("ata_deg", 0.0, axis="ata")
            aa_abs = bucket_abs("aa_tail_deg", 0.0, axis="aa_tail")

            target_speed = bucket_uniform("speed_mps", 270.0)
            own_speed = apply_bucket_dv(target_speed, own_speed)

            if bool(self.init_bucket_require_feasible.any()):
                attempts = max(0, int(target_cfg.get("feasible_resample_attempts", 10)))

                for _ in range(attempts):
                    closing, time_to, feasible, opening = self._initial_feasibility(
                        distance, ata_abs, aa_abs, own_speed, target_speed
                    )
                    bad = (~feasible) & self.init_bucket_require_feasible

                    if not bool(bad.any()):
                        break

                    new_distance = bucket_abs("distance_m", 700.0, axis="distance")
                    new_ata = bucket_abs("ata_deg", 0.0, axis="ata")
                    new_aa = bucket_abs("aa_tail_deg", 0.0, axis="aa_tail")

                    new_target_speed = bucket_uniform("speed_mps", 270.0)
                    new_own_speed = bucket_uniform("own_speed_mps", 285.0)
                    new_own_speed = apply_bucket_dv(new_target_speed, new_own_speed)

                    distance = torch.where(bad, new_distance, distance)
                    ata_abs = torch.where(bad, new_ata, ata_abs)
                    aa_abs = torch.where(bad, new_aa, aa_abs)
                    target_speed = torch.where(bad, new_target_speed, target_speed)
                    own_speed = torch.where(bad, new_own_speed, own_speed)

                closing, time_to, feasible, opening = self._initial_feasibility(
                    distance, ata_abs, aa_abs, own_speed, target_speed
                )
                bad = (~feasible) & self.init_bucket_require_feasible

                if bool(bad.any()):
                    wez_max = float(self.stage.wez.get("max_range_m", 914.4) or 914.4)
                    time_budget = float(self.stage.max_engage_time) * float(
                        target_cfg.get("max_time_to_wez_fraction", 0.80)
                    )
                    min_closing = float(target_cfg.get("min_initial_closing_mps", 8.0))
                    # Last-resort repair for the few samples that remain
                    # impossible after resampling: keep their angles, but pull
                    # them into a reachable band instead of leaving poisoned
                    # episodes in the batch.
                    reachable_limit = wez_max + closing.clamp_min(min_closing) * time_budget
                    distance = torch.where(bad, torch.minimum(distance, reachable_limit), distance)
                    distance = torch.where(
                        bad & (closing < min_closing),
                        torch.full_like(distance, wez_max * 0.95),
                        distance,
                    )
                    closing, time_to, feasible, opening = self._initial_feasibility(
                        distance, ata_abs, aa_abs, own_speed, target_speed
                    )
            closing, time_to, feasible, opening = self._initial_feasibility(
                distance, ata_abs, aa_abs, own_speed, target_speed
            )

            ata_sign = self._bucket_sign(buckets, bucket_ids, "ata_sign")
            aa_sign = self._bucket_sign(buckets, bucket_ids, "aa_sign")

            los_bearing = own_heading + ata_sign * ata_abs
            target_heading = torch.remainder(los_bearing - aa_sign * aa_abs, 360.0)

            altitude_offset = bucket_uniform("altitude_offset_m", 0.0)

            target_alt = (own_alt + altitude_offset).clamp(1200.0, 12000.0)
            target_roll = bucket_uniform("roll_deg", 0.0)
            target_pitch = bucket_uniform("pitch_deg", 0.0)

            self.own.reset(own_alt, own_speed, own_roll, own_pitch, own_heading)
            self.target.reset(target_alt, target_speed, target_roll, target_pitch, target_heading)

            own_n = torch.zeros(self.n, device=self.device)
            own_e = torch.zeros(self.n, device=self.device)
            bearing_rad = torch.deg2rad(los_bearing)

            # fmt: off
            self.own.state[:,0] = own_n/0.3048
            self.own.state[:,1] = own_e/0.3048
            self.target.state[:,0] = (own_n + distance*torch.cos(bearing_rad))/0.3048
            self.target.state[:,1] = (own_e + distance*torch.sin(bearing_rad))/0.3048
            # fmt: on
            self._sync_release_position(self.own, own_n, own_e, own_alt)
            self._sync_release_position(
                self.target,
                own_n + distance * torch.cos(bearing_rad),
                own_e + distance * torch.sin(bearing_rad),
                target_alt,
            )
            bank_abs = bucket_uniform("loiter_bank_abs_deg", 0.0)
            direction = (
                torch.where(torch.rand(self.n, device=self.device) < 0.5, -1.0, 1.0)
                if target_cfg.get("randomize_loiter_direction", True)
                else torch.full(
                    (self.n,), float(target_cfg.get("loiter_direction", 1)), device=self.device
                )
            )

            for i, bucket in enumerate(buckets):
                if "loiter_direction" not in bucket:
                    continue

                mask = bucket_ids == i

                if bool(mask.any()):
                    direction = torch.where(
                        mask,
                        torch.full_like(direction, float(bucket.get("loiter_direction"))),
                        direction,
                    )
            self.loiter_bank = direction * bank_abs.abs()
            self.target_policy_id = self._bucket_policy_ids(target_cfg, buckets, bucket_ids)
            self.init_distance = distance
            self.init_ata_abs = ata_abs
            self.init_aa_abs = aa_abs
            self.init_closing_mps = closing
            self.init_time_to_wez_s = time_to.clamp(0.0, 999.0)
            self.init_feasible = feasible
            self.init_opening = opening
        else:
            own_alt = torch.full((self.n,), 7000.0, device=self.device)
            own_speed = torch.full((self.n,), 300.0, device=self.device)
            own_roll = self._uniform(
                [-float(own_cfg.get("r_roll", 0)), float(own_cfg.get("r_roll", 0))]
            )
            own_pitch = self._uniform(
                [-float(own_cfg.get("r_pitch", 0)), float(own_cfg.get("r_pitch", 0))]
            )
            own_heading = self._uniform(
                [-float(own_cfg.get("r_heading", 0)), float(own_cfg.get("r_heading", 0))]
            )
            target_alt = self._uniform(target_cfg.get("altitude_m"), 7000)
            target_speed = self._uniform(target_cfg.get("speed_mps"), 260)
            target_roll = self._uniform(target_cfg.get("roll_deg"), 0)
            target_pitch = self._uniform(target_cfg.get("pitch_deg"), 0)
            target_heading = self._uniform(target_cfg.get("heading_deg"), 180)
            self.own.reset(own_alt, own_speed, own_roll, own_pitch, own_heading)
            self.target.reset(target_alt, target_speed, target_roll, target_pitch, target_heading)
            distance = self._uniform(target_cfg.get("range_m"), 5000)
            bearing = torch.deg2rad(self._uniform(target_cfg.get("bearing_deg"), 0))
            radius = float(own_cfg.get("radius", 0.0))
            # fmt: off
            own_r = torch.sqrt(torch.rand(self.n, device=self.device))*radius
            own_b = torch.rand(self.n, device=self.device)*2*torch.pi
            own_n = own_r*torch.cos(own_b)
            own_e = own_r*torch.sin(own_b)
            self.own.state[:,0] = own_n/0.3048
            self.own.state[:,1] = own_e/0.3048
            self.target.state[:,0] = (own_n + distance*torch.cos(bearing))/0.3048
            self.target.state[:,1] = (own_e + distance*torch.sin(bearing))/0.3048
            # fmt: on
            self._sync_release_position(self.own, own_n, own_e, own_alt)
            self._sync_release_position(
                self.target,
                own_n + distance * torch.cos(bearing),
                own_e + distance * torch.sin(bearing),
                target_alt,
            )
            bank_abs = self._uniform(target_cfg.get("loiter_bank_abs_deg"), 30)
            direction = (
                torch.where(torch.rand(self.n, device=self.device) < 0.5, -1.0, 1.0)
                if target_cfg.get("randomize_loiter_direction", True)
                else torch.full(
                    (self.n,), float(target_cfg.get("loiter_direction", 1)), device=self.device
                )
            )
            self.loiter_bank = direction * bank_abs.abs()
            self.target_policy_id = torch.zeros(self.n, dtype=torch.long, device=self.device)
            self.bucket_names = ["default"]
            self.init_bucket_id = torch.zeros(self.n, dtype=torch.long, device=self.device)
            self.init_bucket_require_feasible = torch.ones(
                self.n, dtype=torch.bool, device=self.device
            )
            self.init_distance = distance
            self.init_ata_abs = torch.zeros(self.n, device=self.device)
            self.init_aa_abs = torch.zeros(self.n, device=self.device)
            self.init_closing_mps = torch.zeros(self.n, device=self.device)
            self.init_time_to_wez_s = torch.zeros(self.n, device=self.device)
            self.init_feasible = torch.ones(self.n, dtype=torch.bool, device=self.device)
            self.init_opening = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.target_altitude = target_alt
        self.target_speed = target_speed
        # AIP's target is a feasible loiter/BT opponent.  For fast pretraining
        # we keep it safe (altitude/speed hold) but add slow per-env bank/speed
        # variation so the agent does not overfit a perfectly circular target.
        self.target_wave_amp = torch.empty(self.n, device=self.device).uniform_(0.0, 12.0)
        self.target_wave_period = torch.empty(self.n, device=self.device).uniform_(8.0, 24.0)
        self.target_wave_phase = torch.empty(self.n, device=self.device).uniform_(0.0, 2 * torch.pi)
        self.target_speed_wave = torch.empty(self.n, device=self.device).uniform_(-12.0, 12.0)

        if self.domain_randomization:
            self.own.randomize_model(0.01, 0.015, 0.005)
            self.target.randomize_model(0.01, 0.015, 0.005)
        else:
            self.own.aero_scale.fill_(1)
            self.target.aero_scale.fill_(1)
            self.own.thrust_scale.fill_(1)
            self.target.thrust_scale.fill_(1)
        self.target_bt_state = make_bt_state(self.n, self.device)
        self.steps.zero_()
        self.active.fill_(True)
        self.own_health.fill_(1)
        self.target_health.fill_(1)
        self.ep_return = torch.zeros(self.n, device=self.device)
        self.ep_min_distance = torch.full((self.n,), float("inf"), device=self.device)
        self.ep_distance_valid = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.ep_nonfinite_steps = torch.zeros(self.n, device=self.device)
        self.ep_wez_steps = torch.zeros(self.n, device=self.device)
        self.wez_streak = torch.zeros(self.n, device=self.device)
        self.ep_wez_streak_max = torch.zeros(self.n, device=self.device)
        self.ep_inner_violation = torch.zeros(self.n, device=self.device)
        self.ep_bad_3_9 = torch.zeros(self.n, device=self.device)
        self.ep_red_wez = torch.zeros(self.n, device=self.device)
        self.ep_target_damage = torch.zeros(self.n, device=self.device)
        self.ep_own_damage = torch.zeros(self.n, device=self.device)
        self.ep_track_score = torch.zeros(self.n, device=self.device)
        self.ep_overshoot = torch.zeros(self.n, device=self.device)
        self.ep_closure_violation = torch.zeros(self.n, device=self.device)
        self.ep_min_own_alt = torch.full((self.n,), float("inf"), device=self.device)
        self.ep_min_target_alt = torch.full((self.n,), float("inf"), device=self.device)
        o, t = self.own.observation41(), self.target.observation41()
        _, distance0, ata0, aa0, _, _ = self._geometry(o, t)
        self.prev_distance = torch.nan_to_num(distance0, nan=5000.0, posinf=5000.0, neginf=5000.0)
        self.prev_x_tgt = self._target_frame_x(o, t)
        self.prev_phi = self._gun_phi(
            self.prev_distance, torch.nan_to_num(ata0, nan=180.0), torch.nan_to_num(aa0, nan=180.0)
        )
        self.prev_policy_action = torch.zeros(self.n, 4, device=self.device)
        base = self._base_observation()
        frame = self._make_frame(base, torch.zeros(self.n, 4, device=self.device))
        # fmt: off
        self.history = frame[:,None,:].expand(-1, self.frames, -1).clone()
        # fmt: on
        self.cached_observation = self.history.reshape(self.n, -1)

        return self.cached_observation

    def _geometry(self, o, t):
        return CompetitionBatchDogfight._geometry(o, t)

    def _target_frame_x(self, o, t):
        x, _, _ = self._target_frame_rel(o, t)

        return x

    def _target_frame_rel(self, o, t):
        # fmt: off
        rel = o[:,:3] - t[:,:3]
        psi = torch.deg2rad(t[:,5])
        # fmt: on
        forward = torch.stack((torch.cos(psi), torch.sin(psi), torch.zeros_like(psi)), 1)
        right = torch.stack((-torch.sin(psi), torch.cos(psi), torch.zeros_like(psi)), 1)

        # fmt: off
        return (rel*forward).sum(1), (rel*right).sum(1), rel[:,2]
        # fmt: on

    def _gun_phi(self, distance, ata, aa):
        cfg = self.stage.reward
        theta = max(1.0, float(cfg.get("phi_ata_deg", 6.0)))
        center = float(cfg.get("phi_range_center_m", 650.0))
        sigma = max(1.0, float(cfg.get("phi_range_sigma_m", 450.0)))
        # fmt: off
        aim = torch.exp(-torch.square(ata.abs()/theta))
        range_score = torch.exp(-torch.square((distance - center)/sigma))
        aft = 0.5 + 0.5*(1 - aa.abs()/180.0).clamp(0, 1)
        # fmt: on

        return torch.nan_to_num(aim * range_score * aft, nan=0.0, posinf=0.0, neginf=0.0).clamp(
            0, 1
        )

    def _base_observation(self):
        o, t = self.own.observation41(), self.target.observation41()
        d, r, ata, aa, az, el = self._geometry(o, t)
        x = torch.empty(self.n, 16, device=self.device)

        # fmt: off
        x[:,0] = (o[:,3]/180).clamp(-1, 1)
        x[:,1] = (o[:,4]/90).clamp(-1, 1)
        x[:,2] = (torch.remainder(o[:,5], 360)/180 - 1).clamp(-1, 1)
        x[:,3] = (o[:,12]/300 - 1).clamp(-1, 1)
        x[:,4] = ((-o[:,2])/7500 - 1).clamp(-1, 1)
        x[:,5] = 2*self.own_health - 1
        x[:,6] = (d[:,0]/15000).clamp(-1, 1)
        x[:,7] = (d[:,1]/15000).clamp(-1, 1)
        x[:,8] = (d[:,2]/8000).clamp(-1, 1)
        x[:,9] = ata/180
        x[:,10] = aa/180
        x[:,11] = az/180
        x[:,12] = el/90
        x[:,13] = 2*self.target_health - 1
        # fmt: on

        wez = self.stage.wez
        inside = (
            (r >= float(wez["min_range_m"]))
            & (r <= float(wez["max_range_m"]))
            & (ata.abs() <= float(wez["angle_deg"]) / 2)
            if float(wez["max_range_m"]) > 0
            else torch.zeros_like(r, dtype=torch.bool)
        )

        # fmt: off
        x[:,14] = torch.where(inside, 1.0, -1.0)
        x[:,15] = 2*(1 - ata.abs()/30).clamp(0, 1)*(1 - r/3000).clamp(0, 1) - 1
        # fmt: on

        return torch.nan_to_num(x.clamp(-1, 1), nan=0.0, posinf=1.0, neginf=-1.0)

    def _make_frame(self, base, action):
        safe_action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)

        return torch.cat((base, safe_action), 1) if self.include_previous_action else base

    def _target_action(self):
        if self.target_maneuver == "bt":
            action, _ = bt_action(
                self.target.observation41(),
                self.own.observation41(),
                bt_state=getattr(self, "target_bt_state", None),
                dt=1.0 / self.hz,
            )

            return torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)

        if self.target_maneuver == "bt_empty":
            action, _ = bt_empty_action(self.target.observation41(), self.own.observation41())

            return torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)

        if self._is_gun_curriculum() and self.target_maneuver in (
            "random_loiter",
            "gun_curriculum",
            "stage_mix",
            "curriculum",
        ):
            obs = self.target.observation41()
            time = self.target.frame_index / self.hz
            pid = getattr(
                self, "target_policy_id", torch.zeros(self.n, dtype=torch.long, device=self.device)
            )
            bank_cmd = torch.zeros(self.n, device=self.device)
            straight = pid == 0
            weak = pid == 1
            constant = pid == 2
            jink = pid == 3
            defensive = pid == 4
            shooter = pid == 5
            bank_cmd = torch.where(weak, self.loiter_bank.clamp(-22, 22), bank_cmd)
            bank_cmd = torch.where(constant, self.loiter_bank.clamp(-50, 50), bank_cmd)
            # fmt: off
            jink_cmd = (
                self.loiter_bank
                + self.target_wave_amp
                * torch.sin(time/self.target_wave_period*2*torch.pi + self.target_wave_phase)
            ).clamp(-62, 62)
            # fmt: on
            bank_cmd = torch.where(jink, jink_cmd, bank_cmd)
            base_turn = self.loiter_bank.clamp(-45, 45)
            bank_cmd = torch.where(defensive | shooter, base_turn, bank_cmd)
            _, distance, _, _, _, _ = self._geometry(
                self.target.observation41(), self.own.observation41()
            )
            _, _, target_ata, _, az, _ = self._geometry(
                self.target.observation41(), self.own.observation41()
            )
            break_cmd = torch.where(az >= 0, torch.full_like(az, 55.0), torch.full_like(az, -55.0))
            nose_cmd = torch.where(az >= 0, torch.full_like(az, 35.0), torch.full_like(az, -35.0))
            bank_cmd = torch.where(defensive & (distance < 2200.0), break_cmd, bank_cmd)
            bank_cmd = torch.where(shooter & (target_ata > 3.0), nose_cmd, bank_cmd)
            # fmt: off
            speed_wave_cmd = (
                self.target_speed
                + self.target_speed_wave
                * torch.sin(
                    time/(self.target_wave_period*1.7)*2*torch.pi + self.target_wave_phase
                )
            ).clamp(210, 310)
            # fmt: on
            speed_cmd = torch.where(straight, self.target_speed.clamp(210, 310), speed_wave_cmd)
            # fmt: off
            bank_error = torch.remainder(bank_cmd - obs[:,3] + 180, 360) - 180
            # fmt: on
            a = torch.zeros(self.n, 4, device=self.device)

            # fmt: off
            a[:,0] = (0.035*bank_error - 0.012*obs[:,9]).clamp(-0.75, 0.75)
            altitude_error = (self.target_altitude - (-obs[:,2]))/1500
            a[:,1] = (-altitude_error + obs[:,4]/45).clamp(-0.45, 0.45)
            a[:,3] = (0.65 + (speed_cmd - obs[:,27])/150).clamp(0.25, 1.0)
            # fmt: on

            bt_mask = pid == 6

            if bool(bt_mask.any()):
                bt, _ = bt_action(
                    self.target.observation41(),
                    self.own.observation41(),
                    bt_state=getattr(self, "target_bt_state", None),
                    dt=1.0 / self.hz,
                    active_mask=bt_mask,
                )
                # fmt: off
                a = torch.where(bt_mask[:,None], bt, a)
                # fmt: on
            return torch.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)

        obs = self.target.observation41()
        # fmt: off
        bank_error = torch.remainder(self.loiter_bank - obs[:,3] + 180, 360) - 180
        # fmt: on

        if self.target_maneuver != "fixed_loiter":
            time = self.target.frame_index / self.hz
            # fmt: off
            bank_cmd = (
                self.loiter_bank
                + self.target_wave_amp
                * torch.sin(time/self.target_wave_period*2*torch.pi + self.target_wave_phase)
            ).clamp(-65, 65)
            speed_cmd = (
                self.target_speed
                + self.target_speed_wave
                * torch.sin(
                    time/(self.target_wave_period*1.7)*2*torch.pi + self.target_wave_phase
                )
            ).clamp(190, 310)
            # fmt: on
            # fmt: off
            bank_error = torch.remainder(bank_cmd - obs[:,3] + 180, 360) - 180
            # fmt: on
        else:
            speed_cmd = self.target_speed
        a = torch.zeros(self.n, 4, device=self.device)

        # fmt: off
        a[:,0] = (0.035*bank_error - 0.012*obs[:,9]).clamp(-0.65, 0.65)
        # F-16 pitch action is positive-forward/nose-down and negative-aft/up.
        altitude_error = (self.target_altitude - (-obs[:,2]))/1500
        a[:,1] = (-altitude_error + obs[:,4]/45).clamp(-0.40, 0.40)
        a[:,3] = (0.65 + (speed_cmd - obs[:,27])/150).clamp(0.25, 1.0)
        # fmt: on

        return torch.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)

    @staticmethod
    def _linear(angle, half):
        return (1 - angle.abs() / max(1.0, float(half))).clamp(0, 1)

    @staticmethod
    def _range(distance, maximum):
        return (1 - distance / max(1.0, float(maximum))).clamp(0, 1)

    @staticmethod
    def _control_zone(distance, limits):
        low, high = sorted(map(float, limits))
        below = (distance / max(1.0, low)).clamp(0, 1)
        above = (1 - (distance - high) / max(1.0, high)).clamp(0, 1)

        return torch.where(
            distance < low, below, torch.where(distance <= high, torch.ones_like(distance), above)
        )

    @staticmethod
    def _altitude_margin(altitude, floor, nominal):
        return ((altitude - float(floor)) / max(1.0, float(nominal) - float(floor))).clamp(0, 1)

    def _bad_distance_m(self):
        """Finite, bounded distance used only for invalid-episode summaries.

        A previous guard converted non-finite geometry to 1e9.  That protected
        PPO/SAC from NaNs, but the sentinel leaked into ep_min_distance and
        curriculum gates saw impossible million-metre ranges.  Keep the bad
        value finite and stage-local instead: large enough to fail distance
        gates, small enough to be recognizable as a bounded invalid metric.
        """
        target_range = self.stage.target_randomization.get(
            "distance_m", self.stage.target_randomization.get("range_m", 5000.0)
        )

        try:
            if isinstance(target_range, (list, tuple)) and len(target_range) >= 2:
                base = max(map(float, target_range[:2]))
            else:
                base = float(target_range)
        except Exception:
            base = 5000.0
        wez_max = float(self.stage.wez.get("max_range_m", 0.0) or 0.0)
        approach = float(self.stage.reward.get("approach_range_m", 10000.0) or 10000.0)

        return float(max(20000.0, base * 4.0, wez_max * 4.0, approach * 2.0))

    @torch.no_grad()
    def step(self, policy_action):
        valid = self.active.clone()
        a = torch.nan_to_num(
            torch.as_tensor(policy_action, dtype=torch.float32, device=self.device),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).clamp(-1, 1)
        sim = a.clone()
        # fmt: off
        sim[:,3] = (sim[:,3] + 1)/2
        # fmt: on

        total_own_damage = torch.zeros(self.n, device=self.device)
        total_target_damage = torch.zeros_like(total_own_damage)

        for _ in range(self.stage.step_ratio):
            self.own.step(sim)
            self.target.step(self._target_action())
            o, t = self.own.observation41(), self.target.observation41()
            _, distance, ata, _, _, _ = self._geometry(o, t)
            _, _, target_ata, _, _, _ = self._geometry(t, o)

            wez = self.stage.wez
            minimum, maximum, half = (
                float(wez["min_range_m"]),
                float(wez["max_range_m"]),
                float(wez["angle_deg"]) / 2,
            )

            if maximum > minimum:
                scale = ((maximum - distance) / (maximum - minimum)).clamp(0, 1) / self.hz
                in_range = (distance >= minimum) & (distance <= maximum)
                total_target_damage += torch.where(
                    valid & in_range & (ata.abs() <= half), scale, torch.zeros_like(scale)
                )
                total_own_damage += torch.where(
                    valid & in_range & (target_ata.abs() <= half), scale, torch.zeros_like(scale)
                )

        self.target_health = (self.target_health - total_target_damage).clamp_min(0)
        self.own_health = (self.own_health - total_own_damage).clamp_min(0)
        self.steps += valid.long()

        o, t = self.own.observation41(), self.target.observation41()
        _, distance, ata, aa, _, _ = self._geometry(o, t)
        _, _, target_ata, _, _, _ = self._geometry(t, o)

        cfg = self.stage.reward

        finite_state = (
            torch.isfinite(o).all(1)
            & torch.isfinite(t).all(1)
            & torch.isfinite(distance)
            & torch.isfinite(ata)
            & torch.isfinite(aa)
        )
        bad_distance = float(self._bad_distance_m())
        distance_ok = valid & finite_state & (distance >= 0)

        distance_s = torch.where(
            distance_ok, distance.clamp(0.0, bad_distance), torch.full_like(distance, bad_distance)
        )

        ata_s = torch.nan_to_num(ata, nan=180.0, posinf=180.0, neginf=-180.0)
        aa_s = torch.nan_to_num(aa, nan=180.0, posinf=180.0, neginf=-180.0)
        target_ata_s = torch.nan_to_num(target_ata, nan=180.0, posinf=180.0, neginf=-180.0)

        # fmt: off
        own_alt = -o[:,2]
        target_alt = -t[:,2]
        # fmt: on

        own_alt_s = torch.nan_to_num(own_alt, nan=0.0, posinf=20000.0, neginf=0.0)
        target_alt_s = torch.nan_to_num(target_alt, nan=0.0, posinf=20000.0, neginf=0.0)

        if self._is_gun_curriculum():
            wez = self.stage.wez
            minimum, maximum, half = (
                float(wez["min_range_m"]),
                float(wez["max_range_m"]),
                float(wez["angle_deg"]) / 2,
            )

            inside = (
                (maximum > minimum)
                & distance_ok
                & (distance_s >= minimum)
                & (distance_s <= maximum)
                & (ata_s.abs() <= half)
            )

            red_inside = (
                (maximum > minimum)
                & distance_ok
                & (distance_s >= minimum)
                & (distance_s <= maximum)
                & (target_ata_s.abs() <= half)
            )
            self.wez_streak = torch.where(
                inside, self.wez_streak + valid.float(), torch.zeros_like(self.wez_streak)
            )
            self.ep_wez_streak_max = torch.maximum(self.ep_wez_streak_max, self.wez_streak)

            dt = max(1e-6, float(self.stage.step_ratio) / float(self.hz))
            closing = (self.prev_distance - distance_s) / dt

            x_tgt, y_tgt, z_tgt = self._target_frame_rel(o, t)
            phi_now = self._gun_phi(distance_s, ata_s, aa_s)

            reward = torch.full(
                (self.n,), float(cfg.get("step_penalty", -0.002)), device=self.device
            )

            # fmt: off
            reward += float(cfg.get("damage_scale", 12.0))*total_target_damage
            reward -= float(cfg.get("own_damage_scale", 18.0))*total_own_damage

            cap = max(1.0, float(cfg.get("dwell_cap_steps", 20.0)))
            reward += (
                float(cfg.get("dwell_scale", 0.03))
                *(self.wez_streak.clamp(0, cap)/cap)
                *inside.float()
            )
            reward += float(cfg.get("phi_scale", 0.0)) * (
                float(cfg.get("phi_gamma", 0.99))*phi_now - self.prev_phi
            )
            # fmt: on

            aim_scale = float(cfg.get("aim_scale", 0.0))

            if aim_scale > 0.0:
                aim_sigma = max(1.0, float(cfg.get("aim_sigma_deg", 5.0)))
                aim_range_center = float(
                    cfg.get("aim_range_center_m", cfg.get("phi_range_center_m", 650.0))
                )
                aim_range_sigma = max(
                    1.0, float(cfg.get("aim_range_sigma_m", cfg.get("phi_range_sigma_m", 450.0)))
                )
                # fmt: off
                aim_score = torch.exp(
                    -torch.square(ata_s.abs()/aim_sigma)
                    - torch.square((distance_s - aim_range_center)/aim_range_sigma)
                )
                reward += aim_scale*aim_score*distance_ok.float()
                # fmt: on

            inner_soft = float(cfg.get("inner_soft_m", 300.0))
            inner_violation = distance_ok & (distance_s < inner_soft)
            # fmt: off
            inner_term = (
                (inner_soft - distance_s).clamp_min(0.0)/max(1.0, inner_soft)
            ).square()*(1.0 + (closing.clamp_min(0.0)/150.0))
            reward -= float(cfg.get("inner_penalty_scale", 0.65))*inner_term

            hard_collision = distance_ok & (distance_s < float(cfg.get("hard_collision_m", 130.0)))
            reward -= float(cfg.get("hard_collision_penalty", 8.0))*hard_collision.float()
            # fmt: on

            crossed = (self.prev_x_tgt < -50.0) & (x_tgt > 50.0)
            bad_3_9 = (
                crossed
                & (distance_s < float(cfg.get("bad_3_9_range_m", 1200.0)))
                & (closing > 0.0)
                & (ata_s.abs() > float(cfg.get("bad_3_9_ata_deg", 3.0)))
                & (total_target_damage < 1e-4)
            )
            reward -= float(cfg.get("bad_3_9_penalty", 0.55)) * bad_3_9.float()

            ahead_no_aim = (
                (x_tgt > float(cfg.get("ahead_no_aim_x_m", 100.0)))
                & (distance_s < float(cfg.get("ahead_no_aim_range_m", 1500.0)))
                & (ata_s.abs() > float(cfg.get("ahead_no_aim_ata_deg", 5.0)))
                & (~inside)
            )
            # fmt: off
            reward -= float(cfg.get("ahead_no_aim_penalty", 0.025))*ahead_no_aim.float()

            reward -= float(cfg.get("red_wez_penalty", 0.08))*red_inside.float()
            reward -= float(cfg.get("low_altitude_penalty", 2.0))*(
                (own_alt_s < float(cfg.get("low_altitude_m", 1000.0))).float()
            )
            reward -= float(cfg.get("action_rate_penalty", 0.001))*(
                a - self.prev_policy_action
            ).square().sum(1)

            reward += float(cfg.get("altitude_margin_scale", 0))*self._altitude_margin(
                own_alt_s, cfg.get("altitude_floor_m", 1800), cfg.get("altitude_nominal_m", 7000)
            )
            # fmt: on

            track_scale = float(cfg.get("track_scale", 0.0))

            if track_scale > 0.0:
                trail = float(cfg.get("track_trail_m", 900.0))
                sx = max(1.0, float(cfg.get("track_x_sigma_m", 450.0)))
                sy = max(1.0, float(cfg.get("track_y_sigma_m", 300.0)))
                sz = max(1.0, float(cfg.get("track_z_sigma_m", 250.0)))

                # fmt: off
                track_score = torch.exp(
                    -torch.square((x_tgt + trail)/sx)
                    - torch.square(y_tgt/sy)
                    - torch.square(z_tgt/sz)
                )
                # fmt: on
                track_score = torch.nan_to_num(track_score, nan=0.0, posinf=0.0, neginf=0.0).clamp(
                    0, 1
                )

                closure_limit = float(cfg.get("track_closure_limit_mps", 80.0))
                closure_sigma = max(1.0, float(cfg.get("track_closure_sigma_mps", 45.0)))

                closure_violation = distance_ok & (closing.abs() > closure_limit)
                overshoot = distance_ok & (x_tgt > float(cfg.get("track_overshoot_x_m", -80.0)))
                too_close = distance_ok & (distance_s < float(cfg.get("track_too_close_m", 260.0)))

                # fmt: off
                reward += track_scale*track_score
                reward -= float(cfg.get("track_closure_penalty", 0.03))*torch.square(
                    (closing.abs() - closure_limit).clamp_min(0.0)/closure_sigma
                )
                reward -= float(cfg.get("track_overshoot_penalty", 0.08))*overshoot.float()
                reward -= float(cfg.get("track_too_close_penalty", 0.20))*too_close.float()
                # fmt: on
            else:
                track_score = torch.zeros(self.n, device=self.device)
                closure_violation = torch.zeros(self.n, dtype=torch.bool, device=self.device)
                overshoot = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        else:
            alignment = self._linear(ata_s, cfg.get("alignment_half_angle_deg", 90))
            approach = self._range(distance_s, cfg.get("approach_range_m", 10000))
            rear = self._linear(aa_s, cfg.get("rear_half_angle_deg", 60))
            zone = self._control_zone(distance_s, cfg.get("control_zone_range_m", [500, 2500]))
            reward = torch.full(
                (self.n,),
                float(cfg.get("survival_bonus", 0)) + float(cfg.get("step_penalty", 0)),
                device=self.device,
            )
            # fmt: off
            reward += float(cfg.get("altitude_margin_scale", 0))*self._altitude_margin(
                own_alt_s, cfg.get("altitude_floor_m", 1200), cfg.get("altitude_nominal_m", 7000)
            )
            reward += (
                float(cfg.get("alignment_scale", 0))*alignment
                + float(cfg.get("approach_scale", 0))*alignment*approach
                + float(cfg.get("rear_scale", 0))*rear
                + float(cfg.get("control_zone_scale", 0))*alignment*rear*zone
                + float(cfg.get("damage_scale", 0))*(total_target_damage - total_own_damage)
            )
            reward -= float(cfg.get("low_altitude_penalty", 1))*(
                (own_alt_s < float(cfg.get("low_altitude_m", 800))).float()
            )
            # fmt: on
            hard_collision = torch.zeros(self.n, dtype=torch.bool, device=self.device)
            inner_violation = torch.zeros_like(hard_collision)
            bad_3_9 = torch.zeros_like(hard_collision)
            red_inside = torch.zeros_like(hard_collision)
            track_score = torch.zeros(self.n, device=self.device)
            closure_violation = torch.zeros_like(hard_collision)
            overshoot = torch.zeros_like(hard_collision)
            phi_now = getattr(self, "prev_phi", torch.zeros(self.n, device=self.device))
            x_tgt = getattr(self, "prev_x_tgt", torch.zeros(self.n, device=self.device))
        nonfinite = valid & (~finite_state | (~torch.isfinite(reward)))
        reward = torch.nan_to_num(
            reward,
            nan=float(cfg.get("loss_reward", -100)),
            posinf=float(cfg.get("loss_reward", -100)),
            neginf=float(cfg.get("loss_reward", -100)),
        )
        own_crash = (own_alt_s < 300) | nonfinite
        target_crash = target_alt_s < 300
        crash = own_crash | target_crash
        target_crash_without_damage = target_crash & (
            (self.ep_target_damage + total_target_damage)
            < float(cfg.get("target_crash_valid_damage_window", 0.05))
        )

        if self._is_gun_curriculum():
            # fmt: off
            reward -= (
                float(cfg.get("target_crash_without_damage_penalty", 6.0))
                *target_crash_without_damage.float()
            )
            # fmt: on
        unsafe_loss = hard_collision & bool(cfg.get("hard_collision_terminate", False))
        win = (self.target_health <= 0) & (self.own_health > 0)
        loss = (self.own_health <= 0) | own_crash | unsafe_loss
        other_terminal = target_crash
        terminated = valid & (win | loss | other_terminal)
        truncated = valid & (self.steps >= self.stage.decision_limit)
        done = terminated | truncated
        terminal = torch.where(
            win,
            float(cfg.get("win_reward", 100)),
            torch.where(
                loss, float(cfg.get("loss_reward", -100)), float(cfg.get("draw_reward", -30))
            ),
        )
        reward += torch.where(terminated, terminal, torch.zeros_like(reward))
        reward = torch.where(valid, reward, torch.zeros_like(reward))
        reward = torch.nan_to_num(
            reward,
            nan=float(cfg.get("loss_reward", -100)),
            posinf=float(cfg.get("loss_reward", -100)),
            neginf=float(cfg.get("loss_reward", -100)),
        )
        self.ep_return = torch.nan_to_num(self.ep_return, nan=0.0, posinf=0.0, neginf=0.0) + reward
        self.ep_min_distance = torch.minimum(
            self.ep_min_distance, torch.where(distance_ok, distance_s, self.ep_min_distance)
        )
        self.ep_distance_valid |= distance_ok
        self.ep_nonfinite_steps += nonfinite.float() * valid.float()
        self.ep_min_own_alt = torch.minimum(self.ep_min_own_alt, own_alt_s)
        self.ep_min_target_alt = torch.minimum(self.ep_min_target_alt, target_alt_s)
        inside = (
            (self.stage.wez["max_range_m"] > 0)
            & distance_ok
            & (distance_s >= float(self.stage.wez["min_range_m"]))
            & (distance_s <= float(self.stage.wez["max_range_m"]))
            & (ata_s.abs() <= float(self.stage.wez["angle_deg"]) / 2)
        )
        self.ep_wez_steps += inside.float() * valid
        self.ep_inner_violation += (inner_violation & valid).float()
        self.ep_bad_3_9 += (bad_3_9 & valid).float()
        self.ep_red_wez += (red_inside & valid).float()
        self.ep_target_damage += total_target_damage * valid.float()
        self.ep_own_damage += total_own_damage * valid.float()
        self.ep_track_score += track_score * valid.float()
        self.ep_overshoot += (overshoot & valid).float()
        self.ep_closure_violation += (closure_violation & valid).float()
        self.prev_distance = torch.where(valid, distance_s, self.prev_distance)
        self.prev_x_tgt = torch.where(valid, x_tgt, self.prev_x_tgt)
        self.prev_phi = torch.where(valid, phi_now, self.prev_phi)
        # fmt: off
        self.prev_policy_action = torch.where(valid[:,None], a, self.prev_policy_action)
        # fmt: on

        if done.any():
            m = done
            count = int(m.sum())
            own_low = self.ep_min_own_alt < 1500
            target_low = self.ep_min_target_alt < 1500
            safe_ep_min_distance = torch.where(
                self.ep_distance_valid,
                self.ep_min_distance,
                torch.full_like(self.ep_min_distance, bad_distance),
            )
            nonfinite_episode = self.ep_nonfinite_steps > 0
            episode_steps = self.steps.float().clamp_min(1.0)
            record = {
                "episodes": count,
                "win": float(win[m].float().sum()),
                "timeout": float(truncated[m].float().sum()),
                "crash": float(crash[m].float().sum()),
                "own_crash": float(own_crash[m].float().sum()),
                "target_crash": float(target_crash[m].float().sum()),
                "target_crash_without_damage": float(target_crash_without_damage[m].float().sum()),
                "own_low_alt": float(own_low[m].float().sum()),
                "target_low_alt": float(target_low[m].float().sum()),
                "nonfinite": float(nonfinite_episode[m].float().sum()),
                "distance_valid": float(self.ep_distance_valid[m].float().sum()),
                "ep_wez_steps": float(self.ep_wez_steps[m].sum()),
                "ep_wez_streak_max": float(self.ep_wez_streak_max[m].sum()),
                "inner_violation": float((self.ep_inner_violation[m] > 0).float().sum()),
                "bad_3_9": float((self.ep_bad_3_9[m] > 0).float().sum()),
                "red_wez": float((self.ep_red_wez[m] > 0).float().sum()),
                "target_damage": float(self.ep_target_damage[m].sum()),
                "own_damage": float(self.ep_own_damage[m].sum()),
                "track_score": float((self.ep_track_score[m] / episode_steps[m]).sum()),
                "overshoot": float((self.ep_overshoot[m] > 0).float().sum()),
                "closure_violation": float((self.ep_closure_violation[m] > 0).float().sum()),
                "ep_min_distance": float(safe_ep_min_distance[m].sum()),
                "ep_min_own_alt": float(self.ep_min_own_alt[m].sum()),
                "ep_min_target_alt": float(self.ep_min_target_alt[m].sum()),
                "final_ata_deg": float(ata_s[m].abs().sum()),
                "final_aa_deg": float(aa_s[m].abs().sum()),
                "init_distance_m": float(self.init_distance[m].sum()),
                "init_ata_deg": float(self.init_ata_abs[m].sum()),
                "init_aa_deg": float(self.init_aa_abs[m].sum()),
                "init_closing_mps": float(self.init_closing_mps[m].sum()),
                "init_time_to_wez_s": float(self.init_time_to_wez_s[m].sum()),
                "init_feasible": float(self.init_feasible[m].float().sum()),
                "initial_opening": float(self.init_opening[m].float().sum()),
                "return": float(self.ep_return[m].sum()),
            }
            bucket_metrics = {}
            bucket_ids = getattr(self, "init_bucket_id", None)
            bucket_names = getattr(self, "bucket_names", ["default"])

            if bucket_ids is not None:
                for idx, name in enumerate(bucket_names):
                    bm = m & (bucket_ids == idx)
                    bcount = int(bm.sum())

                    if bcount <= 0:
                        continue

                    bsteps = episode_steps[bm]
                    bsafe_distance = torch.where(
                        self.ep_distance_valid,
                        self.ep_min_distance,
                        torch.full_like(self.ep_min_distance, bad_distance),
                    )
                    bucket_metrics[name] = {
                        "episodes": float(bcount),
                        "win": float(win[bm].float().sum()),
                        "timeout": float(truncated[bm].float().sum()),
                        "crash": float(crash[bm].float().sum()),
                        "own_crash": float(own_crash[bm].float().sum()),
                        "target_crash": float(target_crash[bm].float().sum()),
                        "target_crash_without_damage": float(
                            target_crash_without_damage[bm].float().sum()
                        ),
                        "nonfinite": float(nonfinite_episode[bm].float().sum()),
                        "distance_valid": float(self.ep_distance_valid[bm].float().sum()),
                        "ep_wez_steps": float(self.ep_wez_steps[bm].sum()),
                        "ep_wez_streak_max": float(self.ep_wez_streak_max[bm].sum()),
                        "target_damage": float(self.ep_target_damage[bm].sum()),
                        "own_damage": float(self.ep_own_damage[bm].sum()),
                        "track_score": float((self.ep_track_score[bm] / bsteps).sum()),
                        "overshoot": float((self.ep_overshoot[bm] > 0).float().sum()),
                        "closure_violation": float(
                            (self.ep_closure_violation[bm] > 0).float().sum()
                        ),
                        "inner_violation": float((self.ep_inner_violation[bm] > 0).float().sum()),
                        "bad_3_9": float((self.ep_bad_3_9[bm] > 0).float().sum()),
                        "red_wez": float((self.ep_red_wez[bm] > 0).float().sum()),
                        "ep_min_distance": float(bsafe_distance[bm].sum()),
                        "final_ata_deg": float(ata_s[bm].abs().sum()),
                        "final_aa_deg": float(aa_s[bm].abs().sum()),
                        "init_distance_m": float(self.init_distance[bm].sum()),
                        "init_ata_deg": float(self.init_ata_abs[bm].sum()),
                        "init_aa_deg": float(self.init_aa_abs[bm].sum()),
                        "init_closing_mps": float(self.init_closing_mps[bm].sum()),
                        "init_time_to_wez_s": float(self.init_time_to_wez_s[bm].sum()),
                        "init_feasible": float(self.init_feasible[bm].float().sum()),
                        "initial_opening": float(self.init_opening[bm].float().sum()),
                        "return": float(self.ep_return[bm].sum()),
                    }
            if bucket_metrics:
                record["bucket_metrics"] = bucket_metrics
            self.completed.append(record)
        self.active &= ~done
        base = self._base_observation()
        new_frame = self._make_frame(base, a)
        # fmt: off
        new_history = torch.cat((self.history[:,1:], new_frame[:,None,:]), 1)
        self.history = torch.where(valid[:,None,None], new_history, self.history)
        # fmt: on
        obs = self.history.reshape(self.n, -1)
        # fmt: off
        self.cached_observation = torch.where(valid[:,None], obs, self.cached_observation)
        # fmt: on

        return (
            self.cached_observation,
            reward,
            done,
            {
                "valid": valid,
                "active": self.active,
                "terminated": terminated,
                "truncated": truncated,
                "win": win,
                "loss": loss,
                "crash": crash,
                "own_crash": own_crash,
                "target_crash": target_crash,
                "distance_m": distance_s,
                "ata_deg": ata_s,
                "aa_deg": aa_s,
                "ep_min_distance": torch.where(
                    torch.isfinite(self.ep_min_distance),
                    self.ep_min_distance,
                    torch.full_like(self.ep_min_distance, bad_distance),
                ),
                "distance_valid": self.ep_distance_valid,
                "nonfinite": nonfinite,
                "ep_wez_steps": self.ep_wez_steps,
                "ep_min_own_alt": self.ep_min_own_alt,
                "ep_min_target_alt": self.ep_min_target_alt,
                "ep_return": self.ep_return,
                "own_health": self.own_health,
                "target_health": self.target_health,
                "bucket_id": getattr(
                    self,
                    "init_bucket_id",
                    torch.zeros(self.n, dtype=torch.long, device=self.device),
                ),
            },
        )

    def pop_completed_summary(self):
        if not self.completed:
            return None

        records = self.completed
        self.completed = []
        bucket_totals = {}
        numeric_records = []

        for record in records:
            numeric_records.append({k: v for k, v in record.items() if k != "bucket_metrics"})

            for name, metrics in record.get("bucket_metrics", {}).items():
                dst = bucket_totals.setdefault(name, {})

                for key, value in metrics.items():
                    dst[key] = dst.get(key, 0.0) + float(value)
        keys = set().union(*(x.keys() for x in numeric_records))
        total = {k: sum(x.get(k, 0.0) for x in numeric_records) for k in keys}
        count = max(1, total.pop("episodes", 0.0))
        summary = {k: v / count for k, v in total.items()} | {
            "episodes": count,
            "win_rate": total.get("win", 0.0) / count,
            "timeout_rate": total.get("timeout", 0.0) / count,
            "crash_rate": total.get("crash", 0.0) / count,
            "own_crash_rate": total.get("own_crash", 0.0) / count,
            "target_crash_rate": total.get("target_crash", 0.0) / count,
            "target_crash_without_damage_rate": total.get("target_crash_without_damage", 0.0)
            / count,
            "own_low_alt_rate": total.get("own_low_alt", 0.0) / count,
            "target_low_alt_rate": total.get("target_low_alt", 0.0) / count,
            "inner_violation_rate": total.get("inner_violation", 0.0) / count,
            "bad_3_9_rate": total.get("bad_3_9", 0.0) / count,
            "red_wez_rate": total.get("red_wez", 0.0) / count,
            "nonfinite_rate": total.get("nonfinite", 0.0) / count,
            "distance_valid_rate": total.get("distance_valid", 0.0) / count,
            "init_feasible_rate": total.get("init_feasible", 0.0) / count,
            "initial_opening_rate": total.get("initial_opening", 0.0) / count,
            "overshoot_rate": total.get("overshoot", 0.0) / count,
            "closure_violation_rate": total.get("closure_violation", 0.0) / count,
        }
        positive_metrics = (
            "track_score",
            "ep_wez_steps",
            "ep_wez_streak_max",
            "target_damage",
            "return",
            "init_feasible",
        )
        rate_metrics = (
            "win",
            "timeout",
            "crash",
            "own_crash",
            "target_crash",
            "target_crash_without_damage",
            "nonfinite",
            "distance_valid",
            "overshoot",
            "closure_violation",
            "inner_violation",
            "bad_3_9",
            "red_wez",
            "initial_opening",
        )
        good_rate_metrics = {"win", "distance_valid"}
        average_metrics = (
            "ep_min_distance",
            "final_ata_deg",
            "final_aa_deg",
            "init_distance_m",
            "init_ata_deg",
            "init_aa_deg",
            "init_closing_mps",
            "init_time_to_wez_s",
            "own_damage",
        )
        bucket_positive = {metric: [] for metric in positive_metrics}
        bucket_rates = {metric: [] for metric in rate_metrics}

        for name, metrics in bucket_totals.items():
            bcount = max(1.0, float(metrics.get("episodes", 0.0)))
            prefix = f"bucket_{name}"
            summary[f"{prefix}_episodes"] = float(metrics.get("episodes", 0.0))

            for metric in positive_metrics:
                if metric in metrics:
                    value = float(metrics.get(metric, 0.0)) / bcount
                    summary[f"{prefix}_{metric}"] = value
                    bucket_positive[metric].append(value)
            for metric in average_metrics:
                if metric in metrics:
                    summary[f"{prefix}_{metric}"] = float(metrics.get(metric, 0.0)) / bcount
            for metric in rate_metrics:
                if metric in metrics:
                    value = float(metrics.get(metric, 0.0)) / bcount
                    summary[f"{prefix}_{metric}_rate"] = value
                    bucket_rates[metric].append(value)
        for metric, values in bucket_positive.items():
            if values:
                summary[f"bucket_worst_{metric}"] = min(values)
                summary[f"bucket_min_{metric}"] = min(values)
                summary[f"bucket_max_{metric}"] = max(values)
        for metric, values in bucket_rates.items():
            if values:
                summary[f"bucket_worst_{metric}_rate"] = (
                    min(values) if metric in good_rate_metrics else max(values)
                )
                summary[f"bucket_min_{metric}_rate"] = min(values)
                summary[f"bucket_max_{metric}_rate"] = max(values)
        return summary

    def all_inactive(self):
        return not bool(self.active.any())


__all__ = ["CompetitionLoiterCurriculumEnv"]

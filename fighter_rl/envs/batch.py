"""Torch-native 1v1 task around CompetitionNeuralPlane; no per-env Python loop."""

import torch
from fighter_rl.sim.neuralplane import CompetitionNeuralPlane


class CompetitionBatchDogfight:
    observation_dim = 16
    action_dim = 4

    def __init__(
        self,
        num_envs=1024,
        device="cpu",
        hz=60,
        step_ratio=6,
        max_steps=3600,
        domain_randomization=True,
    ):
        self.n = num_envs
        self.device = torch.device(device)
        self.hz = int(hz)
        self.step_ratio = int(step_ratio)
        self.max_steps = max_steps
        self.domain_randomization = bool(domain_randomization)
        if self.hz <= 0 or self.step_ratio <= 0:
            raise ValueError("hz and step_ratio must be positive")
        self.own = CompetitionNeuralPlane(num_envs, device, self.hz)
        self.target = CompetitionNeuralPlane(num_envs, device, self.hz)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.own_health = torch.ones(num_envs, device=device)
        self.target_health = torch.ones(num_envs, device=device)
        self.previous_action = torch.zeros(num_envs, 4, device=device)
        self.action_lag = torch.zeros(num_envs, 1, device=device)
        self.obs_noise = 0.0
        self.reset()

    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        # Whole-tensor reset is intentional for rollout batches; masked state
        # replacement can be added when asynchronous episodes are required.
        if self.domain_randomization:
            alt = torch.empty(self.n, device=self.device).uniform_(2500, 10000)
            speed = torch.empty(self.n, device=self.device).uniform_(180, 320)
            roll = torch.empty(self.n, device=self.device).uniform_(-35, 35)
            pitch = torch.empty(self.n, device=self.device).uniform_(-10, 10)
            heading = torch.empty(self.n, device=self.device).uniform_(-180, 180)
            distance = torch.empty(self.n, device=self.device).uniform_(2000, 10000)
            bearing = torch.empty(self.n, device=self.device).uniform_(-torch.pi, torch.pi)
            self.own.reset(alt, speed, roll, pitch, heading)
            self.target.reset(
                (alt + torch.empty_like(alt).uniform_(-1500, 1500)).clamp(1000, 12000),
                torch.empty_like(speed).uniform_(180, 320),
                torch.empty_like(roll).uniform_(-45, 45),
                torch.empty_like(pitch).uniform_(-12, 12),
                heading + torch.empty_like(heading).uniform_(130, 230),
            )
            self.target.state[:, 0] = distance * torch.cos(bearing) / 0.3048
            self.target.state[:, 1] = distance * torch.sin(bearing) / 0.3048
            self.own.randomize_model(0.01, 0.015, 0.005)
            self.target.randomize_model(0.01, 0.015, 0.005)
            self.action_lag.uniform_(0, 0.12)
            self.obs_noise = 0.003
        else:
            self.own.aero_scale.fill_(1)
            self.target.aero_scale.fill_(1)
            self.own.thrust_scale.fill_(1)
            self.target.thrust_scale.fill_(1)
            self.own.reset(6000, 250, 0, 0, 0)
            self.target.reset(6000, 250, 0, 0, 180)
            self.target.state[:, 0] = 5000 / 0.3048
            self.action_lag.zero_()
            self.obs_noise = 0.0
        self.target_altitude_m = self.target.state[:, 2] * 0.3048
        self.previous_action.zero_()
        self.steps.zero_()
        self.own_health.fill_(1)
        self.target_health.fill_(1)
        return self.observation()

    @staticmethod
    def _geometry(own, target):
        delta = target[:, :3] - own[:, :3]
        distance = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-3)
        los = delta / distance[:, None]

        def ned_to_body(v, s):
            phi, theta, psi = torch.deg2rad(s[:, 3]), torch.deg2rad(s[:, 4]), torch.deg2rad(s[:, 5])
            sp, cp = torch.sin(phi), torch.cos(phi)
            st, ct = torch.sin(theta), torch.cos(theta)
            ss, cs = torch.sin(psi), torch.cos(psi)
            n, e, d = v.unbind(1)
            return torch.stack(
                (
                    ct * cs * n + ct * ss * e - st * d,
                    (sp * st * cs - cp * ss) * n + (sp * st * ss + cp * cs) * e + sp * ct * d,
                    (cp * st * cs + sp * ss) * n + (cp * st * ss - sp * cs) * e + cp * ct * d,
                ),
                1,
            )

        own_los = ned_to_body(los, own)
        target_los = ned_to_body(-los, target)
        ata = torch.rad2deg(torch.acos(own_los[:, 0].clamp(-1, 1)))
        aa_abs = torch.rad2deg(torch.acos((-target_los[:, 0]).clamp(-1, 1)))
        side = -target_los[:, 1]
        sign = torch.where(side < -0.10, -torch.ones_like(side), torch.ones_like(side))
        sign = torch.where((side > -0.01) & (side < 0.01), torch.sign(target_los[:, 2]), sign)
        aa = sign * aa_abs
        az = torch.rad2deg(torch.atan2(own_los[:, 1], own_los[:, 0]))
        el = -torch.rad2deg(torch.asin(own_los[:, 2].clamp(-1, 1)))
        return delta, distance, ata, aa, az, el

    def observation(self):
        o, t = self.own.observation41(), self.target.observation41()
        d, r, ata, aa, az, el = self._geometry(o, t)
        x = torch.empty(self.n, 16, device=self.device)
        x[:, 0] = (o[:, 3] / 180).clamp(-1, 1)
        x[:, 1] = (o[:, 4] / 90).clamp(-1, 1)
        x[:, 2] = (torch.remainder(o[:, 5], 360) / 180 - 1).clamp(-1, 1)
        x[:, 3] = (o[:, 12] / 300 - 1).clamp(-1, 1)
        x[:, 4] = ((-o[:, 2]) / 7500 - 1).clamp(-1, 1)
        x[:, 5] = 2 * self.own_health - 1
        x[:, 6] = (d[:, 0] / 15000).clamp(-1, 1)
        x[:, 7] = (d[:, 1] / 15000).clamp(-1, 1)
        x[:, 8] = (d[:, 2] / 8000).clamp(-1, 1)
        x[:, 9] = ata / 180
        x[:, 10] = aa / 180
        x[:, 11] = az / 180
        x[:, 12] = el / 90
        x[:, 13] = 2 * self.target_health - 1
        in_wez = (r >= 152.4) & (r <= 914.4) & (ata.abs() <= 1.0)
        x[:, 14] = torch.where(in_wez, 1.0, -1.0)
        x[:, 15] = 2 * (1 - ata.abs() / 30).clamp(0, 1) * (1 - r / 3000).clamp(0, 1) - 1
        x = x.clamp(-1, 1)
        if self.obs_noise:
            x = (x + torch.randn_like(x) * self.obs_noise).clamp(-1, 1)
        return x

    @torch.no_grad()
    def step(self, action):
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device).clamp(-1, 1)
        low = a.clone()
        low[:, 3] = (low[:, 3] + 1) / 2
        # Same lightweight level/altitude hold used by AIPDogfightEnv.
        low = (1 - self.action_lag) * low + self.action_lag * self.previous_action
        self.previous_action = low
        own_damage_total = torch.zeros(self.n, device=self.device)
        target_damage_total = torch.zeros(self.n, device=self.device)
        for _ in range(self.step_ratio):
            target_obs = self.target.observation41()
            ta = torch.zeros_like(low)
            ta[:, 0] = (-target_obs[:, 3] / 60).clamp(-0.4, 0.4)
            ta[:, 1] = (
                (self.target_altitude_m - (-target_obs[:, 2])) / 1500 - target_obs[:, 4] / 45
            ).clamp(-0.35, 0.35)
            ta[:, 3] = 0.65
            self.own.step(low)
            self.target.step(ta)
            o, t = self.own.observation41(), self.target.observation41()
            _, distance, ata, _, _, _ = self._geometry(o, t)
            _, _, target_ata, _, _, _ = self._geometry(t, o)
            range_scale = ((914.4 - distance) / (914.4 - 152.4)).clamp(0, 1) / self.hz
            in_range = (distance >= 152.4) & (distance <= 914.4)
            damage = torch.where(
                in_range & (ata.abs() <= 1.0), range_scale, torch.zeros_like(range_scale)
            )
            received = torch.where(
                in_range & (target_ata.abs() <= 1.0), range_scale, torch.zeros_like(range_scale)
            )
            target_damage_total += damage
            own_damage_total += received
        self.target_health = (self.target_health - target_damage_total).clamp_min(0)
        self.own_health = (self.own_health - own_damage_total).clamp_min(0)
        self.steps += 1
        o, t = self.own.observation41(), self.target.observation41()
        _, distance, ata, _, _, _ = self._geometry(o, t)
        _, _, target_ata, _, _, _ = self._geometry(t, o)
        terminated = (
            (self.own_health <= 0) | (self.target_health <= 0) | (-o[:, 2] < 300) | (-t[:, 2] < 300)
        )
        done = terminated | (self.steps >= self.max_steps)
        pursuit = 0.3 * (1 - ata.abs() / 30).clamp(0, 1) * (1 - distance / 3000).clamp(0, 1)
        reward = (
            -0.01
            + pursuit
            + 20 * (target_damage_total - own_damage_total)
            - 0.1 * ((-o[:, 2]) < 600).float()
        )
        reward = reward + torch.where(
            terminated,
            torch.where(
                (self.target_health <= 0) & (self.own_health > 0),
                100.0,
                torch.where((self.own_health <= 0) & (self.target_health > 0), -100.0, -30.0),
            ),
            torch.zeros_like(reward),
        )
        return self.observation(), reward, done, {"distance_m": distance, "ata_deg": ata}

"""Fast SAC actor/critic modules aligned to AIP model profiles.

For AIP lightweight inference, SAC bundles use the actor side of the RLModule.
The Q networks are necessary for fast SAC training, but they are not used by
``RLActionProvider`` during submission-time inference.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from torch import nn


ACTION_DIM = 4
BASE_OBS_DIM = 16


@dataclass(frozen=True)
class AIPSACProfile:
    variant: str
    obs_dim: int
    temporal_frames: int
    include_previous_action: bool
    actor_pre_hiddens: tuple[int, ...]
    actor_post_hiddens: tuple[int, ...]
    critic_pre_hiddens: tuple[int, ...]
    critic_post_hiddens: tuple[int, ...]
    use_lstm: bool
    lstm_cell_size: int = 0
    max_seq_len: int = 1

    @property
    def temporal_config(self) -> dict:
        return {
            "enabled": True,
            "frames": self.temporal_frames,
            "include_previous_action": self.include_previous_action,
        }

    @property
    def network_spec(self) -> dict | None:
        if not self.use_lstm:
            return None
        return {
            "type": "sequence_v1",
            "actor": {
                "pre_lstm_hiddens": list(self.actor_pre_hiddens),
                "pre_lstm_activation": "relu",
                "lstm_cell_size": self.lstm_cell_size,
                "lstm_layers": 1,
                "post_lstm_hiddens": list(self.actor_post_hiddens),
                "post_lstm_activation": "relu",
                "input": "obs",
                "zero_init_state": False,
            },
            "critic": {
                "pre_lstm_hiddens": list(self.critic_pre_hiddens),
                "pre_lstm_activation": "relu",
                "lstm_cell_size": self.lstm_cell_size,
                "lstm_layers": 1,
                "post_lstm_hiddens": list(self.critic_post_hiddens),
                "post_lstm_activation": "relu",
                "input": "obs_action",
                "zero_init_state": True,
            },
        }

    @property
    def model_config(self) -> dict:
        if self.use_lstm:
            return {
                "enabled": True,
                "fcnet_hiddens": list(self.actor_pre_hiddens),
                "fcnet_activation": "relu",
                "head_fcnet_hiddens": list(self.actor_post_hiddens),
                "head_fcnet_activation": "relu",
                "use_lstm": True,
                "lstm_cell_size": self.lstm_cell_size,
                "max_seq_len": self.max_seq_len,
                "network_spec": self.network_spec,
            }
        return {
            "enabled": True,
            "fcnet_hiddens": list(self.actor_pre_hiddens),
            "fcnet_activation": "relu",
            "head_fcnet_hiddens": list(self.actor_post_hiddens),
            "head_fcnet_activation": "relu",
        }

    def as_metadata(self) -> dict:
        return asdict(self) | {
            "action_dim": ACTION_DIM,
            "base_observation_dim": BASE_OBS_DIM,
            "temporal_config": self.temporal_config,
            "model_config": self.model_config,
            "network_spec": self.network_spec,
        }


PROFILES: dict[str, AIPSACProfile] = {
    "sac_mlp": AIPSACProfile(
        variant="sac_mlp",
        obs_dim=80,
        temporal_frames=4,
        include_previous_action=True,
        actor_pre_hiddens=(256, 256),
        actor_post_hiddens=(),
        critic_pre_hiddens=(256, 256),
        critic_post_hiddens=(),
        use_lstm=False,
    ),
    "sac_lstm": AIPSACProfile(
        variant="sac_lstm",
        obs_dim=20,
        temporal_frames=1,
        include_previous_action=True,
        actor_pre_hiddens=(128,),
        actor_post_hiddens=(128,),
        critic_pre_hiddens=(128,),
        critic_post_hiddens=(128,),
        use_lstm=True,
        lstm_cell_size=64,
        max_seq_len=16,
    ),
}


def get_sac_profile(variant: str) -> AIPSACProfile:
    key = str(variant).strip().lower()
    if key not in PROFILES:
        raise ValueError(
            f"Unsupported fast SAC variant: {variant!r}. "
            f"Supported: {', '.join(sorted(PROFILES))}"
        )
    return PROFILES[key]


def mlp(sizes: list[int], activation=nn.ReLU, *, output_activation=None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = i == len(sizes) - 2
        act = output_activation if is_last else activation
        if act is not None:
            layers.append(act())
    return nn.Sequential(*layers)


class SACOutput(NamedTuple):
    logits: torch.Tensor
    state: tuple[torch.Tensor, torch.Tensor] | None


class FastAIPSACActor(nn.Module):
    def __init__(self, profile: AIPSACProfile | str):
        super().__init__()
        self.profile = get_sac_profile(profile) if isinstance(profile, str) else profile
        if self.profile.use_lstm:
            pre = [self.profile.obs_dim, *self.profile.actor_pre_hiddens]
            self.pre = mlp(pre)
            self.lstm = nn.LSTM(self.profile.actor_pre_hiddens[-1], self.profile.lstm_cell_size)
            post_in = self.profile.lstm_cell_size
        else:
            self.pre = mlp([self.profile.obs_dim, *self.profile.actor_pre_hiddens])
            self.lstm = None
            post_in = self.profile.actor_pre_hiddens[-1]
        if self.profile.actor_post_hiddens:
            self.post = mlp([post_in, *self.profile.actor_post_hiddens])
            head_in = self.profile.actor_post_hiddens[-1]
        else:
            self.post = nn.Identity()
            head_in = post_in
        self.head = nn.Linear(head_in, ACTION_DIM * 2)
        self.reset_output_initialization()

    def reset_output_initialization(self) -> None:
        # SAC is very sensitive to violent initial exploration in stage 0.
        # Bias the log-std half of the actor head to a moderate std while
        # keeping the exact AIP/RLlib actor output shape.
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            # Policy actions use [-1, 1], while CompetitionLoiterCurriculumEnv
            # maps the throttle channel by (a + 1) / 2 before calling the
            # simulator.  Bias the initial mean to ~65% simulator throttle
            # (policy action 0.30), otherwise stage-0 safety rollouts start at
            # only 50% throttle and can repeatedly enter the same slow
            # energy-loss crash pattern before SAC gets useful signal.
            self.head.bias[3].fill_(0.30)
            self.head.bias[ACTION_DIM:].fill_(-1.0)

    @property
    def obs_dim(self) -> int:
        return self.profile.obs_dim

    @property
    def recurrent(self) -> bool:
        return self.profile.use_lstm

    def initial_state(
        self, batch_size: int, device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.recurrent:
            return None
        dev = torch.device(device) if device is not None else next(self.parameters()).device
        h = torch.zeros(1, int(batch_size), self.profile.lstm_cell_size, device=dev)
        c = torch.zeros_like(h)
        return h, c

    @staticmethod
    def detach_state(state):
        if state is None:
            return None
        return state[0].detach(), state[1].detach()

    @staticmethod
    def mask_state(state, active: torch.Tensor):
        if state is None:
            return None
        mask = active.reshape(1, -1, 1).to(dtype=state[0].dtype, device=state[0].device)
        return state[0] * mask, state[1] * mask

    def forward_step(self, obs: torch.Tensor, state=None) -> SACOutput:
        x = self.pre(obs)
        next_state = state
        if self.recurrent:
            if state is None:
                state = self.initial_state(obs.shape[0], obs.device)
            assert self.lstm is not None
            x, next_state = self.lstm(x.unsqueeze(0), state)
            x = x.squeeze(0)
        x = self.post(x)
        return SACOutput(self.head(x), next_state)

    def forward_sequence(self, obs: torch.Tensor, state=None) -> SACOutput:
        t, b, d = obs.shape
        x = self.pre(obs.reshape(t * b, d)).reshape(t, b, -1)
        next_state = state
        if self.recurrent:
            if state is None:
                state = self.initial_state(b, obs.device)
            assert self.lstm is not None
            x, next_state = self.lstm(x, state)
        x = self.post(x.reshape(t * b, -1)).reshape(t, b, -1)
        return SACOutput(self.head(x), next_state)

    @staticmethod
    def distribution(logits, log_std_min=-5.0, log_std_max=1.0, mean_clip=10.0):
        mean, log_std = logits.split(ACTION_DIM, dim=-1)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=mean_clip, neginf=-mean_clip)
        if mean_clip > 0:
            mean = mean.clamp(-mean_clip, mean_clip)
        log_std = torch.nan_to_num(
            log_std,
            nan=log_std_min,
            posinf=log_std_max,
            neginf=log_std_min,
        ).clamp(log_std_min, log_std_max)
        return torch.distributions.Normal(mean, log_std.exp())

    @staticmethod
    def _squash(raw_action: torch.Tensor, logp: torch.Tensor | None = None):
        action = torch.tanh(raw_action)
        if logp is None:
            return action, None
        correction = torch.log(1 - action.square() + 1e-6).sum(-1)
        return action, logp - correction

    def sample_step(self, obs, state=None, deterministic=False, return_logits=False):
        out = self.forward_step(obs, state)
        dist = self.distribution(out.logits)
        if deterministic:
            raw = dist.mean
        else:
            raw = dist.rsample()
        raw = torch.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0)
        logp = dist.log_prob(raw).sum(-1)
        action, logp = self._squash(raw, logp)
        if return_logits:
            return action, logp, out.state, out.logits
        return action, logp, out.state

    def sample_sequence(self, obs, state=None, deterministic=False, return_logits=False):
        out = self.forward_sequence(obs, state)
        dist = self.distribution(out.logits)
        raw = dist.mean if deterministic else dist.rsample()
        raw = torch.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0)
        logp = dist.log_prob(raw).sum(-1)
        action, logp = self._squash(raw, logp)
        if return_logits:
            return action, logp, out.state, out.logits
        return action, logp, out.state


class FastAIPSACCritic(nn.Module):
    def __init__(self, profile: AIPSACProfile | str):
        super().__init__()
        self.profile = get_sac_profile(profile) if isinstance(profile, str) else profile
        input_dim = self.profile.obs_dim + ACTION_DIM
        if self.profile.use_lstm:
            self.pre = mlp([input_dim, *self.profile.critic_pre_hiddens])
            self.lstm = nn.LSTM(self.profile.critic_pre_hiddens[-1], self.profile.lstm_cell_size)
            post_in = self.profile.lstm_cell_size
        else:
            self.pre = mlp([input_dim, *self.profile.critic_pre_hiddens])
            self.lstm = None
            post_in = self.profile.critic_pre_hiddens[-1]
        if self.profile.critic_post_hiddens:
            self.post = mlp([post_in, *self.profile.critic_post_hiddens])
            head_in = self.profile.critic_post_hiddens[-1]
        else:
            self.post = nn.Identity()
            head_in = post_in
        self.q = nn.Linear(head_in, 1)

    def initial_state(self, batch_size, device):
        if not self.profile.use_lstm:
            return None
        h = torch.zeros(1, int(batch_size), self.profile.lstm_cell_size, device=device)
        c = torch.zeros_like(h)
        return h, c

    def forward_sequence(self, obs, action, state=None):
        t, b, _ = obs.shape
        x = torch.cat([obs, action], dim=-1)
        x = self.pre(x.reshape(t * b, -1)).reshape(t, b, -1)
        if self.profile.use_lstm:
            if state is None:
                state = self.initial_state(b, obs.device)
            assert self.lstm is not None
            x, _ = self.lstm(x, state)
        x = self.post(x.reshape(t * b, -1)).reshape(t, b, -1)
        return self.q(x).squeeze(-1)

    def forward_flat(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = self.pre(x)
        if self.profile.use_lstm:
            # One-step critic fallback for diagnostics; sequence training uses
            # forward_sequence so recurrent state remains explicit.
            assert self.lstm is not None
            x, _ = self.lstm(x.unsqueeze(0), self.initial_state(x.shape[0], x.device))
            x = x.squeeze(0)
        x = self.post(x)
        return self.q(x).squeeze(-1)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1.0 - tau).add_(sp, alpha=tau)


def _np(t):
    return t.detach().cpu().numpy()


def rllib_sac_actor_weight_dict(actor: FastAIPSACActor, current_state: dict | None = None) -> dict[str, object]:
    """Map fast SAC actor weights to RLlib/AIP actor state keys.

    If ``current_state`` is provided, MLP key names are discovered by matching
    expected shapes.  This keeps the exporter robust across small RLlib key-name
    changes while still failing on ambiguous/missing shapes.
    """
    import numpy as np

    state = actor.state_dict()
    const = np.asarray([20.0], dtype=np.float32)
    if actor.profile.use_lstm:
        return {
            "pi_encoder.tokenizer.net.mlp.0.weight": _np(state["pre.0.weight"]),
            "pi_encoder.tokenizer.net.mlp.0.bias": _np(state["pre.0.bias"]),
            "pi_encoder.lstm.weight_ih_l0": _np(state["lstm.weight_ih_l0"]),
            "pi_encoder.lstm.weight_hh_l0": _np(state["lstm.weight_hh_l0"]),
            "pi_encoder.lstm.bias_ih_l0": _np(state["lstm.bias_ih_l0"]),
            "pi_encoder.lstm.bias_hh_l0": _np(state["lstm.bias_hh_l0"]),
            "pi.log_std_clip_param_const": const,
            "pi.net.mlp.0.weight": _np(state["post.0.weight"]),
            "pi.net.mlp.0.bias": _np(state["post.0.bias"]),
            "pi.net.mlp.2.weight": _np(state["head.weight"]),
            "pi.net.mlp.2.bias": _np(state["head.bias"]),
        }
    if current_state is None:
        # Most Ray 2.x SAC MLP modules use these names.  The exporter passes
        # current_state and will discover/verify if they differ.
        return {
            "pi_encoder.encoder.net.mlp.0.weight": _np(state["pre.0.weight"]),
            "pi_encoder.encoder.net.mlp.0.bias": _np(state["pre.0.bias"]),
            "pi_encoder.encoder.net.mlp.2.weight": _np(state["pre.2.weight"]),
            "pi_encoder.encoder.net.mlp.2.bias": _np(state["pre.2.bias"]),
            "pi.log_std_clip_param_const": const,
            "pi.net.mlp.0.weight": _np(state["head.weight"]),
            "pi.net.mlp.0.bias": _np(state["head.bias"]),
        }

    expected = [
        ("pre.0.weight", _np(state["pre.0.weight"]), ("pi_encoder", "0.weight")),
        ("pre.0.bias", _np(state["pre.0.bias"]), ("pi_encoder", "0.bias")),
        ("pre.2.weight", _np(state["pre.2.weight"]), ("pi_encoder", "2.weight")),
        ("pre.2.bias", _np(state["pre.2.bias"]), ("pi_encoder", "2.bias")),
        ("head.weight", _np(state["head.weight"]), ("pi.", "weight")),
        ("head.bias", _np(state["head.bias"]), ("pi.", "bias")),
    ]
    used: set[str] = set()
    mapped: dict[str, object] = {}
    for _, value, tokens in expected:
        matches = []
        for key, cur in current_state.items():
            if key in used:
                continue
            if all(token in key for token in tokens) and tuple(np.asarray(cur).shape) == tuple(value.shape):
                matches.append(key)
        if len(matches) != 1:
            raise RuntimeError(
                f"Cannot uniquely map SAC MLP actor tensor shape={tuple(value.shape)} tokens={tokens}: {matches}"
            )
        mapped[matches[0]] = value
        used.add(matches[0])
    if "pi.log_std_clip_param_const" in current_state:
        mapped["pi.log_std_clip_param_const"] = const
    return mapped


__all__ = [
    "AIPSACProfile",
    "FastAIPSACActor",
    "FastAIPSACCritic",
    "PROFILES",
    "get_sac_profile",
    "rllib_sac_actor_weight_dict",
    "soft_update",
]

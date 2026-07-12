"""AIP-profile PPO policies for fast GPU-batched NeuralPlane training.

The goal of this module is deliberately narrow:

* train fast with plain PyTorch tensors, not RLlib sampling;
* keep the policy tensor shapes compatible with AIP/RLlib PPO bundles;
* make the observation contract impossible to mix up.

Observed AIP bundle contracts:

* PPO-MLP  : tactical16 stacked 4 frames + previous action per frame = 80 dims.
* PPO-LSTM : tactical16 + previous action = 20 dims, then RLlib LSTM(64).

RLlib's PPO Gaussian head for Box(4) actions emits 8 values
(`mean[4] + log_std[4]`).  The earlier fast trainer used a separate global
`log_std` parameter, which trains fine as a standalone `.pt`, but is not a
clean shape match for the lightweight bundle.  This file fixes that.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from torch import nn


ACTION_DIM = 4
BASE_OBS_DIM = 16


@dataclass(frozen=True)
class AIPPolicyProfile:
    variant: str
    algo: str
    obs_dim: int
    temporal_frames: int
    include_previous_action: bool
    encoder_hiddens: tuple[int, int]
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
    def model_config(self) -> dict:
        payload = {
            "enabled": True,
            "fcnet_hiddens": list(self.encoder_hiddens),
            "fcnet_activation": "relu",
            "head_fcnet_hiddens": [],
            "head_fcnet_activation": "relu",
            "vf_share_layers": True,
        }
        if self.use_lstm:
            payload.update(
                {
                    "use_lstm": True,
                    "max_seq_len": self.max_seq_len,
                    "lstm_cell_size": self.lstm_cell_size,
                }
            )
        return payload

    def as_metadata(self) -> dict:
        return asdict(self) | {
            "action_dim": ACTION_DIM,
            "base_observation_dim": BASE_OBS_DIM,
            "temporal_config": self.temporal_config,
            "model_config": self.model_config,
        }


PROFILES: dict[str, AIPPolicyProfile] = {
    "ppo_mlp": AIPPolicyProfile(
        variant="ppo_mlp",
        algo="ppo",
        obs_dim=80,
        temporal_frames=4,
        include_previous_action=True,
        encoder_hiddens=(256, 256),
        use_lstm=False,
    ),
    "ppo_lstm": AIPPolicyProfile(
        variant="ppo_lstm",
        algo="ppo",
        obs_dim=20,
        temporal_frames=1,
        include_previous_action=True,
        encoder_hiddens=(128, 128),
        use_lstm=True,
        lstm_cell_size=64,
        max_seq_len=16,
    ),
}


def get_profile(variant: str) -> AIPPolicyProfile:
    key = str(variant).strip().lower()
    if key not in PROFILES:
        raise ValueError(
            f"Unsupported fast PPO variant: {variant!r}. "
            f"Supported: {', '.join(sorted(PROFILES))}"
        )
    return PROFILES[key]


class PolicyOutput(NamedTuple):
    logits: torch.Tensor
    value: torch.Tensor
    state: tuple[torch.Tensor, torch.Tensor] | None


class FastAIPPPOPolicy(nn.Module):
    """PPO actor-critic with RLlib-compatible PPO head shapes."""

    def __init__(self, profile: AIPPolicyProfile | str):
        super().__init__()
        self.profile = get_profile(profile) if isinstance(profile, str) else profile
        h1, h2 = self.profile.encoder_hiddens
        self.encoder = nn.Sequential(
            nn.Linear(self.profile.obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        if self.profile.use_lstm:
            self.lstm = nn.LSTM(h2, self.profile.lstm_cell_size)
            head_in = self.profile.lstm_cell_size
        else:
            self.lstm = None
            head_in = h2
        self.pi = nn.Linear(head_in, ACTION_DIM * 2)
        self.vf = nn.Linear(head_in, 1)
        self.reset_output_initialization()

    def reset_output_initialization(self) -> None:
        # Keep the initial policy close to "small controls" instead of a nearly
        # full-range random actuator.  This preserves the RLlib-compatible head
        # shape while making stage-0 survival learnable.
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)
        with torch.no_grad():
            # Policy actions are in [-1, 1], but the fast environment maps the
            # throttle channel to simulator throttle by (a + 1) / 2.  A zero
            # throttle-action therefore means only 50% throttle.  In stage-0
            # safety training that was enough to let many rollouts slowly bleed
            # energy and crash around the 300-600 decision range before PPO
            # received a clean learning signal.  Match the target loiter
            # controller's nominal ~65% throttle instead:
            #   simulator_throttle = 0.65 -> policy_action = 2*0.65 - 1 = 0.30.
            self.pi.bias[3].fill_(0.30)
            self.pi.bias[ACTION_DIM:].fill_(-1.0)
        nn.init.orthogonal_(self.vf.weight, gain=1.0)
        nn.init.zeros_(self.vf.bias)

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
    def detach_state(
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if state is None:
            return None
        return state[0].detach(), state[1].detach()

    @staticmethod
    def mask_state(
        state: tuple[torch.Tensor, torch.Tensor] | None, active: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if state is None:
            return None
        mask = active.reshape(1, -1, 1).to(dtype=torch.bool, device=state[0].device)
        h = torch.nan_to_num(state[0], nan=0.0, posinf=0.0, neginf=0.0)
        c = torch.nan_to_num(state[1], nan=0.0, posinf=0.0, neginf=0.0)
        return torch.where(mask, h, torch.zeros_like(h)), torch.where(mask, c, torch.zeros_like(c))

    def forward_step(
        self,
        obs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> PolicyOutput:
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        features = self.encoder(obs)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        next_state = state
        if self.recurrent:
            if state is None:
                state = self.initial_state(obs.shape[0], obs.device)
            else:
                state = (
                    torch.nan_to_num(state[0], nan=0.0, posinf=0.0, neginf=0.0),
                    torch.nan_to_num(state[1], nan=0.0, posinf=0.0, neginf=0.0),
                )
            assert self.lstm is not None
            out, next_state = self.lstm(features.unsqueeze(0), state)
            features = out.squeeze(0)
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        logits = self.pi(features)
        value = self.vf(features).squeeze(-1)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        return PolicyOutput(logits, value, next_state)

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> PolicyOutput:
        """Run a full rollout sequence.

        Args:
            obs_seq: Tensor shaped [T, B, obs_dim].
            state: Optional recurrent state shaped [1, B, H].
        """
        t, b, d = obs_seq.shape
        if d != self.obs_dim:
            raise ValueError(f"Expected obs_dim={self.obs_dim}, got {d}")
        obs_seq = torch.nan_to_num(obs_seq, nan=0.0, posinf=1.0, neginf=-1.0)
        features = self.encoder(obs_seq.reshape(t * b, d)).reshape(t, b, -1)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        next_state = state
        if self.recurrent:
            if state is None:
                state = self.initial_state(b, obs_seq.device)
            else:
                state = (
                    torch.nan_to_num(state[0], nan=0.0, posinf=0.0, neginf=0.0),
                    torch.nan_to_num(state[1], nan=0.0, posinf=0.0, neginf=0.0),
                )
            assert self.lstm is not None
            features, next_state = self.lstm(features, state)
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        logits = self.pi(features)
        value = self.vf(features).squeeze(-1)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        return PolicyOutput(logits, value, next_state)

    @staticmethod
    def action_distribution(
        logits: torch.Tensor,
        *,
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
        mean_clip: float = 10.0,
    ) -> torch.distributions.Normal:
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

    def sample_step(
        self,
        obs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
        mean_clip: float = 10.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        output = self.forward_step(obs, state)
        dist = self.action_distribution(
            output.logits,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            mean_clip=mean_clip,
        )
        raw_action = dist.rsample()
        raw_action = torch.nan_to_num(raw_action, nan=0.0, posinf=1.0, neginf=-1.0)
        logp = dist.log_prob(raw_action).sum(-1)
        env_action = raw_action.clamp(-1.0, 1.0)
        return env_action, raw_action, logp, output.value, output.state

    def deterministic_action(
        self,
        obs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        output = self.forward_step(obs, state)
        mean, _ = output.logits.split(ACTION_DIM, dim=-1)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        return mean.clamp(-1.0, 1.0), output.state


def evaluate_logp_entropy(
    logits: torch.Tensor,
    raw_action: torch.Tensor,
    *,
    log_std_min: float = -5.0,
    log_std_max: float = 1.0,
    mean_clip: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    dist = FastAIPPPOPolicy.action_distribution(
        logits,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
        mean_clip=mean_clip,
    )
    return dist.log_prob(raw_action).sum(-1), dist.entropy().sum(-1)


def rllib_weight_dict(policy: FastAIPPPOPolicy) -> dict[str, object]:
    """Map a fast PPO policy into AIP/RLlib lightweight-bundle state keys."""
    import numpy as np

    state = policy.state_dict()
    const = np.asarray([20.0], dtype=np.float32)
    result: dict[str, object] = {
        "pi.log_std_clip_param_const": const,
    }
    if policy.profile.use_lstm:
        result.update(
            {
                "encoder.encoder.tokenizer.net.mlp.0.weight": state[
                    "encoder.0.weight"
                ].detach().cpu().numpy(),
                "encoder.encoder.tokenizer.net.mlp.0.bias": state[
                    "encoder.0.bias"
                ].detach().cpu().numpy(),
                "encoder.encoder.tokenizer.net.mlp.2.weight": state[
                    "encoder.2.weight"
                ].detach().cpu().numpy(),
                "encoder.encoder.tokenizer.net.mlp.2.bias": state[
                    "encoder.2.bias"
                ].detach().cpu().numpy(),
                "encoder.encoder.lstm.weight_ih_l0": state[
                    "lstm.weight_ih_l0"
                ].detach().cpu().numpy(),
                "encoder.encoder.lstm.weight_hh_l0": state[
                    "lstm.weight_hh_l0"
                ].detach().cpu().numpy(),
                "encoder.encoder.lstm.bias_ih_l0": state[
                    "lstm.bias_ih_l0"
                ].detach().cpu().numpy(),
                "encoder.encoder.lstm.bias_hh_l0": state[
                    "lstm.bias_hh_l0"
                ].detach().cpu().numpy(),
                "pi.net.mlp.0.weight": state["pi.weight"].detach().cpu().numpy(),
                "pi.net.mlp.0.bias": state["pi.bias"].detach().cpu().numpy(),
                "vf.log_std_clip_param_const": const,
                "vf.net.mlp.0.weight": state["vf.weight"].detach().cpu().numpy(),
                "vf.net.mlp.0.bias": state["vf.bias"].detach().cpu().numpy(),
            }
        )
    else:
        result.update(
            {
                "encoder.encoder.net.mlp.0.weight": state[
                    "encoder.0.weight"
                ].detach().cpu().numpy(),
                "encoder.encoder.net.mlp.0.bias": state[
                    "encoder.0.bias"
                ].detach().cpu().numpy(),
                "encoder.encoder.net.mlp.2.weight": state[
                    "encoder.2.weight"
                ].detach().cpu().numpy(),
                "encoder.encoder.net.mlp.2.bias": state[
                    "encoder.2.bias"
                ].detach().cpu().numpy(),
                "pi.net.mlp.0.weight": state["pi.weight"].detach().cpu().numpy(),
                "pi.net.mlp.0.bias": state["pi.bias"].detach().cpu().numpy(),
            }
        )
    return result


__all__ = [
    "ACTION_DIM",
    "BASE_OBS_DIM",
    "AIPPolicyProfile",
    "FastAIPPPOPolicy",
    "PROFILES",
    "evaluate_logp_entropy",
    "get_profile",
    "rllib_weight_dict",
]

"""Validate the restored fast parallel training contract.

This is a cheap preflight for the server:

* AIP profile -> expected observation dimension.
* CompetitionLoiterCurriculumEnv -> actual observation dimension.
* BT target name is accepted.
* Fast policy -> action/value tensor shapes.
* RLlib export key shapes -> known AIP lightweight-bundle key names.
"""

import json

import torch

from fighter_rl.envs.loiter import CompetitionLoiterCurriculumEnv
from fighter_rl.models.ppo import FastAIPPPOPolicy, get_profile, rllib_weight_dict
from fighter_rl.models.sac import (
    FastAIPSACActor,
    FastAIPSACCritic,
    get_sac_profile,
    rllib_sac_actor_weight_dict,
)
from fighter_rl.training.stages import load_stages
from fighter_rl.utils.config import load_training_config

TARGETS = (
    "random_loiter",
    "fixed_loiter",
    "gun_curriculum",
    "stage_mix",
    "curriculum",
    "bt",
    "bt_empty",
)


def main():
    cfg = load_training_config("configs/ppo_lstm.json")
    num_envs = int(getattr(cfg, "preflight_num_envs", 64))
    stage_index = int(getattr(cfg, "preflight_stage_index", getattr(cfg, "start_stage", 0)))
    skip_env = bool(getattr(cfg, "preflight_skip_env", False))

    is_sac = cfg.variant.startswith("sac_")
    torch.manual_seed(cfg.seed)
    profile = get_sac_profile(cfg.variant) if is_sac else get_profile(cfg.variant)
    report = {
        "variant": cfg.variant,
        "expected_obs_dim": profile.obs_dim,
        "temporal_frames": profile.temporal_frames,
        "include_previous_action": profile.include_previous_action,
        "target_maneuver": cfg.target_maneuver,
    }
    device = torch.device(cfg.device)
    model = FastAIPSACActor(profile).to(device) if is_sac else FastAIPPPOPolicy(profile).to(device)
    obs = torch.zeros(num_envs, profile.obs_dim, device=device)
    state = model.initial_state(num_envs, device)
    with torch.no_grad():
        if is_sac:
            action, logp, next_state = model.sample_step(obs, state)
            raw = torch.zeros_like(action)
            value = torch.zeros(num_envs, device=device)
        else:
            action, raw, logp, value, next_state = model.sample_step(obs, state)
    report["policy_shapes"] = {
        "action": list(action.shape),
        "raw_action": list(raw.shape),
        "logp": list(logp.shape),
        "value": list(value.shape),
        "has_recurrent_state": next_state is not None,
    }
    if is_sac:
        critic = FastAIPSACCritic(profile).to(device)
        q = critic.forward_sequence(obs[None, :, :], action[None, :, :])
        report["critic_shapes"] = {"q_sequence": list(q.shape)}
        report["rllib_weight_shapes"] = {
            key: list(value.shape) for key, value in rllib_sac_actor_weight_dict(model).items()
        }
    else:
        report["rllib_weight_shapes"] = {
            key: list(value.shape) for key, value in rllib_weight_dict(model).items()
        }

    if not skip_env:
        stage = load_stages(schedule=cfg.stage_schedule)[stage_index]
        env = CompetitionLoiterCurriculumEnv(
            stage,
            num_envs=num_envs,
            device=device,
            target_maneuver=cfg.target_maneuver,
            temporal_frames=profile.temporal_frames,
            include_previous_action=profile.include_previous_action,
        )
        env_obs = env.reset()
        if env_obs.shape[-1] != profile.obs_dim:
            raise RuntimeError(
                f"env obs_dim={env_obs.shape[-1]} does not match profile obs_dim={profile.obs_dim}"
            )
        next_obs, reward, done, info = env.step(action)
        required_info = {
            "valid",
            "active",
            "own_crash",
            "target_crash",
            "distance_valid",
            "nonfinite",
            "ep_min_own_alt",
            "ep_min_target_alt",
        }
        missing = sorted(required_info - set(info))
        if missing:
            raise RuntimeError(f"env info is missing safety keys: {missing}")
        report["env_shapes"] = {
            "reset_obs": list(env_obs.shape),
            "next_obs": list(next_obs.shape),
            "reward": list(reward.shape),
            "done": list(done.shape),
            "valid": list(info["valid"].shape),
            "own_crash": list(info["own_crash"].shape),
            "target_crash": list(info["target_crash"].shape),
            "distance_valid": list(info["distance_valid"].shape),
            "nonfinite": list(info["nonfinite"].shape),
            "ep_min_own_alt": list(info["ep_min_own_alt"].shape),
            "ep_min_target_alt": list(info["ep_min_target_alt"].shape),
        }
        report["stage"] = {
            "index": stage.index,
            "name": stage.name,
            "decision_limit": stage.decision_limit,
            "step_ratio": stage.step_ratio,
            "advance_conditions": stage.advance_conditions,
        }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

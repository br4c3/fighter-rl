"""Validate the restored fast parallel training contract.

This is a cheap preflight for the server:

* AIP profile -> expected observation dimension.
* CompetitionLoiterCurriculumEnv -> actual observation dimension.
* BT target name is accepted.
* Fast policy -> action/value tensor shapes.
* RLlib export key shapes -> known AIP lightweight-bundle key names.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from competition_loiter_env import CompetitionLoiterCurriculumEnv
from fast_aip_policy import FastAIPPPOPolicy, get_profile, rllib_weight_dict
from fast_aip_sac import (
    FastAIPSACActor,
    FastAIPSACCritic,
    get_sac_profile,
    rllib_sac_actor_weight_dict,
)
from loiter_gpu_stages import load_stages


TARGETS = ("random_loiter", "fixed_loiter", "gun_curriculum", "stage_mix", "curriculum", "bt", "bt_empty")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["ppo_mlp", "ppo_lstm", "sac_mlp", "sac_lstm"], default="ppo_lstm")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--stage-index", type=int, default=0)
    parser.add_argument(
        "--stage-schedule",
        default=os.environ.get("LOITER_STAGE_SCHEDULE") or os.environ.get("STAGE_SCHEDULE") or "aip",
        help="Stage schedule: 'aip', 'kill_bridge', or 'gun_curriculum'.",
    )
    parser.add_argument("--target-maneuver", choices=TARGETS, default="random_loiter")
    parser.add_argument("--skip-env", action="store_true")
    args = parser.parse_args()

    is_sac = args.variant.startswith("sac_")
    torch.manual_seed(args.seed)
    profile = get_sac_profile(args.variant) if is_sac else get_profile(args.variant)
    report = {
        "variant": args.variant,
        "expected_obs_dim": profile.obs_dim,
        "temporal_frames": profile.temporal_frames,
        "include_previous_action": profile.include_previous_action,
        "target_maneuver": args.target_maneuver,
    }
    device = torch.device(args.device)
    model = FastAIPSACActor(profile).to(device) if is_sac else FastAIPPPOPolicy(profile).to(device)
    obs = torch.zeros(args.num_envs, profile.obs_dim, device=device)
    state = model.initial_state(args.num_envs, device)
    with torch.no_grad():
        if is_sac:
            action, logp, next_state = model.sample_step(obs, state)
            raw = torch.zeros_like(action)
            value = torch.zeros(args.num_envs, device=device)
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

    if not args.skip_env:
        stage = load_stages(schedule=args.stage_schedule)[args.stage_index]
        env = CompetitionLoiterCurriculumEnv(
            stage,
            num_envs=args.num_envs,
            device=device,
            target_maneuver=args.target_maneuver,
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

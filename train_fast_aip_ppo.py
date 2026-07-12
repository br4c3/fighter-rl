"""Fast tensor-native PPO curriculum training for AIP/NeuralPlane.

This is the restored fast path:

    CompetitionLoiterCurriculumEnv (GPU batch)
    -> plain PyTorch PPO update
    -> .pt checkpoints
    -> optional RLlib lightweight bundle export/check

It intentionally does not use RLlib for rollout collection.  RLlib is only used
later by ``export_fast_ppo_to_rllib_bundle.py`` to prove/perform bundle
compatibility.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch
from torch import nn

from competition_loiter_env import CompetitionLoiterCurriculumEnv
from experiment_record import append_jsonl, write_experiment_manifest
from fast_aip_policy import FastAIPPPOPolicy, evaluate_logp_entropy, get_profile
from loiter_gpu_stages import LoiterStage, advancement_satisfied, load_stages


def average_window(window: deque[dict[str, float]]) -> dict[str, float]:
    if not window:
        return {}
    # Each item in ``window`` is not a raw log line.  It is an episodic summary
    # emitted by CompetitionLoiterCurriculumEnv.pop_completed_summary(), and it
    # may represent anything from a handful of completed episodes to a whole
    # 4096-env batch.  Therefore the rolling curriculum metrics must be
    # episode-weighted, not "one summary chunk == one vote".
    weights = [max(0.0, float(item.get("episodes", 0.0))) for item in window]
    total_weight = sum(weights)
    keys = set().union(*(item.keys() for item in window))
    out: dict[str, float] = {}
    if "episodes" in keys:
        out["episodes"] = total_weight
    for key in keys:
        if key == "episodes":
            continue
        if total_weight > 0.0:
            numerator = 0.0
            denominator = 0.0
            for item, weight in zip(window, weights):
                if key in item and weight > 0.0:
                    numerator += float(item[key]) * weight
                    denominator += weight
            if denominator > 0.0:
                out[key] = numerator / denominator
            continue
        values = [float(item[key]) for item in window if key in item]
        if values:
            out[key] = sum(values) / len(values)
    return out


def format_gate(
    stage: LoiterStage,
    rolling: dict[str, float],
    window_len: int,
    required: int,
    pass_streak: int = 0,
    required_passes: int = 1,
) -> str:
    if window_len < required:
        return f"pending({window_len}/{required})"
    ok, reason = advancement_satisfied(stage, rolling)
    prefix = "pass:" if ok else "block:"
    text = prefix + reason.replace(" ", "")
    if ok and required_passes > 1:
        text += f",pass_streak={pass_streak}/{required_passes}"
    return text


def format_safety_metrics(rolling: dict[str, float]) -> str:
    """Compact console view for the gun-curriculum guardrail metrics."""
    if not any(
        key in rolling
        for key in (
            "target_damage",
            "own_damage",
            "ep_wez_steps",
            "ep_wez_streak_max",
            "inner_violation_rate",
            "bad_3_9_rate",
            "red_wez_rate",
            "target_crash_without_damage_rate",
            "init_feasible_rate",
            "initial_opening_rate",
            "init_closing_mps",
            "init_time_to_wez_s",
            "track_score",
            "overshoot_rate",
            "closure_violation_rate",
        )
    ):
        return ""
    return (
        f"dmg={rolling.get('target_damage', 0):.3f}/{rolling.get('own_damage', 0):.3f} "
        f"wez={rolling.get('ep_wez_steps', 0):.1f} "
        f"stk={rolling.get('ep_wez_streak_max', 0):.1f} "
        f"inner={rolling.get('inner_violation_rate', 0):.3f} "
        f"red={rolling.get('red_wez_rate', 0):.3f} "
        f"tcr0={rolling.get('target_crash_without_damage_rate', 0):.3f} "
        f"ifeas={rolling.get('init_feasible_rate', 0):.3f} "
        f"cl0={rolling.get('init_closing_mps', 0):.1f} "
        f"twez={rolling.get('init_time_to_wez_s', 0):.1f} "
        f"trk={rolling.get('track_score', 0):.3f} "
        f"ovr={rolling.get('overshoot_rate', 0):.3f} "
        f"clv={rolling.get('closure_violation_rate', 0):.3f}"
    )


def format_training_line(
    *,
    stage_index: int,
    update: int,
    valid_steps: int,
    decision: int,
    decision_limit: int,
    reward_mean: float,
    loss_text: str,
    rolling: dict[str, float],
    gate: str,
    extra: str = "",
) -> str:
    episodes = float(rolling.get("episodes", 0.0))
    parts = [
        f"stage={stage_index}",
        f"upd={update}",
        f"steps={valid_steps}",
        f"dec={decision}/{decision_limit}",
        f"rew={reward_mean:.4f}",
        loss_text,
        f"ep={episodes:.0f}",
    ]
    if rolling:
        parts.extend(
            [
                f"win={rolling.get('win_rate', 0):.3f}",
                f"to={rolling.get('timeout_rate', 0):.3f}",
                f"cr={rolling.get('crash_rate', 0):.3f}",
                f"d={rolling.get('ep_min_distance', float('nan')):.1f}",
                f"ata={rolling.get('final_ata_deg', float('nan')):.1f}",
                f"aa={rolling.get('final_aa_deg', float('nan')):.1f}",
            ]
        )
        safety = format_safety_metrics(rolling)
        if safety:
            parts.append(safety)
    parts.append(f"gate={gate}")
    if extra:
        parts.append(extra)
    return " ".join(part for part in parts if part)


def stage_env_config(stage: LoiterStage, profile) -> dict[str, Any]:
    return {
        "observation_mode": "tactical16",
        "target_mode": "loiter",
        "target_behavior_dll": "AIP_BASE_target.dll",
        "ownship_control_mode": "rl",
        "reward_module": "student.loiter_stage_reward",
        "reward": stage.reward,
        "wez": stage.wez,
        "episode_step_limit": stage.decision_limit,
        "ownship_randomization": stage.ownship_randomization,
        "step_ratio": stage.step_ratio,
        "target_randomization": stage.target_randomization,
        "curriculum_require_advancement": True,
        "max_engage_time": stage.max_engage_time,
        "temporal_observation": profile.temporal_config,
        "base_observation_size": 16,
        "observation_size": profile.obs_dim,
    }


def save_checkpoint(
    path: Path,
    *,
    model: FastAIPPPOPolicy,
    stage: LoiterStage,
    stage_update: int,
    total_valid_steps: int,
    args,
    metrics: dict[str, float],
    status: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "trainer": "train_fast_aip_ppo",
            "variant": model.profile.variant,
            "profile": model.profile.as_metadata(),
            "model": model.state_dict(),
            "obs_dim": model.profile.obs_dim,
            "action_dim": 4,
            "stage_index": stage.index,
            "stage_name": stage.name,
            "stage_update": int(stage_update),
            "steps": int(total_valid_steps),
            "metrics": metrics,
            "status": status,
            "env_config": stage_env_config(stage, model.profile),
            "config": vars(args),
        },
        path,
    )


def load_resume(path: Path, model: FastAIPPPOPolicy, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    variant = payload.get("variant")
    if variant and variant != model.profile.variant:
        raise ValueError(
            f"Resume checkpoint variant mismatch: checkpoint={variant}, "
            f"requested={model.profile.variant}"
        )
    model.load_state_dict(payload["model"])
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["ppo_mlp", "ppo_lstm"], default="ppo_lstm")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--horizon", type=int, default=0, help="0 = profile default")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=65536)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.002)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--log-std-min", type=float, default=-5.0)
    p.add_argument("--log-std-max", type=float, default=1.0)
    p.add_argument(
        "--action-mean-clip",
        type=float,
        default=10.0,
        help=(
            "Clamp PPO Gaussian mean only for training-time distribution "
            "construction. This prevents rare non-finite actor outputs from "
            "terminating long server runs; the exported network is still "
            "regularized to keep raw means small."
        ),
    )
    p.add_argument(
        "--action-mean-l2-coef",
        type=float,
        default=1e-4,
        help=(
            "Small penalty on raw PPO action means.  It discourages saturated "
            "unexportable policies such as throttle mean >> 1 while leaving "
            "normal clipped actions essentially unchanged."
        ),
    )
    p.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.05,
        help=(
            "Skip the PPO update and reset the synchronous batch when fewer "
            "than this fraction of horizon*num_envs transitions are valid. "
            "This prevents late-episode tail rollouts with only a handful of "
            "survivors from destabilizing LSTM PPO."
        ),
    )
    p.add_argument("--max-updates-per-stage", type=int, default=20000)
    p.add_argument("--advance-window", type=int, default=8)
    p.add_argument(
        "--advance-patience",
        type=int,
        default=3,
        help="Number of consecutive passing rolling-window checks required before advancing.",
    )
    p.add_argument("--checkpoint-interval", type=int, default=50)
    p.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Console print interval in updates. metrics.jsonl is still written every update.",
    )
    p.add_argument("--start-stage", type=int, default=0)
    p.add_argument("--stop-stage", type=int, default=None)
    p.add_argument(
        "--stage-schedule",
        default=os.environ.get("LOITER_STAGE_SCHEDULE") or os.environ.get("STAGE_SCHEDULE") or "aip",
        help=(
            "Stage schedule: 'aip' for original 12 stages, 'kill_bridge' "
            "for the loiter-to-kill bridge schedule, or 'gun_curriculum' "
            "for the tight-WEZ trailing/shooting curriculum. Use "
            "'gun_bucket_curriculum' for the axis-separated bucket schedule."
        ),
    )
    p.add_argument("--resume", type=Path)
    p.add_argument("--output", type=Path, default=Path("fast_aip_ppo_runs"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--target-maneuver", default="random_loiter")
    p.add_argument("--residual", type=Path)
    p.add_argument("--residual-gain", type=float, default=0.3)
    p.add_argument("--residual-ramp-seconds", type=float, default=5.0)
    p.add_argument("--no-domain-randomization", action="store_true")
    p.add_argument("--no-auto-advance", action="store_true")
    return p


def ppo_update_mlp(
    model: FastAIPPPOPolicy,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    raw: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    valid: torch.Tensor,
    args,
) -> list[torch.Tensor]:
    mask = valid.reshape(-1)
    flat_obs = obs.reshape(-1, model.obs_dim)[mask]
    flat_raw = raw.reshape(-1, 4)[mask]
    flat_old_logp = old_logp.reshape(-1)[mask]
    flat_adv = advantages.reshape(-1)[mask]
    flat_ret = returns.reshape(-1)[mask]
    if flat_adv.numel() < 2:
        return []
    flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std(unbiased=False) + 1e-8)
    losses: list[torch.Tensor] = []
    count = flat_adv.shape[0]
    for _ in range(args.epochs):
        order = torch.randperm(count, device=flat_adv.device)
        for start in range(0, count, args.minibatch):
            idx = order[start : start + args.minibatch]
            output = model.forward_step(flat_obs[idx])
            logp, entropy = evaluate_logp_entropy(
                output.logits,
                flat_raw[idx],
                log_std_min=args.log_std_min,
                log_std_max=args.log_std_max,
                mean_clip=args.action_mean_clip,
            )
            action_mean = output.logits[..., :4]
            ratio = (logp - flat_old_logp[idx]).exp()
            policy_loss = -torch.minimum(
                ratio * flat_adv[idx],
                ratio.clamp(1 - args.clip, 1 + args.clip) * flat_adv[idx],
            ).mean()
            value_loss = (output.value - flat_ret[idx]).square().mean()
            mean_loss = action_mean.square().mean()
            loss = (
                policy_loss
                + args.vf_coef * value_loss
                - args.entropy_coef * entropy.mean()
                + args.action_mean_l2_coef * mean_loss
            )
            if not bool(torch.isfinite(loss)):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not bool(torch.isfinite(grad_norm)):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            losses.append(loss.detach())
    return losses


def ppo_update_lstm(
    model: FastAIPPPOPolicy,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    raw: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    valid: torch.Tensor,
    initial_state: tuple[torch.Tensor, torch.Tensor],
    args,
) -> list[torch.Tensor]:
    t_steps, n_envs, _ = obs.shape
    envs_per_minibatch = max(1, int(args.minibatch) // max(1, t_steps))
    losses: list[torch.Tensor] = []
    for _ in range(args.epochs):
        order = torch.randperm(n_envs, device=obs.device)
        for start in range(0, n_envs, envs_per_minibatch):
            idx = order[start : start + envs_per_minibatch]
            h0 = initial_state[0][:, idx].contiguous()
            c0 = initial_state[1][:, idx].contiguous()
            output = model.forward_sequence(obs[:, idx], (h0, c0))
            logp, entropy = evaluate_logp_entropy(
                output.logits,
                raw[:, idx],
                log_std_min=args.log_std_min,
                log_std_max=args.log_std_max,
                mean_clip=args.action_mean_clip,
            )
            mask = valid[:, idx].reshape(-1)
            if not bool(mask.any()):
                continue
            action_mean = output.logits[..., :4].reshape(-1, 4)[mask]
            mb_old_logp = old_logp[:, idx].reshape(-1)[mask]
            mb_adv = advantages[:, idx].reshape(-1)[mask]
            mb_ret = returns[:, idx].reshape(-1)[mask]
            mb_logp = logp.reshape(-1)[mask]
            mb_entropy = entropy.reshape(-1)[mask]
            mb_value = output.value.reshape(-1)[mask]
            if mb_adv.numel() < 2:
                continue
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=False) + 1e-8)
            ratio = (mb_logp - mb_old_logp).exp()
            policy_loss = -torch.minimum(
                ratio * mb_adv,
                ratio.clamp(1 - args.clip, 1 + args.clip) * mb_adv,
            ).mean()
            value_loss = (mb_value - mb_ret).square().mean()
            mean_loss = action_mean.square().mean()
            loss = (
                policy_loss
                + args.vf_coef * value_loss
                - args.entropy_coef * mb_entropy.mean()
                + args.action_mean_l2_coef * mean_loss
            )
            if not bool(torch.isfinite(loss)):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not bool(torch.isfinite(grad_norm)):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            losses.append(loss.detach())
    return losses


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    profile = get_profile(args.variant)
    if args.horizon <= 0:
        args.horizon = profile.max_seq_len if profile.use_lstm else 64
    if profile.use_lstm and args.horizon > profile.max_seq_len:
        print(
            f"[warn] PPO-LSTM horizon {args.horizon} > AIP max_seq_len "
            f"{profile.max_seq_len}; export still uses max_seq_len={profile.max_seq_len}",
            flush=True,
        )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(1)
    device = torch.device(args.device)
    stages = load_stages(schedule=args.stage_schedule)
    if args.stop_stage is None:
        args.stop_stage = len(stages) - 1
    if args.stop_stage >= len(stages):
        raise ValueError(f"stop-stage {args.stop_stage} exceeds available stages {len(stages)-1}")

    model = FastAIPPPOPolicy(profile).to(device)
    total_valid = 0
    resume_update = 0
    start_stage = args.start_stage
    if args.resume:
        payload = load_resume(args.resume, model, device)
        saved_stage = int(payload.get("stage_index", start_stage))
        if args.start_stage == 0:
            start_stage = saved_stage
            resume_update = int(payload.get("stage_update", 0))
        elif args.start_stage == saved_stage:
            resume_update = int(payload.get("stage_update", 0))
        total_valid = int(payload.get("steps", 0))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run = args.output / f"{args.variant}_{args.target_maneuver}_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(
        json.dumps(vars(args) | {"profile": profile.as_metadata()}, default=str, indent=2),
        encoding="utf-8",
    )
    write_experiment_manifest(
        run,
        trainer="train_fast_aip_ppo",
        args=args,
        profile=profile,
        stages=stages,
        extra_code_files=[Path(__file__)],
    )
    metrics_log_path = run / "metrics.jsonl"

    curriculum_log: list[dict[str, Any]] = []
    for stage_index in range(start_stage, args.stop_stage + 1):
        stage = stages[stage_index]
        stage_dir = run / f"stage_{stage_index:02d}_{stage.name}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        env = CompetitionLoiterCurriculumEnv(
            stage,
            num_envs=args.num_envs,
            device=device,
            domain_randomization=not args.no_domain_randomization,
            residual=args.residual,
            residual_gain=args.residual_gain,
            residual_ramp_seconds=args.residual_ramp_seconds,
            target_maneuver=args.target_maneuver,
            temporal_frames=profile.temporal_frames,
            include_previous_action=profile.include_previous_action,
        )
        obs_now = env.reset()
        if obs_now.shape[-1] != profile.obs_dim:
            raise RuntimeError(
                f"Observation contract mismatch for {args.variant}: "
                f"env returned {obs_now.shape[-1]}, expected {profile.obs_dim}"
            )
        state = model.initial_state(args.num_envs, device)
        window: deque[dict[str, float]] = deque(maxlen=args.advance_window)
        first_update = resume_update + 1 if stage_index == start_stage else 1
        advanced = False
        pass_streak = 0
        print(
            f"[stage {stage_index}/{args.stop_stage}] {stage.name} "
            f"variant={args.variant} obs={profile.obs_dim} frames={profile.temporal_frames} "
            f"target={args.target_maneuver} decision_limit={stage.decision_limit} "
            f"step_ratio={stage.step_ratio} advance_window={args.advance_window} "
            f"advance_patience={args.advance_patience}",
            flush=True,
        )

        for update in range(first_update, args.max_updates_per_stage + 1):
            obs_buf = []
            raw_buf = []
            logp_buf = []
            value_buf = []
            reward_buf = []
            done_buf = []
            valid_buf = []
            initial_state = model.detach_state(state)
            for _ in range(args.horizon):
                with torch.no_grad():
                    action, raw, logp, value, next_state = model.sample_step(
                        obs_now,
                        state,
                        log_std_min=args.log_std_min,
                        log_std_max=args.log_std_max,
                        mean_clip=args.action_mean_clip,
                    )
                next_obs, reward, done, info = env.step(action)
                valid = info["valid"].bool()
                obs_buf.append(obs_now)
                raw_buf.append(raw)
                logp_buf.append(logp)
                value_buf.append(value)
                reward_buf.append(reward)
                done_buf.append((done | ~valid).float())
                valid_buf.append(valid)
                state = model.mask_state(model.detach_state(next_state), info["active"].bool())
                obs_now = next_obs
                if env.all_inactive():
                    break

            summary = env.pop_completed_summary()
            summary_added = False
            if summary:
                window.append(summary)
                summary_added = True

            obs = torch.stack(obs_buf)
            raw = torch.stack(raw_buf)
            old_logp = torch.stack(logp_buf)
            values = torch.stack(value_buf)
            rewards = torch.stack(reward_buf)
            dones = torch.stack(done_buf)
            valids = torch.stack(valid_buf).bool()

            with torch.no_grad():
                next_value = model.forward_step(obs_now, state).value
            advantages = torch.zeros_like(rewards)
            last = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(rewards.shape[0])):
                nv = next_value if t == rewards.shape[0] - 1 else values[t + 1]
                nonterminal = (1.0 - dones[t]) * valids[t].float()
                delta = (rewards[t] + args.gamma * nv * nonterminal - values[t]) * valids[t].float()
                last = delta + args.gamma * args.gae_lambda * nonterminal * last
                advantages[t] = last
            returns = advantages + values

            valid_count = int(valids.sum().item())
            total_valid += valid_count
            min_valid_count = max(
                2,
                int(valids.numel() * max(0.0, float(args.min_valid_fraction))),
            )
            if valid_count < min_valid_count:
                rolling = average_window(window)
                reward_mean = rewards[valids].mean().item() if valid_count else float("nan")
                decision_max = int(env.steps.max().item())
                gate = format_gate(
                    stage,
                    rolling,
                    len(window),
                    args.advance_window,
                    pass_streak,
                    args.advance_patience,
                )
                print(
                    format_training_line(
                        stage_index=stage_index,
                        update=update,
                        valid_steps=total_valid,
                        decision=decision_max,
                        decision_limit=stage.decision_limit,
                        reward_mean=reward_mean,
                        loss_text="loss=nan",
                        rolling=rolling,
                        gate=gate,
                        extra=f"skip=low_valid({valid_count}<{min_valid_count})",
                    ),
                    flush=True,
                )
                append_jsonl(metrics_log_path, {
                    "stage": stage_index,
                    "stage_name": stage.name,
                    "update": update,
                    "status": "skip_low_valid",
                    "valid_steps": total_valid,
                    "valid_count": valid_count,
                    "min_valid_count": min_valid_count,
                    "decision": decision_max,
                    "decision_limit": stage.decision_limit,
                    "reward_mean": reward_mean,
                    "loss": None,
                    "episodes": rolling.get("episodes", 0.0),
                    "metrics": rolling,
                    "gate": gate,
                })
                obs_now = env.reset()
                state = model.initial_state(args.num_envs, device)
                continue

            if profile.use_lstm:
                assert initial_state is not None
                losses = ppo_update_lstm(
                    model,
                    optimizer,
                    obs,
                    raw,
                    old_logp,
                    advantages,
                    returns,
                    valids,
                    initial_state,
                    args,
                )
            else:
                losses = ppo_update_mlp(
                    model,
                    optimizer,
                    obs,
                    raw,
                    old_logp,
                    advantages,
                    returns,
                    valids,
                    args,
                )

            rolling = average_window(window)
            reward_mean = rewards[valids].mean().item() if valid_count else float("nan")
            decision_max = int(env.steps.max().item())
            gate_ok = False
            display_streak = pass_streak
            if summary_added and len(window) == args.advance_window:
                gate_ok, _ = advancement_satisfied(stage, rolling)
                display_streak = pass_streak + 1 if gate_ok else 0
            gate = format_gate(
                stage,
                rolling,
                len(window),
                args.advance_window,
                display_streak,
                args.advance_patience,
            )
            if not losses:
                print(
                    format_training_line(
                        stage_index=stage_index,
                        update=update,
                        valid_steps=total_valid,
                        decision=decision_max,
                        decision_limit=stage.decision_limit,
                        reward_mean=reward_mean,
                        loss_text="loss=nan",
                        rolling=rolling,
                        gate=gate,
                        extra="skip=no_finite_minibatch",
                    ),
                    flush=True,
                )
                append_jsonl(metrics_log_path, {
                    "stage": stage_index,
                    "stage_name": stage.name,
                    "update": update,
                    "status": "skip_no_finite_minibatch",
                    "valid_steps": total_valid,
                    "valid_count": valid_count,
                    "decision": decision_max,
                    "decision_limit": stage.decision_limit,
                    "reward_mean": reward_mean,
                    "loss": None,
                    "episodes": rolling.get("episodes", 0.0),
                    "metrics": rolling,
                    "gate": gate,
                })
                obs_now = env.reset()
                state = model.initial_state(args.num_envs, device)
                continue
            loss_mean = torch.stack(losses).mean().item()
            if update % max(1, int(args.log_interval)) == 0 or summary_added:
                print(
                    format_training_line(
                        stage_index=stage_index,
                        update=update,
                        valid_steps=total_valid,
                        decision=decision_max,
                        decision_limit=stage.decision_limit,
                        reward_mean=reward_mean,
                        loss_text=f"loss={loss_mean:.4f}",
                        rolling=rolling,
                        gate=gate,
                    ),
                    flush=True,
                )
            append_jsonl(metrics_log_path, {
                "stage": stage_index,
                "stage_name": stage.name,
                "update": update,
                "status": "running",
                "valid_steps": total_valid,
                "valid_count": valid_count,
                "decision": decision_max,
                "decision_limit": stage.decision_limit,
                "reward_mean": reward_mean,
                "loss": loss_mean,
                "episodes": rolling.get("episodes", 0.0),
                "metrics": rolling,
                "gate": gate,
                "summary_added": summary_added,
                "pass_streak": display_streak,
            })

            if env.all_inactive():
                obs_now = env.reset()
                state = model.initial_state(args.num_envs, device)

            if update % args.checkpoint_interval == 0:
                save_checkpoint(
                    stage_dir / "checkpoint.pt",
                    model=model,
                    stage=stage,
                    stage_update=update,
                    total_valid_steps=total_valid,
                    args=args,
                    metrics=rolling,
                    status="running",
                )

            if not args.no_auto_advance and summary_added and len(window) == args.advance_window:
                ok, reason = advancement_satisfied(stage, rolling)
                pass_streak = pass_streak + 1 if ok else 0
                if ok:
                    if pass_streak < max(1, args.advance_patience):
                        continue
                    advanced = True
                    save_checkpoint(
                        stage_dir / "final_checkpoint.pt",
                        model=model,
                        stage=stage,
                        stage_update=update,
                        total_valid_steps=total_valid,
                        args=args,
                        metrics=rolling,
                        status="advanced",
                    )
                    curriculum_log.append(
                        {
                            "stage": stage_index,
                            "name": stage.name,
                            "update": update,
                            "status": "advanced",
                            "reason": reason,
                            "metrics": rolling,
                        }
                    )
                    (run / "curriculum_state.json").write_text(
                        json.dumps(curriculum_log, indent=2),
                        encoding="utf-8",
                    )
                    print(f"[advance] stage={stage_index} {reason}", flush=True)
                    break

        if not advanced:
            rolling = average_window(window)
            save_checkpoint(
                stage_dir / "final_checkpoint.pt",
                model=model,
                stage=stage,
                stage_update=args.max_updates_per_stage,
                total_valid_steps=total_valid,
                args=args,
                metrics=rolling,
                status="stalled",
            )
            curriculum_log.append(
                {
                    "stage": stage_index,
                    "name": stage.name,
                    "update": args.max_updates_per_stage,
                    "status": "stalled",
                    "metrics": rolling,
                }
            )
            (run / "curriculum_state.json").write_text(
                json.dumps(curriculum_log, indent=2),
                encoding="utf-8",
            )
            print(f"[stalled] stage={stage_index}; stopping curriculum", flush=True)
            break
        resume_update = 0

    print(run, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

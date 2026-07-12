"""Fast tensor-native SAC curriculum training for AIP/NeuralPlane."""
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
from fast_aip_sac import (
    FastAIPSACActor,
    FastAIPSACCritic,
    get_sac_profile,
    soft_update,
)
from loiter_gpu_stages import LoiterStage, advancement_satisfied, load_stages
from train_fast_aip_ppo import average_window, format_gate, format_training_line, stage_env_config


class SequenceReplay:
    def __init__(self, capacity_chunks: int = 64, seq_len: int | None = None):
        self.capacity_chunks = max(1, int(capacity_chunks))
        self.seq_len = int(seq_len) if seq_len else None
        self.chunks: list[dict[str, torch.Tensor]] = []
        self.position = 0

    def _fix_length(self, chunk: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.seq_len is None:
            self.seq_len = int(next(iter(chunk.values())).shape[0])
        seq_len = int(self.seq_len)
        current = int(next(iter(chunk.values())).shape[0])
        if current == seq_len:
            return chunk
        fixed: dict[str, torch.Tensor] = {}
        for key, value in chunk.items():
            if current > seq_len:
                fixed[key] = value[-seq_len:]
                continue
            pad_shape = (seq_len - current, *value.shape[1:])
            if key in {"valid", "reward"}:
                pad = torch.zeros(pad_shape, dtype=value.dtype, device=value.device)
            elif key == "done":
                pad = torch.ones(pad_shape, dtype=value.dtype, device=value.device)
            else:
                pad = value[-1:].expand(pad_shape).clone()
            fixed[key] = torch.cat([value, pad], dim=0)
        return fixed

    def add(self, **chunk: torch.Tensor) -> None:
        stored = self._fix_length({key: value.detach().clone() for key, value in chunk.items()})
        if len(self.chunks) < self.capacity_chunks:
            self.chunks.append(stored)
        else:
            self.chunks[self.position] = stored
            self.position = (self.position + 1) % self.capacity_chunks

    @property
    def transitions(self) -> int:
        if not self.chunks:
            return 0
        return sum(int(item["obs"].shape[0] * item["obs"].shape[1]) for item in self.chunks)

    @property
    def valid_transitions(self) -> int:
        if not self.chunks:
            return 0
        return sum(int(item["valid"].sum().item()) for item in self.chunks)

    def sample(self, batch_sequences: int, device: torch.device) -> dict[str, torch.Tensor]:
        if not self.chunks:
            raise RuntimeError("replay buffer is empty")
        by_chunk: dict[int, list[int]] = {}
        for _ in range(int(batch_sequences)):
            ci = random.randrange(len(self.chunks))
            n = self.chunks[ci]["obs"].shape[1]
            ei = random.randrange(n)
            by_chunk.setdefault(ci, []).append(ei)
        out: dict[str, list[torch.Tensor]] = {}
        for ci, env_indices in by_chunk.items():
            idx = torch.as_tensor(env_indices, dtype=torch.long, device=self.chunks[ci]["obs"].device)
            for key, value in self.chunks[ci].items():
                out.setdefault(key, []).append(value[:, idx].to(device))
        return {key: torch.cat(parts, dim=1) for key, parts in out.items()}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["sac_mlp", "sac_lstm"], default="sac_lstm")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--horizon", type=int, default=0)
    p.add_argument("--batch-sequences", type=int, default=512)
    p.add_argument("--updates-per-rollout", type=int, default=16)
    p.add_argument("--learning-starts", type=int, default=65536)
    p.add_argument("--replay-chunks", type=int, default=64)
    p.add_argument(
        "--reset-replay-on-stage",
        action="store_true",
        help=(
            "Clear SAC replay at each curriculum stage boundary. This is safer "
            "for the trail-tracking -> gun-kill curriculum because reward "
            "semantics change across phases."
        ),
    )
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--alpha-lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--target-entropy", type=float, default=-4.0)
    p.add_argument(
        "--action-mean-l2-coef",
        type=float,
        default=1e-4,
        help="Small penalty on raw SAC actor means to avoid saturated, hard-to-export policies.",
    )
    p.add_argument("--min-valid-fraction", type=float, default=0.05)
    p.add_argument("--max-updates-per-stage", type=int, default=20000)
    p.add_argument("--advance-window", type=int, default=8)
    p.add_argument("--advance-patience", type=int, default=3)
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
    p.add_argument("--output", type=Path, default=Path("fast_aip_sac_runs"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--target-maneuver", default="random_loiter")
    p.add_argument("--residual", type=Path)
    p.add_argument("--residual-gain", type=float, default=0.3)
    p.add_argument("--residual-ramp-seconds", type=float, default=5.0)
    p.add_argument(
        "--resume-weights-only",
        action="store_true",
        help="Load model weights from --resume, but honor --start-stage and start that stage from update 1.",
    )
    p.add_argument(
        "--reset-alpha-on-resume",
        action="store_true",
        help="Reset SAC entropy temperature to alpha=1 after loading a checkpoint.",
    )
    p.add_argument("--no-domain-randomization", action="store_true")
    p.add_argument("--no-auto-advance", action="store_true")
    return p


def save_checkpoint(path: Path, *, actor, q1, q2, target_q1, target_q2, log_alpha, stage: LoiterStage, stage_update: int, total_valid_steps: int, args, metrics, status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "trainer": "train_fast_aip_sac",
            "variant": actor.profile.variant,
            "profile": actor.profile.as_metadata(),
            "actor": actor.state_dict(),
            "q1": q1.state_dict(),
            "q2": q2.state_dict(),
            "target_q1": target_q1.state_dict(),
            "target_q2": target_q2.state_dict(),
            "log_alpha": log_alpha.detach().cpu(),
            "obs_dim": actor.profile.obs_dim,
            "action_dim": 4,
            "stage_index": stage.index,
            "stage_name": stage.name,
            "stage_update": int(stage_update),
            "steps": int(total_valid_steps),
            "metrics": metrics,
            "status": status,
            "env_config": stage_env_config(stage, actor.profile),
            "config": vars(args),
        },
        path,
    )


def load_resume(path, actor, q1, q2, target_q1, target_q2, log_alpha, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    variant = payload.get("variant")
    if variant and variant != actor.profile.variant:
        raise ValueError(f"Resume variant mismatch: checkpoint={variant}, requested={actor.profile.variant}")
    actor.load_state_dict(payload["actor"])
    q1.load_state_dict(payload["q1"])
    q2.load_state_dict(payload["q2"])
    target_q1.load_state_dict(payload.get("target_q1", payload["q1"]))
    target_q2.load_state_dict(payload.get("target_q2", payload["q2"]))
    if "log_alpha" in payload:
        with torch.no_grad():
            log_alpha.copy_(torch.as_tensor(payload["log_alpha"], device=device))
    return payload


def sac_updates(actor, q1, q2, target_actor, target_q1, target_q2, actor_opt, critic_opt, log_alpha, alpha_opt, replay: SequenceReplay, args, device):
    losses: list[dict[str, float]] = []
    for _ in range(args.updates_per_rollout):
        batch = replay.sample(args.batch_sequences, device)
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]
        valid = batch["valid"].float()
        with torch.no_grad():
            next_action, next_logp, _ = target_actor.sample_sequence(next_obs)
            tq = torch.minimum(
                target_q1.forward_sequence(next_obs, next_action),
                target_q2.forward_sequence(next_obs, next_action),
            )
            alpha = log_alpha.exp()
            target = reward + args.gamma * (1.0 - done) * valid * (tq - alpha * next_logp)
        q1_pred = q1.forward_sequence(obs, action)
        q2_pred = q2.forward_sequence(obs, action)
        mask = valid > 0
        if not bool(mask.any()):
            continue
        q_loss = ((q1_pred[mask] - target[mask]).square().mean() + (q2_pred[mask] - target[mask]).square().mean())
        if not bool(torch.isfinite(q_loss)):
            continue
        critic_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 5.0)
        if not bool(torch.isfinite(critic_grad)):
            critic_opt.zero_grad(set_to_none=True)
            continue
        critic_opt.step()

        new_action, logp, _, actor_logits = actor.sample_sequence(obs, return_logits=True)
        q_pi = torch.minimum(q1.forward_sequence(obs, new_action), q2.forward_sequence(obs, new_action))
        alpha = log_alpha.exp().detach()
        actor_mean = actor_logits[..., :4][mask].square().mean()
        actor_loss = ((alpha * logp - q_pi)[mask]).mean() + args.action_mean_l2_coef * actor_mean
        if not bool(torch.isfinite(actor_loss)):
            continue
        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        if not bool(torch.isfinite(actor_grad)):
            actor_opt.zero_grad(set_to_none=True)
            continue
        actor_opt.step()

        alpha_loss = -(log_alpha * (logp.detach() + args.target_entropy))[mask].mean()
        if not bool(torch.isfinite(alpha_loss)):
            continue
        alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_grad = nn.utils.clip_grad_norm_([log_alpha], 5.0)
        if not bool(torch.isfinite(alpha_grad)):
            alpha_opt.zero_grad(set_to_none=True)
            continue
        alpha_opt.step()

        soft_update(target_actor, actor, args.tau)
        soft_update(target_q1, q1, args.tau)
        soft_update(target_q2, q2, args.tau)
        losses.append(
            {
                "q": float(q_loss.detach().cpu()),
                "actor": float(actor_loss.detach().cpu()),
                "alpha": float(log_alpha.exp().detach().cpu()),
            }
        )
    return losses


def main() -> int:
    args = build_parser().parse_args()
    profile = get_sac_profile(args.variant)
    if args.horizon <= 0:
        args.horizon = profile.max_seq_len if profile.use_lstm else 32
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
    actor = FastAIPSACActor(profile).to(device)
    target_actor = FastAIPSACActor(profile).to(device)
    q1 = FastAIPSACCritic(profile).to(device)
    q2 = FastAIPSACCritic(profile).to(device)
    target_q1 = FastAIPSACCritic(profile).to(device)
    target_q2 = FastAIPSACCritic(profile).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())
    log_alpha = torch.zeros((), device=device, requires_grad=True)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.critic_lr)
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.alpha_lr)

    start_stage = args.start_stage
    resume_update = 0
    total_valid = 0
    if args.resume:
        payload = load_resume(args.resume, actor, q1, q2, target_q1, target_q2, log_alpha, device)
        target_actor.load_state_dict(actor.state_dict())
        if args.reset_alpha_on_resume:
            with torch.no_grad():
                log_alpha.zero_()
        saved_stage = int(payload.get("stage_index", start_stage))
        if not args.resume_weights_only:
            if args.start_stage == 0:
                start_stage = saved_stage
                resume_update = int(payload.get("stage_update", 0))
            elif args.start_stage == saved_stage:
                resume_update = int(payload.get("stage_update", 0))
        total_valid = 0 if args.resume_weights_only else int(payload.get("steps", 0))

    run = args.output / f"{args.variant}_{args.target_maneuver}_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(json.dumps(vars(args) | {"profile": profile.as_metadata()}, default=str, indent=2), encoding="utf-8")
    write_experiment_manifest(
        run,
        trainer="train_fast_aip_sac",
        args=args,
        profile=profile,
        stages=stages,
        extra_code_files=[Path(__file__)],
    )
    metrics_log_path = run / "metrics.jsonl"
    replay = SequenceReplay(args.replay_chunks, seq_len=args.horizon)
    curriculum_log: list[dict[str, Any]] = []
    for stage_index in range(start_stage, args.stop_stage + 1):
        stage = stages[stage_index]
        if args.reset_replay_on_stage and stage_index != start_stage:
            replay = SequenceReplay(args.replay_chunks, seq_len=args.horizon)
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
            raise RuntimeError(f"Observation mismatch: got {obs_now.shape[-1]}, expected {profile.obs_dim}")
        state = actor.initial_state(args.num_envs, device)
        window: deque[dict[str, float]] = deque(maxlen=args.advance_window)
        first_update = resume_update + 1 if stage_index == start_stage else 1
        advanced = False
        pass_streak = 0
        print(
            f"[stage {stage_index}/{args.stop_stage}] {stage.name} variant={args.variant} "
            f"obs={profile.obs_dim} frames={profile.temporal_frames} target={args.target_maneuver} "
            f"decision_limit={stage.decision_limit} step_ratio={stage.step_ratio} "
            f"advance_window={args.advance_window} advance_patience={args.advance_patience}",
            flush=True,
        )
        for update in range(first_update, args.max_updates_per_stage + 1):
            obs_buf=[]; action_buf=[]; reward_buf=[]; next_obs_buf=[]; done_buf=[]; valid_buf=[]
            for _ in range(args.horizon):
                with torch.no_grad():
                    action, _, next_state = actor.sample_step(obs_now, state)
                next_obs, reward, done, info = env.step(action)
                valid = info["valid"].bool()
                obs_buf.append(obs_now); action_buf.append(action); reward_buf.append(reward); next_obs_buf.append(next_obs); done_buf.append((done | ~valid).float()); valid_buf.append(valid)
                state = actor.mask_state(actor.detach_state(next_state), info["active"].bool())
                obs_now = next_obs
                if env.all_inactive():
                    break
            obs = torch.stack(obs_buf); action = torch.stack(action_buf); reward = torch.stack(reward_buf); next_obs = torch.stack(next_obs_buf); done = torch.stack(done_buf); valid = torch.stack(valid_buf).bool()
            summary = env.pop_completed_summary()
            summary_added = False
            if summary:
                window.append(summary)
                summary_added = True
            valid_count = int(valid.sum().item())
            total_valid += valid_count
            min_valid_count = max(2, int(valid.numel() * max(0.0, float(args.min_valid_fraction))))
            low_valid = valid_count < min_valid_count
            if not low_valid:
                replay.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, valid=valid.float())
            losses = []
            if replay.valid_transitions >= args.learning_starts:
                losses = sac_updates(actor, q1, q2, target_actor, target_q1, target_q2, actor_opt, critic_opt, log_alpha, alpha_opt, replay, args, device)
            rolling = average_window(window)
            gate_ok = False
            display_streak = pass_streak
            if summary_added and len(window) == args.advance_window:
                gate_ok, _ = advancement_satisfied(stage, rolling)
                display_streak = pass_streak + 1 if gate_ok else 0
            gate = format_gate(stage, rolling, len(window), args.advance_window, display_streak, args.advance_patience)
            q_loss = sum(x["q"] for x in losses) / len(losses) if losses else float("nan")
            actor_loss = sum(x["actor"] for x in losses) / len(losses) if losses else float("nan")
            alpha = float(log_alpha.exp().detach().cpu())
            reward_mean = reward[valid].mean().item() if valid_count else float("nan")
            decision_max = int(env.steps.max().item())
            skip_text = f" skip=low_valid({valid_count}<{min_valid_count})" if low_valid else ""
            if low_valid or update % max(1, int(args.log_interval)) == 0 or summary_added:
                print(
                    format_training_line(
                        stage_index=stage_index,
                        update=update,
                        valid_steps=total_valid,
                        decision=decision_max,
                        decision_limit=stage.decision_limit,
                        reward_mean=reward_mean,
                        loss_text=f"q={q_loss:.4f} pi={actor_loss:.4f} alpha={alpha:.3f}",
                        rolling=rolling,
                        gate=gate,
                        extra=f"replay={replay.valid_transitions}/{replay.transitions}{skip_text}",
                    ),
                    flush=True,
                )
            append_jsonl(metrics_log_path, {
                "stage": stage_index,
                "stage_name": stage.name,
                "update": update,
                "status": "skip_low_valid" if low_valid else "running",
                "valid_steps": total_valid,
                "valid_count": valid_count,
                "min_valid_count": min_valid_count,
                "replay_valid": replay.valid_transitions,
                "replay_slots": replay.transitions,
                "decision": decision_max,
                "decision_limit": stage.decision_limit,
                "reward_mean": reward_mean,
                "q_loss": q_loss,
                "actor_loss": actor_loss,
                "alpha": alpha,
                "episodes": rolling.get("episodes", 0.0),
                "metrics": rolling,
                "gate": gate,
                "summary_added": summary_added,
                "pass_streak": display_streak,
            })
            if low_valid:
                obs_now = env.reset()
                state = actor.initial_state(args.num_envs, device)
                continue
            if env.all_inactive():
                obs_now = env.reset()
                state = actor.initial_state(args.num_envs, device)
            if update % args.checkpoint_interval == 0:
                save_checkpoint(stage_dir/"checkpoint.pt", actor=actor, q1=q1, q2=q2, target_q1=target_q1, target_q2=target_q2, log_alpha=log_alpha, stage=stage, stage_update=update, total_valid_steps=total_valid, args=args, metrics=rolling, status="running")
            if not args.no_auto_advance and summary_added and len(window) == args.advance_window:
                ok, reason = advancement_satisfied(stage, rolling)
                pass_streak = pass_streak + 1 if ok else 0
                if ok:
                    if pass_streak < max(1, args.advance_patience):
                        continue
                    advanced = True
                    save_checkpoint(stage_dir/"final_checkpoint.pt", actor=actor, q1=q1, q2=q2, target_q1=target_q1, target_q2=target_q2, log_alpha=log_alpha, stage=stage, stage_update=update, total_valid_steps=total_valid, args=args, metrics=rolling, status="advanced")
                    curriculum_log.append({"stage":stage_index,"name":stage.name,"update":update,"status":"advanced","reason":reason,"metrics":rolling})
                    (run/"curriculum_state.json").write_text(json.dumps(curriculum_log, indent=2), encoding="utf-8")
                    print(f"[advance] stage={stage_index} {reason}", flush=True)
                    break
        if not advanced:
            rolling = average_window(window)
            save_checkpoint(stage_dir/"final_checkpoint.pt", actor=actor, q1=q1, q2=q2, target_q1=target_q1, target_q2=target_q2, log_alpha=log_alpha, stage=stage, stage_update=args.max_updates_per_stage, total_valid_steps=total_valid, args=args, metrics=rolling, status="stalled")
            curriculum_log.append({"stage":stage_index,"name":stage.name,"update":args.max_updates_per_stage,"status":"stalled","metrics":rolling})
            (run/"curriculum_state.json").write_text(json.dumps(curriculum_log, indent=2), encoding="utf-8")
            print(f"[stalled] stage={stage_index}; stopping curriculum", flush=True)
            break
        resume_update = 0
    print(run, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

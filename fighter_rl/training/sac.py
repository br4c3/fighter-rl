import json
import math
import random
import time
from collections import deque
from pathlib import Path

import torch
from torch import nn

from fighter_rl.envs.loiter import CompetitionLoiterCurriculumEnv
from fighter_rl.models.sac import (
    FastAIPSACActor,
    FastAIPSACCritic,
    get_sac_profile,
    soft_update,
)
from fighter_rl.training.ppo import (
    action_diagnostics,
    average_window,
    deterministic_curriculum_evaluation,
    format_gate,
    format_action_diagnostics,
    format_training_line,
    stage_env_config,
)
from fighter_rl.training.stages import LoiterStage, advancement_satisfied, load_stages
from fighter_rl.utils.config import load_training_config
from fighter_rl.utils.experiment_record import append_jsonl, write_experiment_manifest


class SequenceReplay:
    def __init__(self, capacity_chunks=64, seq_len=None):
        self.capacity_chunks = max(1, int(capacity_chunks))
        self.seq_len = int(seq_len) if seq_len else None
        self.chunks = []
        self.position = 0

    def _fix_length(self, chunk):
        if self.seq_len is None:
            self.seq_len = int(next(iter(chunk.values())).shape[0])
        seq_len = int(self.seq_len)
        current = int(next(iter(chunk.values())).shape[0])

        if current == seq_len:
            return chunk

        fixed = {}

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

    def add(self, **chunk):
        stored = self._fix_length({key: value.detach().clone() for key, value in chunk.items()})

        if len(self.chunks) < self.capacity_chunks:
            self.chunks.append(stored)
        else:
            self.chunks[self.position] = stored
            self.position = (self.position + 1) % self.capacity_chunks

    @property
    def transitions(self):
        if not self.chunks:
            return 0

        return sum(int(item["obs"].shape[0] * item["obs"].shape[1]) for item in self.chunks)

    @property
    def valid_transitions(self):
        if not self.chunks:
            return 0

        return sum(int(item["valid"].sum().item()) for item in self.chunks)

    def sample(self, batch_sequences, device):
        if not self.chunks:
            raise RuntimeError("replay buffer is empty")

        by_chunk = {}

        for _ in range(int(batch_sequences)):
            ci = random.randrange(len(self.chunks))
            n = self.chunks[ci]["obs"].shape[1]
            ei = random.randrange(n)
            by_chunk.setdefault(ci, []).append(ei)
        out = {}

        for ci, env_indices in by_chunk.items():
            idx = torch.as_tensor(
                env_indices, dtype=torch.long, device=self.chunks[ci]["obs"].device
            )

            for key, value in self.chunks[ci].items():
                out.setdefault(key, []).append(value[:, idx].to(device))
        return {key: torch.cat(parts, dim=1) for key, parts in out.items()}


def reset_learned_parameters(module):
    def reset(child):
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()

    module.apply(reset)


def save_checkpoint(
    path,
    *,
    actor,
    q1,
    q2,
    target_q1,
    target_q2,
    log_alpha,
    stage,
    stage_update,
    total_valid_steps,
    cfg,
    metrics,
    status,
):
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
            "config": vars(cfg),
        },
        path,
    )


def load_resume(path, actor, q1, q2, target_q1, target_q2, log_alpha, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    variant = payload.get("variant")

    if variant and variant != actor.profile.variant:
        raise ValueError(
            f"Resume variant mismatch: checkpoint={variant}, requested={actor.profile.variant}"
        )
    actor.load_state_dict(payload["actor"])
    q1.load_state_dict(payload["q1"])
    q2.load_state_dict(payload["q2"])
    target_q1.load_state_dict(payload.get("target_q1", payload["q1"]))
    target_q2.load_state_dict(payload.get("target_q2", payload["q2"]))

    if "log_alpha" in payload:
        with torch.no_grad():
            log_alpha.copy_(torch.as_tensor(payload["log_alpha"], device=device))
    return payload


def sac_updates(
    actor,
    q1,
    q2,
    target_actor,
    target_q1,
    target_q2,
    actor_opt,
    critic_opt,
    log_alpha,
    alpha_opt,
    replay,
    cfg,
    device,
    *,
    update_actor=True,
):
    losses = []

    for _ in range(cfg.updates_per_rollout):
        batch = replay.sample(cfg.batch_sequences, device)
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
            # fmt: off
            target = reward + cfg.gamma*(1.0 - done)*valid*(tq - alpha*next_logp)
            # fmt: on
        q1_pred = q1.forward_sequence(obs, action)
        q2_pred = q2.forward_sequence(obs, action)
        mask = valid > 0

        if not bool(mask.any()):
            continue

        q_loss = (q1_pred[mask] - target[mask]).square().mean() + (
            q2_pred[mask] - target[mask]
        ).square().mean()

        if not bool(torch.isfinite(q_loss)):
            continue

        critic_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 5.0)

        if not bool(torch.isfinite(critic_grad)):
            critic_opt.zero_grad(set_to_none=True)
            continue

        critic_opt.step()

        if not update_actor:
            soft_update(target_q1, q1, cfg.tau)
            soft_update(target_q2, q2, cfg.tau)
            losses.append(
                {
                    "q": float(q_loss.detach().cpu()),
                    "actor": float("nan"),
                    "alpha": float(log_alpha.exp().detach().cpu()),
                }
            )
            continue

        new_action, logp, _, actor_logits = actor.sample_sequence(obs, return_logits=True)
        q_pi = torch.minimum(
            q1.forward_sequence(obs, new_action), q2.forward_sequence(obs, new_action)
        )
        alpha = log_alpha.exp().detach()
        actor_mean = actor_logits[..., :4][mask].square().mean()
        # fmt: off
        actor_loss = ((alpha*logp - q_pi)[mask]).mean() + cfg.action_mean_l2_coef*actor_mean
        # fmt: on

        if not bool(torch.isfinite(actor_loss)):
            continue

        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = nn.utils.clip_grad_norm_(actor.parameters(), 5.0)

        if not bool(torch.isfinite(actor_grad)):
            actor_opt.zero_grad(set_to_none=True)
            continue

        actor_opt.step()

        # fmt: off
        alpha_loss = -(log_alpha*(logp.detach() + cfg.target_entropy))[mask].mean()
        # fmt: on

        if not bool(torch.isfinite(alpha_loss)):
            continue

        alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_grad = nn.utils.clip_grad_norm_([log_alpha], 5.0)

        if not bool(torch.isfinite(alpha_grad)):
            alpha_opt.zero_grad(set_to_none=True)
            continue

        alpha_opt.step()
        with torch.no_grad():
            alpha_min = max(1e-6, float(getattr(cfg, "alpha_min", 0.05)))
            log_alpha.clamp_(min=math.log(alpha_min))

        soft_update(target_actor, actor, cfg.tau)
        soft_update(target_q1, q1, cfg.tau)
        soft_update(target_q2, q2, cfg.tau)
        losses.append(
            {
                "q": float(q_loss.detach().cpu()),
                "actor": float(actor_loss.detach().cpu()),
                "alpha": float(log_alpha.exp().detach().cpu()),
            }
        )
    return losses


def main():
    cfg = load_training_config("configs/sac_lstm.json")
    cfg.reset_critic_stages = list(getattr(cfg, "reset_critic_stages", []))
    cfg.reset_alpha_stages = list(getattr(cfg, "reset_alpha_stages", []))
    cfg.stage_alpha = float(getattr(cfg, "stage_alpha", 0.20))
    cfg.alpha_min = float(getattr(cfg, "alpha_min", 0.05))
    cfg.critic_warmup_updates_on_reset = int(getattr(cfg, "critic_warmup_updates_on_reset", 0))
    profile = get_sac_profile(cfg.variant)

    if cfg.horizon <= 0:
        cfg.horizon = profile.max_seq_len if profile.use_lstm else 32
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    if cfg.device == "cpu":
        torch.set_num_threads(1)
    device = torch.device(cfg.device)
    stages = load_stages(schedule=cfg.stage_schedule)

    if cfg.stop_stage is None:
        cfg.stop_stage = len(stages) - 1
    if cfg.stop_stage >= len(stages):
        raise ValueError(f"stop-stage {cfg.stop_stage} exceeds available stages {len(stages)-1}")

    actor = FastAIPSACActor(profile).to(device)
    target_actor = FastAIPSACActor(profile).to(device)
    q1 = FastAIPSACCritic(profile).to(device)
    q2 = FastAIPSACCritic(profile).to(device)
    target_q1 = FastAIPSACCritic(profile).to(device)
    target_q2 = FastAIPSACCritic(profile).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())
    log_alpha = torch.tensor(math.log(float(cfg.stage_alpha)), device=device, requires_grad=True)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=cfg.critic_lr)
    alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr)

    start_stage = cfg.start_stage
    resume_update = 0
    resume_stage_name = None
    total_valid = 0

    if cfg.resume:
        payload = load_resume(cfg.resume, actor, q1, q2, target_q1, target_q2, log_alpha, device)
        target_actor.load_state_dict(actor.state_dict())

        if cfg.reset_alpha_on_resume:
            with torch.no_grad():
                log_alpha.fill_(math.log(float(cfg.stage_alpha)))
        saved_stage = int(payload.get("stage_index", start_stage))
        resume_stage_name = payload.get("stage_name")

        if not cfg.resume_weights_only:
            if cfg.start_stage == 0:
                start_stage = saved_stage
                resume_update = int(payload.get("stage_update", 0))
            elif cfg.start_stage == saved_stage:
                resume_update = int(payload.get("stage_update", 0))
        total_valid = 0 if cfg.resume_weights_only else int(payload.get("steps", 0))

    run = cfg.output / f"{cfg.variant}_{cfg.target_maneuver}_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(
        json.dumps(vars(cfg) | {"profile": profile.as_metadata()}, default=str, indent=2),
        encoding="utf-8",
    )
    write_experiment_manifest(
        run,
        trainer="train_fast_aip_sac",
        cfg=cfg,
        profile=profile,
        stages=stages,
        extra_code_files=[Path(__file__)],
    )
    metrics_log_path = run / "metrics.jsonl"
    replay = SequenceReplay(cfg.replay_chunks, seq_len=cfg.horizon)
    curriculum_log = []

    for stage_index in range(start_stage, cfg.stop_stage + 1):
        stage = stages[stage_index]
        reward_regime_changed = stage_index != start_stage or (
            stage_index == start_stage
            and resume_stage_name is not None
            and resume_stage_name != stage.name
        )

        if cfg.reset_replay_on_stage and stage_index != start_stage:
            replay = SequenceReplay(cfg.replay_chunks, seq_len=cfg.horizon)

        reset_critic = reward_regime_changed and stage_index in set(cfg.reset_critic_stages)
        reset_alpha = reward_regime_changed and stage_index in set(cfg.reset_alpha_stages)

        if reset_critic:
            reset_learned_parameters(q1)
            reset_learned_parameters(q2)
            target_q1.load_state_dict(q1.state_dict())
            target_q2.load_state_dict(q2.state_dict())
            critic_opt = torch.optim.Adam(
                list(q1.parameters()) + list(q2.parameters()), lr=cfg.critic_lr
            )
            print(f"[stage-reset] stage={stage_index} critic", flush=True)

        if reset_alpha:
            with torch.no_grad():
                log_alpha.fill_(math.log(float(cfg.stage_alpha)))
            alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr)
            print(
                f"[stage-reset] stage={stage_index} alpha={float(cfg.stage_alpha):.3f}",
                flush=True,
            )

        stage_dir = run / f"stage_{stage_index:02d}_{stage.name}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        env = CompetitionLoiterCurriculumEnv(
            stage,
            num_envs=cfg.num_envs,
            device=device,
            domain_randomization=not cfg.no_domain_randomization,
            target_maneuver=cfg.target_maneuver,
            temporal_frames=profile.temporal_frames,
            include_previous_action=profile.include_previous_action,
        )
        obs_now = env.reset()

        if obs_now.shape[-1] != profile.obs_dim:
            raise RuntimeError(
                f"Observation mismatch: got {obs_now.shape[-1]}, expected {profile.obs_dim}"
            )

        state = actor.initial_state(cfg.num_envs, device)
        window = deque(maxlen=cfg.advance_window)

        first_update = resume_update + 1 if stage_index == start_stage else 1
        critic_warmup_until = (
            first_update + int(cfg.critic_warmup_updates_on_reset) if reset_critic else first_update
        )
        advanced = False
        pass_streak = 0

        print(
            f"[stage {stage_index}/{cfg.stop_stage}] {stage.name} variant={cfg.variant} "
            f"obs={profile.obs_dim} frames={profile.temporal_frames} target={cfg.target_maneuver} "
            f"decision_limit={stage.decision_limit} step_ratio={stage.step_ratio} "
            f"advance_window={cfg.advance_window} advance_patience={cfg.advance_patience}",
            flush=True,
        )

        for update in range(first_update, cfg.max_updates_per_stage + 1):
            obs_buf = []
            action_buf = []
            reward_buf = []
            next_obs_buf = []
            done_buf = []
            valid_buf = []
            logp_buf = []
            logits_buf = []

            for _ in range(cfg.horizon):
                with torch.no_grad():
                    action, logp, next_state, actor_logits = actor.sample_step(
                        obs_now, state, return_logits=True
                    )

                next_obs, reward, done, info = env.step(action)
                valid = info["valid"].bool()

                obs_buf.append(obs_now)
                action_buf.append(action)
                reward_buf.append(reward)
                next_obs_buf.append(next_obs)
                done_buf.append((done | ~valid).float())
                valid_buf.append(valid)
                logp_buf.append(logp)
                logits_buf.append(actor_logits)

                state = actor.mask_state(actor.detach_state(next_state), info["active"].bool())
                obs_now = next_obs

                if env.all_inactive():
                    break

            obs = torch.stack(obs_buf)
            action = torch.stack(action_buf)
            reward = torch.stack(reward_buf)
            next_obs = torch.stack(next_obs_buf)
            done = torch.stack(done_buf)
            valid = torch.stack(valid_buf).bool()
            rollout_logp = torch.stack(logp_buf)
            rollout_logits = torch.stack(logits_buf)

            valid_count = int(valid.sum().item())
            action_metrics = action_diagnostics(action, valid)
            if valid_count:
                action_metrics["policy_entropy_proxy"] = float(
                    -rollout_logp[valid].mean().detach().cpu()
                )
                action_metrics["policy_log_std_mean"] = float(
                    rollout_logits[..., 4:][valid].mean().detach().cpu()
                )
            action_text = format_action_diagnostics(action_metrics)
            total_valid += valid_count

            min_valid_count = max(2, int(valid.numel() * max(0.0, float(cfg.min_valid_fraction))))
            low_valid = valid_count < min_valid_count
            batch_complete = env.all_inactive() or low_valid

            summary = env.pop_completed_summary() if batch_complete else None
            summary_added = False

            if summary:
                window.append(summary)
                summary_added = True

            if not low_valid:
                replay.add(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=done,
                    valid=valid.float(),
                )

            losses = []

            if replay.valid_transitions >= cfg.learning_starts:
                losses = sac_updates(
                    actor,
                    q1,
                    q2,
                    target_actor,
                    target_q1,
                    target_q2,
                    actor_opt,
                    critic_opt,
                    log_alpha,
                    alpha_opt,
                    replay,
                    cfg,
                    device,
                    update_actor=update >= critic_warmup_until,
                )

            rolling = average_window(window)
            gate_ok = False
            display_streak = pass_streak

            if summary_added and len(window) == cfg.advance_window:
                gate_ok, _ = advancement_satisfied(stage, rolling)
                display_streak = pass_streak + 1 if gate_ok else 0

            gate = format_gate(
                stage,
                rolling,
                len(window),
                cfg.advance_window,
                display_streak,
                cfg.advance_patience,
            )
            q_values = [x["q"] for x in losses if math.isfinite(x["q"])]
            actor_values = [x["actor"] for x in losses if math.isfinite(x["actor"])]
            q_loss = sum(q_values) / len(q_values) if q_values else float("nan")
            actor_loss = sum(actor_values) / len(actor_values) if actor_values else float("nan")
            alpha = float(log_alpha.exp().detach().cpu())
            reward_mean = reward[valid].mean().item() if valid_count else float("nan")
            decision_max = int(env.steps.max().item())

            skip_text = f" skip=low_valid({valid_count}<{min_valid_count})" if low_valid else ""

            if low_valid or update % max(1, int(cfg.log_interval)) == 0 or summary_added:
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
                        extra=(
                            f"{action_text} replay={replay.valid_transitions}/"
                            f"{replay.transitions}{skip_text}"
                        ),
                    ),
                    flush=True,
                )

            append_jsonl(
                metrics_log_path,
                {
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
                    "actions": action_metrics,
                    "episodes": rolling.get("episodes", 0.0),
                    "metrics": rolling,
                    "gate": gate,
                    "summary_added": summary_added,
                    "batch_complete": batch_complete,
                    "pass_streak": display_streak,
                },
            )

            if low_valid:
                obs_now = env.reset()
                state = actor.initial_state(cfg.num_envs, device)

                if not cfg.no_auto_advance and summary_added and len(window) == cfg.advance_window:
                    ok, reason = advancement_satisfied(stage, rolling)
                    pass_streak = pass_streak + 1 if ok else 0

                    if ok and pass_streak >= max(1, cfg.advance_patience):
                        eval_metrics = deterministic_curriculum_evaluation(
                            actor, stage, profile, cfg, device, sac=True
                        )
                        eval_ok, eval_reason = advancement_satisfied(stage, eval_metrics)
                        if not eval_ok:
                            pass_streak = 0
                            print(f"[eval-block] stage={stage_index} {eval_reason}", flush=True)
                            continue
                        advanced = True
                        save_checkpoint(
                            stage_dir / "final_checkpoint.pt",
                            actor=actor,
                            q1=q1,
                            q2=q2,
                            target_q1=target_q1,
                            target_q2=target_q2,
                            log_alpha=log_alpha,
                            stage=stage,
                            stage_update=update,
                            total_valid_steps=total_valid,
                            cfg=cfg,
                            metrics=rolling,
                            status="advanced",
                        )
                        curriculum_log.append(
                            {
                                "stage": stage_index,
                                "name": stage.name,
                                "update": update,
                                "status": "advanced",
                                "reason": f"train:{reason}; eval:{eval_reason}",
                                "metrics": rolling,
                                "evaluation_metrics": eval_metrics,
                            }
                        )
                        (run / "curriculum_state.json").write_text(
                            json.dumps(curriculum_log, indent=2), encoding="utf-8"
                        )
                        print(f"[advance] stage={stage_index} {reason}", flush=True)
                        break

                continue

            if env.all_inactive():
                obs_now = env.reset()
                state = actor.initial_state(cfg.num_envs, device)

            if update % cfg.checkpoint_interval == 0:
                save_checkpoint(
                    stage_dir / "checkpoint.pt",
                    actor=actor,
                    q1=q1,
                    q2=q2,
                    target_q1=target_q1,
                    target_q2=target_q2,
                    log_alpha=log_alpha,
                    stage=stage,
                    stage_update=update,
                    total_valid_steps=total_valid,
                    cfg=cfg,
                    metrics=rolling,
                    status="running",
                )

            if not cfg.no_auto_advance and summary_added and len(window) == cfg.advance_window:
                ok, reason = advancement_satisfied(stage, rolling)
                pass_streak = pass_streak + 1 if ok else 0

                if ok:
                    if pass_streak < max(1, cfg.advance_patience):
                        continue

                    eval_metrics = deterministic_curriculum_evaluation(
                        actor, stage, profile, cfg, device, sac=True
                    )
                    eval_ok, eval_reason = advancement_satisfied(stage, eval_metrics)
                    if not eval_ok:
                        pass_streak = 0
                        print(f"[eval-block] stage={stage_index} {eval_reason}", flush=True)
                        continue
                    advanced = True
                    save_checkpoint(
                        stage_dir / "final_checkpoint.pt",
                        actor=actor,
                        q1=q1,
                        q2=q2,
                        target_q1=target_q1,
                        target_q2=target_q2,
                        log_alpha=log_alpha,
                        stage=stage,
                        stage_update=update,
                        total_valid_steps=total_valid,
                        cfg=cfg,
                        metrics=rolling,
                        status="advanced",
                    )
                    curriculum_log.append(
                        {
                            "stage": stage_index,
                            "name": stage.name,
                            "update": update,
                            "status": "advanced",
                            "reason": f"train:{reason}; eval:{eval_reason}",
                            "metrics": rolling,
                            "evaluation_metrics": eval_metrics,
                        }
                    )
                    (run / "curriculum_state.json").write_text(
                        json.dumps(curriculum_log, indent=2), encoding="utf-8"
                    )
                    print(f"[advance] stage={stage_index} {reason}", flush=True)
                    break

        if not advanced:
            rolling = average_window(window)
            save_checkpoint(
                stage_dir / "final_checkpoint.pt",
                actor=actor,
                q1=q1,
                q2=q2,
                target_q1=target_q1,
                target_q2=target_q2,
                log_alpha=log_alpha,
                stage=stage,
                stage_update=cfg.max_updates_per_stage,
                total_valid_steps=total_valid,
                cfg=cfg,
                metrics=rolling,
                status="stalled",
            )
            curriculum_log.append(
                {
                    "stage": stage_index,
                    "name": stage.name,
                    "update": cfg.max_updates_per_stage,
                    "status": "stalled",
                    "metrics": rolling,
                }
            )
            (run / "curriculum_state.json").write_text(
                json.dumps(curriculum_log, indent=2), encoding="utf-8"
            )
            print(f"[stalled] stage={stage_index}; stopping curriculum", flush=True)
            break

        resume_update = 0
    print(run, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

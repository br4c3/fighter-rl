from fighter_rl.training.launcher import launch

if __name__ == "__main__":
    raise SystemExit(launch("fighter_rl.training.ppo", "configs/ppo_lstm.json"))

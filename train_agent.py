import argparse
import sys
from pathlib import Path
from typing import Callable

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

from rl.env import AudioEffectEnv


def make_env(seed: int = None) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = AudioEffectEnv()
        if seed is not None:
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agent for audio style transfer with parallel environments"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100000,
        help="Total timesteps to train"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel environments"
    )
    args = parser.parse_args()

    # Prepare directories
    models_dir = Path("models")
    best_dir = models_dir / "best"
    ckpt_dir = models_dir / "checkpoints"
    logs_dir = models_dir / "logs"
    for d in (best_dir, ckpt_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Create vectorized train env (subproc or dummy)
    env_fns = [make_env(seed=i) for i in range(args.workers)]
    # Use DummyVecEnv if you prefer single-process stability
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, filename=str(logs_dir / "monitor.csv"))

    # Create evaluation env wrapped the same way
    eval_fns = [make_env(seed=999)]
    eval_env = DummyVecEnv(eval_fns)
    eval_env = VecMonitor(eval_env, filename=str(logs_dir / "eval_monitor.csv"))

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(logs_dir),
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=300,
        save_path=str(ckpt_dir),
        name_prefix="rl_checkpoint"
    )

    # Instantiate or resume model
    # Auto-resume from last checkpoint if exists
    import glob
    import re

    def extract_ckpt_number(path: str) -> int:
        match = re.search(r"rl_checkpoint_(\d+)_steps", Path(path).stem)
        return int(match.group(1)) if match else -1

    ckpts = [
        p for p in glob.glob(str(ckpt_dir / "rl_checkpoint_*.zip"))
        if extract_ckpt_number(p) >= 0
    ]
    ckpts = sorted(ckpts, key=extract_ckpt_number)
    if ckpts:
        last_ckpt = ckpts[-1]
        print(f"Resuming from checkpoint {last_ckpt}")
        model = PPO.load(last_ckpt, env=vec_env, tensorboard_log=str(logs_dir / "tb"), verbose=1)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=str(logs_dir / "tb")
        )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[eval_callback, checkpoint_callback]
        )
    except KeyboardInterrupt:
        print("Training interrupted; saving current model...")
        model.save(str(models_dir / "ppo_audio_effect_interrupted"))
        sys.exit(0)

    # Final save
    model.save(str(models_dir / "ppo_audio_effect"))
    print("✅ Training complete. Model saved to models/ppo_audio_effect.zip")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
from typing import cast
from gym import Env
import numpy as np
import soundfile as sf
import librosa
import openl3
from stable_baselines3 import PPO

from rl.etalon import ETALON_VECTOR      # same as in training
from rl.env import AudioEffectEnv         # identical environment to train_agent.py


def apply_effects(audio: np.ndarray, sr: int, action: np.ndarray) -> np.ndarray:
    bass_gain, tempo_factor, reverb_amount, pitch_steps = action
    # 1) Bass gain
    y = audio.astype(np.float32) * (10.0 ** (bass_gain / 20.0))
    # 2) Time-stretch
    try:
        y = librosa.effects.time_stretch(y, rate=float(tempo_factor))
    except Exception:
        pass
    # 3) Reverb (simple moving‐average)
    k = max(1, int(reverb_amount * 10))
    y = np.convolve(y, np.ones(k) / k, mode="same")
    # 4) Pitch shift
    try:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=float(pitch_steps))
    except Exception:
        pass
    return y


def main():
    parser = argparse.ArgumentParser(
        description="Apply trained RL audio style-transfer model to a WAV file"
    )
    parser.add_argument("--input",  type=Path, default=Path("data/raw/sample.wav"))
    parser.add_argument("--output", type=Path, default=Path("out/pugach_sample.wav"))
    parser.add_argument("--model",  type=Path, default=Path("models/rl_checkpoint_3600_steps.zip"))
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # 1) load and mono-convert
    audio, sr = sf.read(str(args.input))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # 2) compute OpenL3 embedding and center
    emb, _ = openl3.get_audio_embedding(
        audio, sr=sr,
        input_repr="mel256",
        content_type="music",
        embedding_size=512,
    )
    vec = emb.mean(axis=0).astype(np.float32)
    obs = vec - ETALON_VECTOR

    # 3) load model into matching env
    from stable_baselines3.common.vec_env import DummyVecEnv
    single_env = DummyVecEnv([
        lambda: cast(Env, AudioEffectEnv())
    ])
    model = PPO.load(str(args.model), env=single_env)

    # 4) predict action
    action, _ = model.predict(obs, deterministic=True)

    # 5) apply effects & save
    stylized = apply_effects(audio, sr, action)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    sf.write(str(args.output), stylized, sr)

    print(f"✅ Stylized audio saved to {args.output}")
    print(
        f"Applied effects: bass_gain={action[0]:.2f} dB, "
        f"tempo_factor={action[1]:.2f}×, "
        f"reverb_amount={action[2]:.2f}, "
        f"pitch_steps={action[3]:.2f}"
    )


if __name__ == "__main__":
    main()

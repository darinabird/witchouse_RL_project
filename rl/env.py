import gym
import random
import numpy as np
import soundfile as sf
import librosa
import openl3
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from rl.etalon import ETALON_VECTOR  # убедитесь, что имя переменной совпадает с изменённым
from rl.compute_electronic_etalon import MEAN_ELEC_VECTOR

class AudioEffectEnv(gym.Env):
    """
    Custom Gym environment for audio style transfer via RL.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, data_dir: str = 'data/processed/target_wav'):
        super().__init__()
        # Load etalon vectors
        self._witchouse = ETALON_VECTOR
        self._electronic = MEAN_ELEC_VECTOR

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=np.array([-10.0, 0.5, 0.0, -5.0], dtype=np.float32),
            high=np.array([10.0, 2.0, 1.0, 5.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self._witchouse.shape,
            dtype=np.float32
        )

        # Directory of WAV files
        self.data_dir = Path(data_dir)
        self.audio: np.ndarray
        self.sr: int

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment and returns initial observation and info.

        Returns:
            obs: Initial observation
            info: Empty info dict for compatibility
        """
        if seed is not None:
            random.seed(seed)

        wav_path = self._get_random_wav()
        self.audio, self.sr = sf.read(wav_path)
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Applies action, computes next state, reward, and termination flags.

        Returns:
            obs: Observation
            reward: Reward value
            terminated: True if episode ended due to task
            truncated: True if episode ended due to time limit or truncation
            info: Additional info dict
        """
        bass_gain, tempo_factor, reverb_amount, pitch_steps = action
        y = self.audio.astype(np.float32) * (10.0 ** (bass_gain / 20.0))
        y = librosa.effects.time_stretch(y, rate=tempo_factor)
        kernel_size = int(reverb_amount * 10) + 1
        y = np.convolve(y, np.ones(kernel_size) / kernel_size, mode="same")
        try:
            y = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=float(pitch_steps))
        except Exception:
            pass

        emb, _ = openl3.get_audio_embedding(
            y, sr=self.sr, input_repr="mel256",
            content_type="music", embedding_size=512
        )
        new_vec = emb.mean(axis=0)

        dist_w = np.linalg.norm(new_vec - self._witchouse)
        dist_e = np.linalg.norm(new_vec - self._electronic)
        reward = - (dist_w + dist_e)

        obs = (new_vec - self._witchouse).astype(np.float32)
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        emb, _ = openl3.get_audio_embedding(
            self.audio, sr=self.sr, input_repr="mel256",
            content_type="music", embedding_size=512
        )
        vec = emb.mean(axis=0)
        return (vec - self._witchouse).astype(np.float32)

    def _get_random_wav(self) -> str:
        files = list(self.data_dir.glob('*.wav'))
        if not files:
            raise FileNotFoundError(f"No WAV files found in {self.data_dir}")
        return str(random.choice(files))

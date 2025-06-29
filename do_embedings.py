import openl3
import soundfile as sf
from pathlib import Path
import numpy as np

# Папка с .wav файлами
input_dir = Path("data/processed/instrumentals")
# Куда сохранять эмбеддинги
output_dir = Path("embeddings/electronic")
output_dir.mkdir(parents=True, exist_ok=True)

for wav in input_dir.glob("*.wav"):
    audio, sr = sf.read(str(wav))
    emb, _ = openl3.get_audio_embedding(audio, sr, content_type="music", embedding_size=512)
    mean_emb = emb.mean(axis=0)
    np.save(output_dir / f"{wav.stem}.npy", mean_emb)
    print(f"Saved embedding for {wav.name}")

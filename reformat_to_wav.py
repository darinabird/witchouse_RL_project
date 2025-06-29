from pydub import AudioSegment
from pathlib import Path

source_dir = Path("data")
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

for mp3_path in source_dir.glob("*.mp3"):
    wav_path = output_dir / (mp3_path.stem + ".wav")
    track = AudioSegment.from_mp3(mp3_path)
    track = track.set_channels(1).set_frame_rate(44100)
    track.export(wav_path, format="wav")
    print(f"Converted: {mp3_path.name} → {wav_path.name}")

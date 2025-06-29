#!/usr/bin/env python3
"""
Script to separate instrumentals from WAV files using Spleeter in parallel (up to 4 workers).
"""
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def process_file(wav_path_str: str, output_dir_str: str, temp_dir_str: str):
    wav_path = Path(wav_path_str)
    output_dir = Path(output_dir_str)
    temp_dir = Path(temp_dir_str)

    # Call Spleeter to separate into vocals and accompaniment
    subprocess.run([
        "spleeter", "separate",
        "-p", "spleeter:2stems",
        "-o", str(temp_dir),
        str(wav_path)
    ], check=True)

    # Move the accompaniment (instrumental) to the output directory
    src = temp_dir / wav_path.stem / "accompaniment.wav"
    dest = output_dir / wav_path.name
    shutil.move(str(src), str(dest))

    # Clean up temporary folder for this file
    shutil.rmtree(temp_dir / wav_path.stem)


def main():
    input_dir = Path("data/processed/target_wav")
    output_dir = Path("data/processed/instrumentals")
    temp_dir = Path("data/processed/spleeter_temp")

    # Create necessary directories
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Collect all WAV files
    wav_files = list(input_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} files to process.")

    # Process in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                process_file,
                str(wav),
                str(output_dir),
                str(temp_dir)
            )
            for wav in wav_files
        ]
        for future in futures:
            future.result()  # will raise if subprocess failed

    # Remove the temporary directory
    shutil.rmtree(temp_dir)
    print("All instrumentals extracted to data/processed/instrumentals")


if __name__ == "__main__":
    main()

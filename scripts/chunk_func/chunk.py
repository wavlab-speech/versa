#!/usr/bin/env python3

# Copyright 2025 BoHao Su
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from versa.scorer_shared import (
    audio_loader_setup,
    load_audio,
    wav_normalize,
)


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(description="Chunk audios into fixed durations.")
    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="Wav.scp for generated waveforms, or a dir depending on --io.",
    )
    parser.add_argument(
        "--io",
        type=str,
        default="kaldi",
        choices=["kaldi", "soundfile", "dir"],
        help="IO interface to use.",
    )
    parser.add_argument(
        "--chunk_duration",
        type=float,
        default=3.0,
        help="Duration (sec) of each chunk window.",
    )
    parser.add_argument(
        "--hop_duration",
        type=float,
        default=None,
        help="Hop size (sec) between chunk starts. "
             "If None, equals --chunk_duration (non-overlap).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write chunked wav files.",
    )
    parser.add_argument(
        "--min_last_chunk",
        type=float,
        default=0.0,
        help="Minimum duration (sec) required to keep the final (short) chunk. "
             "Set >0 to drop very short tails.",
    )
    return parser

def main():
    args = get_parser().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.chunk_duration <= 0:
        raise ValueError("--chunk_duration must be > 0")

    hop_duration = args.hop_duration if args.hop_duration is not None else args.chunk_duration
    if hop_duration <= 0:
        raise ValueError("--hop_duration must be > 0")

    if args.min_last_chunk < 0:
        raise ValueError("--min_last_chunk must be >= 0")

    gen_files = audio_loader_setup(args.pred, args.io)
    if len(gen_files) == 0:
        raise FileNotFoundError("Not found any generated audio files from --pred with --io.")

    total_chunks = 0
    for key in tqdm(list(gen_files.keys()), desc="Chunking"):
        src_path = gen_files[key]
        try:
            sr, wav = load_audio(src_path, args.io)
            wav = wav_normalize(wav)
            if wav.ndim > 1:
                # Convert to mono if multichannel
                wav = np.mean(wav, axis=-1)
        except Exception as e:
            print(f"[WARN] Failed to load {key} from {src_path}: {e}")
            continue

        chunk_len = int(round(args.chunk_duration * sr))
        hop_len = int(round(hop_duration * sr))
        min_last_len = int(round(args.min_last_chunk * sr))

        if chunk_len <= 0 or hop_len <= 0:
            print(f"[WARN] Non-positive chunk/hop for key={key}; skipping.")
            continue

        n_samples = len(wav)
        if n_samples == 0:
            print(f"[WARN] Empty audio for key={key}; skipping.")
            continue

        # Iterate chunk start positions
        chunk_idx = 0
        start = 0
        while start < n_samples:
            end = start + chunk_len
            if end > n_samples:
                # last (short) chunk
                if (n_samples - start) < min_last_len:
                    break  # drop the tail if too short
                end = n_samples

            chunk = wav[start:end]
            if len(chunk) == 0:
                break

            # Include time range in filename for traceability
            t0 = start / sr
            t1 = end / sr
            out_name = f"{key}_chunk{chunk_idx:04d}_{t0:.3f}-{t1:.3f}.wav"
            out_path = output_dir / out_name

            try:
                sf.write(str(out_path), chunk, sr, subtype="PCM_16")
                total_chunks += 1
            except Exception as e:
                print(f"[WARN] Failed to write {out_path}: {e}")

            chunk_idx += 1
            start += hop_len

    print(f"Done. Wrote {total_chunks} chunks to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

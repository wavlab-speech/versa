#!/usr/bin/env python3

# Copyright 2025 BoHao Su
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Scorer Interface for Speech Evaluation with optional CHUNKED scoring."""

import argparse
import logging
import os
from pathlib import Path
import re

import numpy as np
import soundfile as sf
import torch
import yaml

from versa.scorer_shared import (
    audio_loader_setup,
    corpus_scoring,
    list_scoring,
    load_corpus_modules,
    load_score_modules,
    load_summary,
    load_audio,
    wav_normalize,
)


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(description="Speech Evaluation Interface")
    parser.add_argument(
        "--pred",
        type=str,
        help="Wav.scp for generated waveforms.",
    )
    parser.add_argument(
        "--score_config", type=str, default=None, help="Configuration of Score Config"
    )
    parser.add_argument(
        "--gt",
        type=str,
        default=None,
        help="Wav.scp for ground truth waveforms.",
    )
    parser.add_argument(
        "--text", type=str, default=None, help="Path of ground truth transcription."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path of directory to write the results.",
    )
    parser.add_argument(
        "--cache_folder", type=str, default=None, help="Path of cache saving"
    )
    parser.add_argument(
        "--use_gpu", type=bool, default=False, help="whether to use GPU if it can"
    )
    parser.add_argument(
        "--io",
        type=str,
        default="kaldi",
        choices=["kaldi", "soundfile", "dir"],
        help="io interface to use",
    )
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    parser.add_argument(
        "--rank",
        default=0,
        type=int,
        help="the overall rank in the batch processing, used to specify GPU rank",
    )
    parser.add_argument(
        "--no_match",
        action="store_true",
        help="Do not match the groundtruth and generated files.",
    )

    # ---------- NEW: chunking options ----------
    parser.add_argument(
        "--enable_chunking",
        action="store_true",
        help="If set, score on fixed-length chunks instead of full utterances.",
    )
    parser.add_argument(
        "--chunk_duration",
        type=float,
        default=0.5,
        help="Chunk window length in seconds.",
    )
    parser.add_argument(
        "--hop_duration",
        type=float,
        default=0.2,
        help="Hop size in seconds. If not set, equals --chunk_duration (no overlap).",
    )
    parser.add_argument(
        "--min_last_chunk",
        type=float,
        default=0.0,
        help="Keep final short tail only if >= this many seconds. 0 to keep any tail.",
    )
    parser.add_argument(
        "--chunk_tmp_dir",
        type=str,
        default=None,
        help="Directory to write temporary chunk wavs. "
             "Defaults to <output_file>.chunks or ./chunks when not provided.",
    )
    # -------------------------------------------

    return parser

def _write_wav(path: Path, wav: np.ndarray, sr: int):
    """Write mono PCM16 WAV safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    sf.write(str(path), wav, sr, subtype="PCM_16")


def _chunk_bounds(n_samples: int, sr: int, chunk_sec: float, hop_sec: float, min_last_sec: float):
    """Yield (start, end) sample indices for chunks covering [0, n_samples]."""
    chunk_len = int(round(chunk_sec * sr))
    hop_len = int(round(hop_sec * sr))
    min_last = int(round(min_last_sec * sr))
    if chunk_len <= 0 or hop_len <= 0:
        raise ValueError("chunk/hop must be > 0")
    start = 0
    while start < n_samples:
        end = start + chunk_len
        if end > n_samples:
            if n_samples - start < min_last:
                break
            end = n_samples
        yield start, end
        start += hop_len


def _chunk_pair_to_tmp(
    key: str,
    gen_path: str,
    gt_path: str | None,
    io: str,
    chunk_sec: float,
    hop_sec: float,
    min_last_sec: float,
    tmp_root: Path,
) -> tuple[dict, dict | None]:
    """
    Chunk a generated file (and optionally its GT pair) into aligned windows.
    - If GT is provided, both are truncated to the MIN of their lengths, then chunked
      on the same boundaries for fair, aligned scoring.
    Returns:
      gen_chunks: {new_key -> wavpath}
      gt_chunks:  {new_key -> wavpath} or None
    """
    # Load gen
    gen_sr, gen_wav = load_audio(gen_path, io)
    gen_wav = wav_normalize(gen_wav)
    if gen_wav.ndim > 1:
        gen_wav = np.mean(gen_wav, axis=-1)
    n_gen = len(gen_wav)

    # Load gt (optional)
    if gt_path is not None:
        gt_sr, gt_wav = load_audio(gt_path, io)
        gt_wav = wav_normalize(gt_wav)
        if gt_wav.ndim > 1:
            gt_wav = np.mean(gt_wav, axis=-1)
        # Resample check (assume same SR; if not, we must resample – here we assert)
        if gt_sr != gen_sr:
            raise ValueError(f"SR mismatch for key={key}: gen {gen_sr} vs gt {gt_sr}")
        n_gt = len(gt_wav)
        n_use = min(n_gen, n_gt)
        gen_wav = gen_wav[:n_use]
        gt_wav = gt_wav[:n_use]
    else:
        gt_wav = None
        n_use = n_gen

    gen_out = {}
    gt_out = {} if gt_wav is not None else None

    for idx, (s, e) in enumerate(_chunk_bounds(n_use, gen_sr, chunk_sec, hop_sec, min_last_sec)):
        t0 = s / gen_sr
        t1 = e / gen_sr
        new_key = f"{key}@{t0:.3f}-{t1:.3f}"
        stem = f"{key}_chunk{idx:04d}_{t0:.3f}-{t1:.3f}"

        gen_path_out = tmp_root / "pred" / f"{stem}.wav"
        _write_wav(gen_path_out, gen_wav[s:e], gen_sr)
        gen_out[new_key] = str(gen_path_out)

        if gt_wav is not None:
            gt_path_out = tmp_root / "gt" / f"{stem}.wav"
            _write_wav(gt_path_out, gt_wav[s:e], gen_sr)
            gt_out[new_key] = str(gt_path_out)

    return gen_out, gt_out


def _maybe_chunk_filelists(
    args,
    gen_files: dict,
    gt_files: dict | None,
    text_info: dict | None,
) -> tuple[dict, dict | None, dict | None, Path | None]:
    """
    If chunking is enabled, create on-disk chunked wavs and return updated mappings.
    Also replicates text_info per chunk key.
    """
    if not args.enable_chunking:
        return gen_files, gt_files, text_info, None

    chunk_sec = float(args.chunk_duration)
    hop_sec = float(args.hop_duration) if args.hop_duration is not None else chunk_sec
    min_last_sec = float(args.min_last_chunk)

    # Choose temp root for chunks
    if args.chunk_tmp_dir:
        tmp_root = Path(args.chunk_tmp_dir)
    elif args.output_file:
        tmp_root = Path(str(args.output_file) + ".chunks")
    else:
        tmp_root = Path("./chunks")
    tmp_root.mkdir(parents=True, exist_ok=True)

    logging.info(
        f"Chunking enabled: chunk={chunk_sec}s, hop={hop_sec}s, min_last={min_last_sec}s, dir={tmp_root}"
    )

    gen_chunks_all: dict = {}
    gt_chunks_all: dict | None = {} if gt_files is not None else None
    text_chunks_all: dict | None = {} if text_info is not None else None

    for key, pred_path in gen_files.items():
        gt_path = gt_files.get(key) if gt_files is not None else None
        try:
            g_map, r_map = _chunk_pair_to_tmp(
                key,
                pred_path,
                gt_path,
                args.io,
                chunk_sec,
                hop_sec,
                min_last_sec,
                tmp_root,
            )
        except Exception as e:
            logging.warning(f"Chunking failed for key={key}: {e}")
            continue

        # Merge into global dicts
        gen_chunks_all.update(g_map)
        if gt_chunks_all is not None and r_map is not None:
            gt_chunks_all.update(r_map)
        elif gt_chunks_all is not None and r_map is None:
            # keep structure consistent
            gt_chunks_all = None

        # Duplicate text per chunk if provided
        if text_chunks_all is not None and text_info is not None and key in text_info:
            for ck in g_map.keys():
                text_chunks_all[ck] = text_info[key]

    return gen_chunks_all, gt_chunks_all, text_chunks_all, tmp_root


def main():
    args = get_parser().parse_args()

    # In case of using `local` backend, all GPU will be visible to all process.
    if args.use_gpu:
        gpu_rank = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(gpu_rank)
        logging.info(f"using device: cuda:{gpu_rank}")

    # logging info
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    gen_files = audio_loader_setup(args.pred, args.io)

    # find reference file
    args.gt = None if args.gt == "None" else args.gt
    if args.gt is not None and not args.no_match:
        gt_files = audio_loader_setup(args.gt, args.io)
    else:
        gt_files = None

    # find ground truth transcription
    if args.text is not None:
        text_info = {}
        with open(args.text) as f:
            for line in f.readlines():
                key, value = line.strip().split(maxsplit=1)
                text_info[key] = value
    else:
        text_info = None

    # Get and divide list
    if len(gen_files) == 0:
        raise FileNotFoundError("Not found any generated audio files.")
    if gt_files is not None and len(gen_files) > len(gt_files) and not args.enable_chunking:
        # (For chunking, we later truncate to min length per pair, so we don't pre-check count equality.)
        raise ValueError(
            "#groundtruth files are less than #generated files "
            f"(#gen={len(gen_files)} vs. #gt={len(gt_files)}). "
            "Please check the groundtruth directory."
        )

    logging.info("The number of utterances (pre-chunk) = %d", len(gen_files))

    # Optional: build chunked filelists and override maps
    gen_files, gt_files, text_info, chunk_tmp_dir = _maybe_chunk_filelists(
        args, gen_files, gt_files, text_info
    )

    if args.enable_chunking:
        logging.info("The number of items (post-chunk) = %d", len(gen_files))

    with open(args.score_config, "r", encoding="utf-8") as f:
        score_config = yaml.full_load(f)

    score_modules = load_score_modules(
        score_config,
        use_gt=(True if gt_files is not None else False),
        use_gt_text=(True if text_info is not None else False),
        use_gpu=args.use_gpu,
    )

    if len(score_modules) > 0:
        score_info = list_scoring(
            gen_files,
            score_modules,
            gt_files,
            text_info,
            output_file=args.output_file,
            io=args.io,
        )
        logging.info("Summary: %s", load_summary(score_info))
    else:
        logging.info("No utterance-level scoring function is provided.")

    corpus_score_modules = load_corpus_modules(
        score_config,
        use_gpu=args.use_gpu,
        cache_folder=args.cache_folder,
        io=args.io,
    )
    assert (
        len(corpus_score_modules) > 0 or len(score_modules) > 0
    ), "no scoring function is provided"

    # NOTE: For corpus scoring we keep original (non-chunked) paths unless you explicitly want
    # to aggregate over chunks. If you want corpus over chunks, pass args.pred as the CHUNK TMP dir
    # and ensure your corpus scorer supports directory inputs.
    if len(corpus_score_modules) > 0:
        pred_for_corpus = args.pred
        if args.enable_chunking and chunk_tmp_dir is not None:
            # Optionally switch corpus to chunk directory:
            pred_for_corpus = str(chunk_tmp_dir / "pred")
            logging.info(f"Corpus scoring over chunk directory: {pred_for_corpus}")

        corpus_score_info = corpus_scoring(
            pred_for_corpus,
            corpus_score_modules,
            args.gt if (args.gt is not None and not args.enable_chunking) else None,
            text_info if (text_info is not None and args.enable_chunking) else None,
            output_file=(args.output_file + ".corpus") if args.output_file else None,
        )
        logging.info("Corpus Summary: %s", corpus_score_info)
    else:
        logging.info("No corpus-level scoring function is provided.")


if __name__ == "__main__":
    main()

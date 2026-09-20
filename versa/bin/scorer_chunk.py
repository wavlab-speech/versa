#!/usr/bin/env python3

# Copyright 2025 BoHao Su
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Scorer Interface for Speech Evaluation with optional CHUNKED scoring."""

import argparse
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from versa.bin.cli_options import add_resume_arguments, enforce_run_status
from versa.completion import RunStatus
from versa.bin.scoring import (
    configure_runtime,
    load_inputs,
    load_score_config,
    run_scoring,
)
from versa.config_validation import validate_score_config
from versa.scorer_shared import VersaScorer
from versa.scorer_shared import load_audio, wav_normalize


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
        "--use_gpu", action="store_true", help="whether to use GPU if it can"
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
    add_resume_arguments(parser)

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


def _chunk_bounds(
    n_samples: int, sr: int, chunk_sec: float, hop_sec: float, min_last_sec: float
):
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

    for idx, (s, e) in enumerate(
        _chunk_bounds(n_use, gen_sr, chunk_sec, hop_sec, min_last_sec)
    ):
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
    Also replicates text_info per chunk key. Reject missing reference keys before
    writing any chunks.
    """
    if not args.enable_chunking:
        return gen_files, gt_files, text_info, None

    if gt_files is not None:
        missing_gt = sorted(set(gen_files) - set(gt_files))
        if missing_gt:
            raise ValueError(
                f"Ground truth is missing for generated keys: {missing_gt}"
            )

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
        gt_path = gt_files[key] if gt_files is not None else None
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

        # Duplicate text per chunk if provided
        if text_chunks_all is not None and text_info is not None and key in text_info:
            for ck in g_map.keys():
                text_chunks_all[ck] = text_info[key]

    return gen_chunks_all, gt_chunks_all, text_chunks_all, tmp_root


def main():
    """Validate CLI inputs, optionally materialize audio chunks, and run scoring.

    Chunk mode uses the generated prediction directory for corpus metrics;
    chunk files remain available after scoring."""
    parser = get_parser()
    args = parser.parse_args()

    configure_runtime(args)
    args.gt = None if args.gt == "None" else args.gt
    score_config = load_score_config(args)
    scorer = VersaScorer()
    try:
        validate_score_config(
            score_config,
            registry=scorer.registry,
            use_gt=(args.gt is not None and not args.no_match),
            use_gt_text=(args.text is not None),
            use_gpu=args.use_gpu,
        )
    except ValueError as e:
        parser.error(str(e))

    gen_files, gt_files, text_info = load_inputs(
        args, check_reference_count=not args.enable_chunking
    )

    logging.info("The number of utterances (pre-chunk) = %d", len(gen_files))

    # Optional: build chunked filelists and override maps
    gen_files, gt_files, text_info, chunk_tmp_dir = _maybe_chunk_filelists(
        args, gen_files, gt_files, text_info
    )

    if args.enable_chunking:
        logging.info("The number of items (post-chunk) = %d", len(gen_files))

    # Preserve this entrypoint's corpus path selection, including chunk directories.
    pred_for_corpus = args.pred
    gt_for_corpus = args.gt if args.gt is not None and not args.no_match else None
    if args.enable_chunking and chunk_tmp_dir is not None:
        pred_for_corpus = str(chunk_tmp_dir / "pred")
        logging.info(f"Corpus scoring over chunk directory: {pred_for_corpus}")
        gt_for_corpus = str(chunk_tmp_dir / "gt") if gt_files is not None else None

    run_status = RunStatus()
    has_metrics, _ = run_scoring(
        args,
        scorer,
        score_config,
        gen_files,
        gt_files,
        text_info,
        corpus_inputs=(pred_for_corpus, gt_for_corpus),
        parser=parser,
        corpus_defaults={"io": "dir" if args.enable_chunking else args.io},
        corpus_use_gt=gt_for_corpus is not None,
        run_status=run_status,
    )
    assert has_metrics, "no scoring function is provided"
    enforce_run_status(args, run_status)


if __name__ == "__main__":
    main()

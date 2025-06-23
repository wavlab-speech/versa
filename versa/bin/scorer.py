#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Scorer Interface for Speech Evaluation."""

import argparse
import logging

import torch
import yaml

from versa.scorer_shared import (
    audio_loader_setup,
    VersaScorer,
    compute_summary,
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
    return parser


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

    # fine ground truth transcription
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
    if gt_files is not None and len(gen_files) > len(gt_files):
        raise ValueError(
            "#groundtruth files are less than #generated files "
            f"(#gen={len(gen_files)} vs. #gt={len(gt_files)}). "
            "Please check the groundtruth directory."
        )

    logging.info("The number of utterances = %d" % len(gen_files))

    with open(args.score_config, "r", encoding="utf-8") as f:
        score_config = yaml.full_load(f)

    # Initialize VersaScorer
    scorer = VersaScorer()

    # Load utterance-level metrics
    utterance_metrics = scorer.load_metrics(
        score_config,
        use_gt=(True if gt_files is not None else False),
        use_gt_text=(True if text_info is not None else False),
        use_gpu=args.use_gpu,
    )

    # Perform utterance-level scoring
    if len(utterance_metrics.metrics) > 0:
        score_info = scorer.score_utterances(
            gen_files,
            utterance_metrics,
            gt_files,
            text_info,
            output_file=args.output_file,
            io=args.io,
        )
        logging.info("Summary: {}".format(compute_summary(score_info)))
    else:
        logging.info("No utterance-level scoring function is provided.")

    # Load corpus-level metrics (distributional metrics)
    corpus_metrics = scorer.load_metrics(
        score_config,
        use_gt=(True if gt_files is not None else False),
        use_gt_text=(True if text_info is not None else False),
        use_gpu=args.use_gpu,
    )

    # Filter for corpus-level metrics and perform corpus scoring  
    from versa.definition import MetricCategory
    corpus_suite = corpus_metrics.filter_by_category(MetricCategory.DISTRIBUTIONAL)
    if len(corpus_suite.metrics) > 0:
        corpus_score_info = scorer.score_corpus(
            gen_files,
            corpus_suite,
            gt_files,
            text_info,
            output_file=args.output_file + ".corpus" if args.output_file else None,
        )
        logging.info("Corpus Summary: {}".format(corpus_score_info))
    else:
        logging.info("No corpus-level scoring function is provided.")

    # Ensure at least one scoring function is provided
    if len(utterance_metrics.metrics) == 0 and len(corpus_suite.metrics) == 0:
        raise ValueError("No scoring function is provided")


if __name__ == "__main__":
    main()

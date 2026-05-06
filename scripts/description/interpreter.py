#!/usr/bin/env python3

# Copyright 2025 BoHao Su
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Interpreter Interface for Speech Evaluation."""

import argparse
import json
import logging

import torch
import yaml
from text_llm_description import describe_all
from interpreter_shared import load_interpreter_modules, metric_loader_setup


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Interpretation for Speech Evaluation Interface"
    )
    parser.add_argument(
        "--score_output_file",
        type=str,
        default=None,
        help="Path of directory of the score results.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML with interpreter_config (list of model_name dicts)",
    )
    parser.add_argument(
        "--output_file", required=True, help="Where to dump the JSON descriptions"
    )
    parser.add_argument(
        "--use_gpu", type=bool, default=False, help="whether to use GPU if it can"
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

    metrics = metric_loader_setup(args.score_output_file)
    logging.info("The number of utterances = %d" % len(metrics))

    # 2) Load interpreter modules from YAML
    with open(args.config) as cf:
        cfg = yaml.safe_load(cf)
    interpreter_modules = load_interpreter_modules(
        cfg["interpreter_config"],
        use_gpu=args.use_gpu,
    )

    # 3) Run description for each model
    all_results = []
    for model_cfg in cfg["interpreter_config"]:
        name = model_cfg["model_name"]
        logging.info(f"Describing with {name}")
        res = describe_all(metrics, name, interpreter_modules)
        all_results.extend(res)

    # 4) Dump
    with open(args.output_file, "w") as outf:
        json.dump(all_results, outf, ensure_ascii=False, indent=2)
    logging.info(f"Wrote descriptions to {args.output_file}")


if __name__ == "__main__":
    main()

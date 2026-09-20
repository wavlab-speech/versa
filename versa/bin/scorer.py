#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Scorer Interface for Speech Evaluation."""

import argparse
import logging

from versa.bin.cli_options import add_resume_arguments, enforce_run_status
from versa.completion import RunStatus
from versa.metric_discovery import (
    create_metric_discovery_registry,
    describe_metric,
    format_metric_list,
    parse_metric_category,
    parse_metric_type,
    recommend_config,
    supported_recommendation_tasks,
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
        "--pred_sources",
        "--pred-sources",
        nargs="+",
        default=None,
        metavar="SCP",
        help=(
            "Ordered source-specific wav.scp files for multi-source metrics. "
            "Use with --gt_sources; source position defines the assignment."
        ),
    )
    parser.add_argument(
        "--gt_sources",
        "--gt-sources",
        nargs="+",
        default=None,
        metavar="SCP",
        help="Ordered reference source wav.scp files for multi-source metrics.",
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
        "--num_workers",
        type=int,
        default=1,
        help="Number of local CPU worker processes for utterance scoring.",
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
    parser.add_argument(
        "--scoring_mode",
        type=str,
        default="utterance",
        choices=["utterance", "metric"],
        help=(
            "Scoring loop order. Use 'metric' to load one metric at a time, "
            "reducing peak GPU memory when many metrics are configured."
        ),
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help=(
            "Optional report path generated from utterance-level scores after "
            "scoring. Format is inferred from extension unless --report-format "
            "is set."
        ),
    )
    parser.add_argument(
        "--report-format",
        choices=["auto", "html", "csv", "md"],
        default="auto",
        help="Report format for --report.",
    )
    parser.add_argument(
        "--report-group-by",
        default=None,
        help="Optional scored record field used for per-metric report rankings.",
    )
    parser.add_argument(
        "--report-outlier-limit",
        type=int,
        default=3,
        help="Maximum outlier examples to keep per metric in --report.",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List registered metrics and exit.",
    )
    parser.add_argument(
        "--describe-metric",
        type=str,
        default=None,
        metavar="NAME",
        help="Describe one metric by name or alias and exit.",
    )
    parser.add_argument(
        "--recommend-config",
        action="store_true",
        help="Print a recommended YAML score config and exit.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help=(
            "Task for --recommend-config. Supported tasks: "
            + ", ".join(supported_recommendation_tasks())
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Target device for --recommend-config.",
    )
    parser.add_argument(
        "--metric-category",
        type=str,
        default=None,
        choices=["independent", "dependent", "non_match", "distributional"],
        help="Filter --list-metrics by metric category.",
    )
    parser.add_argument(
        "--metric-type",
        type=str,
        default=None,
        choices=[
            "string",
            "float",
            "int",
            "bool",
            "list",
            "dict",
            "tuple",
            "array",
            "time",
        ],
        help="Filter --list-metrics by output type.",
    )
    return parser


def _text_required_multi_source_metrics(score_config, registry):
    """Return configured multi-source metrics needing unsupported text input."""
    if not isinstance(score_config, list):
        return []

    unsupported = []
    for config in score_config:
        if not isinstance(config, dict):
            continue
        metric_name = config.get("name")
        metadata = registry.get_metadata(metric_name)
        if metadata and metadata.requires_multiple_sources and metadata.requires_text:
            unsupported.append(metric_name)
    return unsupported


def main():
    """Parse discovery or scoring options, validate inputs, and run the selected mode.

    Scoring may initialize models and write or append results. Ordered
    multi-source mode requires paired SCP lists and one utterance worker.
    When requested, write a report from the resulting utterance records."""
    parser = get_parser()
    args = parser.parse_args()

    if args.num_workers < 1:
        parser.error("--num_workers must be at least 1")
    if args.num_workers > 1 and args.use_gpu:
        parser.error("--num_workers > 1 is CPU-only and cannot be used with --use_gpu")

    if args.list_metrics or args.describe_metric or args.recommend_config:
        try:
            if args.list_metrics:
                registry = create_metric_discovery_registry()
                category = parse_metric_category(args.metric_category)
                metric_type = parse_metric_type(args.metric_type)
                print(format_metric_list(registry, category, metric_type))
                return
            if args.describe_metric:
                registry = create_metric_discovery_registry()
                print(describe_metric(registry, args.describe_metric))
                return
            if args.recommend_config:
                if not args.task:
                    parser.error("--recommend-config requires --task")
                print(recommend_config(args.task, args.device))
                return
        except ValueError as e:
            parser.error(str(e))

    from versa.scorer_shared import audio_loader_setup, VersaScorer, compute_summary
    from versa.bin.scoring import (
        configure_runtime,
        load_inputs,
        load_score_config,
        run_scoring,
    )

    configure_runtime(args)
    score_config = load_score_config(args)

    multi_source_mode = args.pred_sources is not None or args.gt_sources is not None
    if multi_source_mode:
        if args.pred_sources is None or args.gt_sources is None:
            parser.error("--pred_sources and --gt_sources must be provided together")
        if args.pred is not None or args.gt is not None:
            parser.error(
                "--pred_sources/--gt_sources cannot be combined with --pred/--gt"
            )
        if args.no_match:
            parser.error("--no_match is not valid for multi-source scoring")
        if args.num_workers != 1:
            parser.error("multi-source scoring currently requires --num_workers 1")
        if args.scoring_mode != "utterance":
            parser.error("multi-source scoring currently uses --scoring_mode utterance")
    elif args.pred is None:
        parser.error("--pred is required unless --pred_sources is used")

    # Validate before any scoring or model setup begins.
    scorer = VersaScorer()
    if multi_source_mode:
        text_required_metrics = _text_required_multi_source_metrics(
            score_config, scorer.registry
        )
        if text_required_metrics:
            parser.error(
                "multi-source scoring does not yet support metrics requiring "
                "reference text: " + ", ".join(text_required_metrics)
            )

    try:
        from versa.config_validation import validate_score_config

        validate_score_config(
            score_config,
            registry=scorer.registry,
            use_gt=(
                args.gt_sources is not None
                if multi_source_mode
                else args.gt is not None and args.gt != "None" and not args.no_match
            ),
            use_gt_text=(args.text is not None),
            use_gpu=args.use_gpu,
        )
    except ValueError as e:
        parser.error(str(e))

    score_metadata = {
        config["name"]: scorer.registry.get_metadata(config["name"])
        for config in score_config
    }
    multi_source_score_config = [
        config
        for config in score_config
        if score_metadata[config["name"]]
        and score_metadata[config["name"]].requires_multiple_sources
    ]

    if multi_source_mode:
        if len(multi_source_score_config) != len(score_config):
            parser.error(
                "--pred_sources/--gt_sources can only be used with metrics that "
                "declare multi-source input"
            )
        if len(args.pred_sources) < 2 or len(args.gt_sources) < 2:
            parser.error("multi-source scoring requires at least two source SCPs")
        if len(args.pred_sources) != len(args.gt_sources):
            parser.error(
                "--pred_sources and --gt_sources must contain the same number of SCPs"
            )

        gen_source_files = [
            audio_loader_setup(path, args.io) for path in args.pred_sources
        ]
        gt_source_files = [
            audio_loader_setup(path, args.io) for path in args.gt_sources
        ]
        run_status = RunStatus()
        multi_source_metrics = scorer.load_metrics(
            multi_source_score_config,
            use_gt=True,
            use_gt_text=False,
            use_gpu=args.use_gpu,
            run_status=run_status,
        )
        if not multi_source_metrics.metrics:
            raise ValueError("No multi-source scoring function is available")
        score_info = scorer.score_multi_source_utterances(
            gen_source_files,
            multi_source_metrics,
            gt_source_files,
            output_file=args.output_file,
            io=args.io,
            resume=args.resume,
            legacy_resume=args.legacy_resume,
            input_identity=args.input_identity,
            run_status=run_status,
        )
        logging.info("Summary: %s", compute_summary(score_info))
        if args.report:
            if not score_info:
                raise ValueError("--report requires at least one utterance-level score")
            _write_report(
                score_info,
                args.report,
                report_format=args.report_format,
                group_by=args.report_group_by,
                outlier_limit=args.report_outlier_limit,
                registry=scorer.registry,
            )
        enforce_run_status(args, run_status)
        return

    if multi_source_score_config:
        parser.error(
            "multi-source metrics require ordered --pred_sources and --gt_sources"
        )

    gen_files, gt_files, text_info = load_inputs(args)
    logging.info("The number of utterances = %d", len(gen_files))
    run_status = RunStatus()
    has_metrics, score_info = run_scoring(
        args,
        scorer,
        score_config,
        gen_files,
        gt_files,
        text_info,
        parser=parser,
        run_status=run_status,
    )
    if not has_metrics:
        raise ValueError("No scoring function is provided")

    if args.report:
        if not score_info:
            raise ValueError("--report requires at least one utterance-level score")
        _write_report(
            score_info,
            args.report,
            report_format=args.report_format,
            group_by=args.report_group_by,
            outlier_limit=args.report_outlier_limit,
            registry=scorer.registry,
        )
    # Report run completeness last so a strict failure still leaves every
    # requested artifact behind.
    enforce_run_status(args, run_status)


def _write_report(
    score_info,
    report_path,
    *,
    report_format,
    group_by,
    outlier_limit,
    registry,
):
    """Create the parent directory and overwrite a report from utterance records.

    Infer HTML, CSV, or Markdown from the extension in auto mode; unknown
    extensions default to HTML."""
    from pathlib import Path

    from versa.reporting import (
        analyze_records,
        write_csv_report,
        write_html_report,
        write_markdown_report,
    )

    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if report_format == "auto":
        report_format = {
            ".html": "html",
            ".htm": "html",
            ".csv": "csv",
            ".md": "md",
            ".markdown": "md",
        }.get(output_path.suffix.lower(), "html")

    analysis = analyze_records(
        score_info,
        group_by=group_by,
        outlier_limit=outlier_limit,
        registry=registry,
    )
    if report_format == "html":
        write_html_report(analysis, report_path)
    elif report_format == "csv":
        write_csv_report(analysis, report_path)
    else:
        write_markdown_report(analysis, report_path)

    logging.info("Wrote %s report to %s", report_format, report_path)


if __name__ == "__main__":
    main()

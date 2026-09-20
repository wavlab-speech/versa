"""Shared CLI setup and scoring after entrypoint-specific input preparation."""

import logging

import torch
import yaml

from versa.completion import INPUT_IDENTITY_PATH, LEGACY_RECOMPUTE
from versa.definition import MetricCategory
from versa.scorer_shared import (
    audio_loader_setup,
    compute_summary,
    configure_metric_cache_dirs,
    configure_shared_cache_environment,
)


def configure_runtime(args):
    # In case of using `local` backend, all GPU will be visible to all process.
    """Configure logging and select a CUDA device by rank when GPU use is requested.

    Raise RuntimeError when GPU execution is requested without a CUDA device."""
    if args.use_gpu:
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError("--use_gpu was set, but no CUDA device is available")
        gpu_rank = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(gpu_rank)
        logging.info(f"using device: cuda:{gpu_rank}")

    level = (
        logging.DEBUG
        if args.verbose > 1
        else logging.INFO if args.verbose > 0 else logging.WARN
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    if args.verbose <= 0:
        logging.warning("Skip DEBUG/INFO messages")


def load_inputs(args, *, check_reference_count=True):
    """Load prediction, reference, and transcript mappings from CLI arguments.

    Normalize the literal ground-truth value ``None`` on args and omit paired
    references in no-match mode. Reject empty predictions and, when enabled,
    a reference collection smaller than the prediction collection."""
    gen_files = audio_loader_setup(args.pred, args.io)

    # find reference file
    args.gt = None if args.gt == "None" else args.gt
    if args.gt is not None and not args.no_match:
        gt_files = audio_loader_setup(args.gt, args.io)
    else:
        gt_files = None

    # Load ground truth transcription
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
    if (
        gt_files is not None
        and len(gen_files) > len(gt_files)
        and check_reference_count
    ):
        raise ValueError(
            "#groundtruth files are less than #generated files "
            f"(#gen={len(gen_files)} vs. #gt={len(gt_files)}). "
            "Please check the groundtruth directory."
        )

    return gen_files, gt_files, text_info


def load_score_config(args):
    """Read YAML and apply the shared cache policy before validation."""
    with open(args.score_config, "r", encoding="utf-8") as f:
        score_config = yaml.safe_load(f)
    configure_shared_cache_environment(args.cache_folder)
    return configure_metric_cache_dirs(score_config, args.cache_folder)


def run_scoring(
    args,
    scorer,
    score_config,
    gen_files,
    gt_files,
    text_info,
    *,
    parser,
    corpus_inputs=None,
    corpus_defaults=None,
    corpus_use_gt=None,
    run_status=None,
):
    """Score prepared inputs; return availability and utterance records.

    The chunk CLI supplies its historical corpus paths and config defaults.
    Metric/resource and result-file lifetimes remain owned by VersaScorer.
    Configuration validation happens in the entrypoints before input loading.
    Run completeness is collected in ``run_status`` when one is supplied, and
    the caller decides whether an incomplete run is fatal.
    """
    corpus_score_config = []
    utterance_score_config = []
    for config in score_config:
        metadata = scorer.registry.get_metadata(config["name"])
        if metadata and metadata.category == MetricCategory.DISTRIBUTIONAL:
            if corpus_defaults is not None:
                config = {**corpus_defaults, **config}
            corpus_score_config.append(config)
        else:
            utterance_score_config.append(config)

    scoring_mode = getattr(args, "scoring_mode", "utterance")
    num_workers = getattr(args, "num_workers", 1)
    if num_workers > 1 and scoring_mode == "metric":
        parser.error(
            "--num_workers > 1 is only supported with --scoring_mode utterance"
        )
    if num_workers > 1 and not utterance_score_config:
        parser.error("--num_workers > 1 requires at least one utterance-level metric")

    score_info = []
    if scoring_mode == "metric":
        score_info = scorer.score_utterances_by_metric(
            gen_files,
            utterance_score_config,
            gt_files,
            text_info,
            output_file=args.output_file,
            io=args.io,
            resume=args.resume,
            use_gpu=args.use_gpu,
            legacy_resume=getattr(args, "legacy_resume", LEGACY_RECOMPUTE),
            input_identity=getattr(args, "input_identity", INPUT_IDENTITY_PATH),
            run_status=run_status,
        )
        logging.info("Summary: {}".format(compute_summary(score_info)))
        utterance_metric_count = int(
            any(any(key != "key" for key in score) for score in score_info)
        )
    else:
        # Load utterance-level metrics
        utterance_metrics = scorer.load_metrics(
            utterance_score_config,
            use_gt=gt_files is not None,
            use_gt_text=text_info is not None,
            use_gpu=args.use_gpu,
            run_status=run_status,
        )

        utterance_metric_count = len(
            [
                metric
                for metric in utterance_metrics.metrics.values()
                if metric.get_metadata().category != MetricCategory.DISTRIBUTIONAL
            ]
        )

    # Perform utterance-level scoring
    if scoring_mode == "utterance" and len(utterance_metrics.metrics) > 0:
        score_info = scorer.score_utterances(
            gen_files,
            utterance_metrics,
            gt_files,
            text_info,
            output_file=args.output_file,
            io=args.io,
            resume=args.resume,
            num_workers=num_workers,
            legacy_resume=getattr(args, "legacy_resume", LEGACY_RECOMPUTE),
            input_identity=getattr(args, "input_identity", INPUT_IDENTITY_PATH),
            run_status=run_status,
        )
        logging.info("Summary: {}".format(compute_summary(score_info)))
    elif utterance_metric_count == 0:
        logging.info("No utterance-level scoring function is provided.")

    # Load corpus-level metrics (distributional metrics)
    corpus_metrics = scorer.load_metrics(
        corpus_score_config,
        use_gt=(gt_files is not None if corpus_use_gt is None else corpus_use_gt),
        use_gt_text=text_info is not None,
        use_gpu=args.use_gpu,
        run_status=run_status,
    )

    # Filter for corpus-level metrics and perform corpus scoring
    corpus_suite = corpus_metrics.filter_by_category(MetricCategory.DISTRIBUTIONAL)
    if len(corpus_suite.metrics) > 0:
        corpus_pred, corpus_gt = (
            (gen_files, gt_files) if corpus_inputs is None else corpus_inputs
        )
        corpus_score_info = scorer.score_corpus(
            corpus_pred,
            corpus_suite,
            corpus_gt,
            text_info,
            output_file=args.output_file + ".corpus" if args.output_file else None,
            run_status=run_status,
        )
        logging.info("Corpus Summary: {}".format(corpus_score_info))
    else:
        logging.info("No corpus-level scoring function is provided.")

    return (utterance_metric_count > 0 or len(corpus_suite.metrics) > 0), score_info

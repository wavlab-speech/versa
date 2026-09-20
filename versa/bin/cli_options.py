"""Command-line options shared by the scoring entrypoints.

This module stays free of model and framework imports so building a parser,
listing metrics, or printing help does not load a backend.
"""

import json
import logging

from versa.completion import (
    INPUT_IDENTITY_CONTENT,
    INPUT_IDENTITY_PATH,
    LEGACY_RECOMPUTE,
    LEGACY_TRUST,
)


def add_resume_arguments(parser):
    """Add the shared resume, input identity, and strict-run options.

    Both scoring entrypoints expose identical semantics, so the contract is
    defined once here."""
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume utterance scoring from an existing output_file. An "
            "utterance is skipped only when every configured metric completed "
            "successfully under the same metric, configuration, and input "
            "identity; missing and failed metrics are recomputed and merged."
        ),
    )
    parser.add_argument(
        "--legacy_resume",
        type=str,
        default=LEGACY_RECOMPUTE,
        choices=[LEGACY_RECOMPUTE, LEGACY_TRUST],
        help=(
            "How to treat result rows written before completion records "
            "existed. 'recompute' rescores them; 'trust' keeps any row that "
            "already has a value, reproducing the historical key-only rule."
        ),
    )
    parser.add_argument(
        "--input_identity",
        type=str,
        default=INPUT_IDENTITY_PATH,
        choices=[INPUT_IDENTITY_PATH, INPUT_IDENTITY_CONTENT],
        help=(
            "How resume detects changed inputs. 'path' compares input "
            "locations only; 'content' also hashes each input file, which "
            "detects edited audio at the cost of reading every input."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit unsuccessfully when a metric fails to load, fails to run, or "
            "an utterance is skipped. Completed results are still written."
        ),
    )
    return parser


def enforce_run_status(args, run_status):
    """Log run completeness and fail a strict run that did not complete.

    Tolerant runs, the default, only report the counts. Results written before
    the failure are retained either way."""
    logging.info("Run status: %s", run_status.describe())
    if getattr(args, "strict", False) and run_status.has_failures:
        raise SystemExit(
            "Strict run did not complete: " + json.dumps(run_status.as_dict())
        )

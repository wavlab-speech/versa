#!/usr/bin/env python3

"""Helpers for keeping Hugging Face model cache locations explicit."""

import os
from contextlib import contextmanager
from pathlib import Path

DEFAULT_HF_CACHE_DIR = "versa_cache/huggingface"
VERSA_HF_CACHE_ENV = "VERSA_HF_CACHE_DIR"


def get_hf_cache_dir(config_cache_dir=None):
    """Return the visible Hugging Face cache directory used by Versa metrics."""
    return (
        os.environ.get(VERSA_HF_CACHE_ENV) or config_cache_dir or DEFAULT_HF_CACHE_DIR
    )


def configure_huggingface_cache(cache_dir):
    """Configure Hugging Face tooling to use a repo-visible cache directory."""
    cache_path = Path(cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_path.parent))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_path))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_path))
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    return cache_path


def model_snapshot_has_file(cache_dir, repo_id, filename):
    """Return True when a cached model snapshot contains filename."""
    model_dir = Path(cache_dir) / ("models--" + repo_id.replace("/", "--"))
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return False
    return any((snapshot / filename).is_file() for snapshot in snapshots_dir.iterdir())


def local_files_only_kwargs(cache_dir, required_files):
    """Return local_files_only=True when the requested files are cached."""
    if all(
        model_snapshot_has_file(cache_dir, repo_id, filename)
        for repo_id, filename in required_files
    ):
        return {"local_files_only": True}
    return {}


@contextmanager
def offline_if_cached(cache_dir, required_files):
    """Temporarily force offline loading when all required cache files exist."""
    all_cached = all(
        model_snapshot_has_file(cache_dir, repo_id, filename)
        for repo_id, filename in required_files
    )
    old_value = os.environ.get("TRANSFORMERS_OFFLINE")
    if all_cached:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        if all_cached:
            if old_value is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = old_value

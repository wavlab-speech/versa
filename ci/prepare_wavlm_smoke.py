"""Download an explicitly pinned WavLM snapshot and record smoke-test provenance."""

import argparse
import json
import re
from importlib.metadata import version
from pathlib import Path


def prepare(revision, cache_dir, manifest):
    """Require a full commit SHA, cache its model assets, and record exact versions."""
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise ValueError("WavLM revision must be a full 40-character commit SHA")
    from huggingface_hub import snapshot_download

    revision = revision.lower()
    model_id = "microsoft/wavlm-base-sv"
    snapshot = snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=["*.json", "*.safetensors", "*.bin"],
    )
    result = {
        "model_id": model_id,
        "revision": revision,
        "snapshot_path": str(Path(snapshot).resolve()),
        "packages": {
            name: version(name)
            for name in (
                "versa-speech-audio-toolkit",
                "torch",
                "torchaudio",
                "transformers",
                "huggingface-hub",
            )
        },
    }
    Path(manifest).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    prepare(args.revision, args.cache_dir, args.manifest)

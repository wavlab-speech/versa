"""Collect auditable website statistics without importing model backends."""

import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parent.parent
API = "https://api.github.com/repos/wavlab-speech/versa"
CATEGORY_LABELS = {
    "independent": "Independent",
    "dependent": "Dependent",
    "non_match": "Non-match",
    "distributional": "Distributional",
}


def collect_statistics():
    """Fetch GitHub counts and discover canonical metric names from this checkout.

    Fail the build on unavailable or malformed data so the last successful
    deployment and its original timestamp remain visible.
    """
    sys.path.insert(0, str(REPO))
    from versa.metric_discovery import create_metric_discovery_registry

    registry = create_metric_discovery_registry(include_runtime_imports=False)
    names = registry.list_metrics()
    if not names:
        raise ValueError("Metric discovery returned no metrics")
    metrics = [
        {"name": name, "category": registry.get_metadata(name).category.value}
        for name in names
    ]
    categories = dict(sorted(Counter(item["category"] for item in metrics).items()))
    if not set(categories) <= set(CATEGORY_LABELS):
        raise ValueError("Update website category labels for new metric categories")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "VERSA-website-statistics",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(API, headers=headers), timeout=30) as response:
        repo = json.load(response)
    if repo.get("full_name") != "wavlab-speech/versa":
        raise ValueError("Unexpected GitHub repository in statistics response")
    for key in ("stargazers_count", "forks_count"):
        if type(repo.get(key)) is not int or repo[key] < 0:
            raise ValueError(f"Invalid GitHub count: {key}")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    return {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository": "wavlab-speech/versa",
        "source_revision": revision,
        "metric_count": len(metrics),
        "categories": categories,
        "stars": repo["stargazers_count"],
        "forks": repo["forks_count"],
        "metrics": metrics,
        "github_source": API,
    }


def statistics_markdown(stats):
    """Explain the current snapshot and list the exact metrics counted."""
    lines = [
        f"Last updated: **{stats['updated_at'].replace('T', ' ').replace('Z', ' UTC')}**.",
        "",
        "## Current collection",
        "",
        "| Statistic | Count |",
        "| --- | ---: |",
        f"| Discoverable metrics | {stats['metric_count']} |",
        f"| GitHub stars | {stats['stars']} |",
        f"| GitHub forks | {stats['forks']} |",
        "",
        "## Metrics by category",
        "",
        "| Category | Metrics |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {CATEGORY_LABELS[key]} | {count} |"
        for key, count in stats["categories"].items()
    )
    lines += [
        "",
        "## How we count",
        "",
        "The metric total is the number of unique canonical names returned by "
        "VERSA’s source-based discovery registry for the published checkout. "
        "Aliases are excluded. Named Qwen prompt metrics are included separately; "
        "a metric returning multiple output scores is counted once. "
        "This is a count of discoverable metrics, not installed backends or model variants. "
        "Discovery reads source metadata without loading models; optional dependencies "
        "are still needed to run many metrics. It recognizes the metadata patterns "
        "supported by VERSA’s discovery code.",
        "",
        "Stars and forks come from the official repository’s GitHub API "
        "at build time. These are dated snapshots, not real-time counters. "
        "The website refreshes daily and when relevant code or documentation changes. "
        "Scheduled runs can be delayed by GitHub. If collection fails, the previous "
        "successful website stays online with its original timestamp.",
        "",
        "## Sources",
        "",
        f"- [GitHub repository statistics]({API})",
        f"- [Metric discovery source](https://github.com/wavlab-speech/versa/blob/{stats['source_revision']}/versa/metric_discovery.py)",
        f"- Source revision: `{stats['source_revision']}`",
        "- [Download the complete statistics snapshot](../../stats.json)",
        "- [Metric catalog and references](../metrics/)",
        "",
        "## Metrics included in this count",
        "",
        "| Canonical name | Category |",
        "| --- | --- |",
    ]
    lines.extend(
        f"| `{item['name']}` | {CATEGORY_LABELS[item['category']]} |"
        for item in stats["metrics"]
    )
    return "\n".join(lines)

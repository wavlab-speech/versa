import json
import argparse
from collections import defaultdict
import statistics
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional


def safe_stdev(values):
    """Safe standard deviation calculation"""
    try:
        if len(values) <= 1:
            return 0.0

        # Filter out inf and nan values
        clean_values = [
            v for v in values if not (v == float("inf") or v == float("-inf") or v != v)
        ]

        if len(clean_values) <= 1:
            return 0.0

        # Check if all values are the same (or very close)
        if max(clean_values) - min(clean_values) < 1e-10:
            return 0.0

        # Manual calculation to avoid statistics module issues
        mean_val = sum(clean_values) / len(clean_values)
        variance = sum((x - mean_val) ** 2 for x in clean_values) / (
            len(clean_values) - 1
        )

        if variance < 0:
            return 0.0

        import math

        return math.sqrt(variance)

    except (statistics.StatisticsError, ValueError, ZeroDivisionError, OverflowError):
        return 0.0


def read_input_files(input_path):
    """Read JSONL files from either a single file or directory"""
    files_to_process = []

    if os.path.isfile(input_path):
        files_to_process = [input_path]
    elif os.path.isdir(input_path):
        # Look for common JSONL file patterns
        patterns = ["*.jsonl", "*.json", "*.jl", "*.txt"]
        for pattern in patterns:
            files_to_process.extend(glob.glob(os.path.join(input_path, pattern)))

        if not files_to_process:
            print(f"No JSONL files found in directory: {input_path}")
            return []
    else:
        print(f"Input path does not exist: {input_path}")
        return []

    print(
        f"Processing {len(files_to_process)} file(s): {[os.path.basename(f) for f in files_to_process]}"
    )

    # Read all data
    all_data = []
    for file_path in files_to_process:
        print(f"Reading: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data = json.loads(line)
                            # Add source file information
                            data["_source_file"] = os.path.basename(file_path)
                            all_data.append(data)
                        except json.JSONDecodeError as e:
                            print(
                                f"Warning: Invalid JSON on line {line_num} in {file_path}: {e}"
                            )
                        except Exception as e:
                            print(
                                f"Warning: Error parsing line {line_num} in {file_path}: {e}"
                            )
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    return all_data


def discover_metrics(info: List[Dict]) -> Dict[str, Any]:
    """Automatically discover all metrics from the data and categorize them"""
    discovery = {
        "all_metrics": set(),
        "metric_categories": defaultdict(set),
        "ignored_fields": set(),
    }

    # Define known metric categories based on your specification
    metric_category_mapping = {
        # Audio Quality Metrics
        "audio_quality": [
            "dnsmos_overall",
            "dnsmos_p808",
            "nisqa",
            "utmos",
            "plcmos",
            "singmos",
            "sheet_ssqa",
            "utmosv2",
            "scoreq_nr",
            "scoreq_ref",
            "noresqa",
            "torch_squim_mos",
            "warpq",
            "dnsmos_pro_bvcc",
            "dnsmos_pro_nisqa",
            "dnsmos_pro_vcc2018",
        ],
        # Speech Enhancement Metrics
        "speech_enhancement": [
            "torch_squim_pesq",
            "torch_squim_stoi",
            "torch_squim_si_sdr",
            "se_si_snr",
            "se_ci_sdr",
            "se_sar",
            "se_sdr",
            "pesq",
            "stoi",
            "sir",
            "sar",
            "sdr",
            "ci-sdr",
            "si-snr",
            "visqol",
        ],
        # Psychoacoustic/Perceptual Metrics
        "psychoacoustic": [
            "pysepm_fwsegsnr",
            "pysepm_wss",
            "pysepm_cd",
            "pysepm_c_sig",
            "pysepm_c_bak",
            "pysepm_c_ovl",
            "pysepm_csii_high",
            "pysepm_csii_mid",
            "pysepm_csii_low",
            "pysepm_ncm",
            "pysepm_llr",
            "pam",
            "srmr",
        ],
        # ASR/Text Metrics - WER/CER components
        "asr_wer_cer": [
            "espnet_wer",
            "espnet_wer_delete",
            "espnet_wer_insert",
            "espnet_wer_replace",
            "espnet_wer_equal",
            "espnet_cer",
            "espnet_cer_delete",
            "espnet_cer_insert",
            "espnet_cer_replace",
            "espnet_cer_equal",
            "owsm_wer",
            "owsm_wer_delete",
            "owsm_wer_insert",
            "owsm_wer_replace",
            "owsm_wer_equal",
            "owsm_cer",
            "owsm_cer_delete",
            "owsm_cer_insert",
            "owsm_cer_replace",
            "owsm_cer_equal",
            "whisper_wer",
            "whisper_cer",
            "whisper_cer_delete",
            "asr_match_error_rate",
        ],
        # Semantic/Content Metrics
        "semantic": [
            "speech_bert",
            "speech_belu",
            "speech_token_distance",
            "clap_score",
        ],
        # Similarity Metrics
        "similarity": ["emotion_similarity", "spk_similarity", "singer_similarity"],
        # Pitch/F0 Metrics
        "pitch_f0": ["f0_corr", "f0_rmse", "mcd"],
        # Other Audio Features
        "audio_features": ["speaking_rate", "log_wmse"],
        # Audio Aesthetics
        "aesthetics": [
            "audiobox_aesthetics_CE",
            "audiobox_aesthetics_CU",
            "audiobox_aesthetics_PC",
            "audiobox_aesthetics_PQ",
        ],
        # Security/Spoofing
        "security": ["asvspoof_score", "nomad"],
        # Other/Misc
        "other_metrics": ["apa"],
    }

    # Create reverse mapping for quick lookup
    metric_to_category = {}
    for category, metrics in metric_category_mapping.items():
        for metric in metrics:
            metric_to_category[metric] = category

    # Collect all numeric metrics and categorize them
    for item in info:
        for key, value in item.items():
            # Skip non-numeric fields and internal fields
            if key.startswith("_") or key in ["key"] or "text" in key.lower():
                discovery["ignored_fields"].add(key)
                continue

            if isinstance(value, (int, float)):
                discovery["all_metrics"].add(key)

                # First try exact match with known metrics
                if key in metric_to_category:
                    category = metric_to_category[key]
                    discovery["metric_categories"][category].add(key)
                else:
                    # Fall back to pattern matching for custom metrics
                    key_lower = key.lower()
                    categorized = False

                    # Custom pattern matching for metrics not in the predefined list
                    if "similarity" in key_lower or "sim" in key_lower:
                        discovery["metric_categories"]["similarity"].add(key)
                        categorized = True
                    elif "chroma" in key_lower:
                        discovery["metric_categories"]["chroma"].add(key)
                        categorized = True
                    elif "dtw" in key_lower:
                        discovery["metric_categories"]["dtw"].add(key)
                        categorized = True
                    elif any(x in key_lower for x in ["mfcc", "spectral", "mel"]):
                        discovery["metric_categories"]["spectral"].add(key)
                        categorized = True
                    elif any(x in key_lower for x in ["pitch", "f0", "fundamental"]):
                        discovery["metric_categories"]["pitch_f0"].add(key)
                        categorized = True
                    elif any(x in key_lower for x in ["energy", "rms", "power"]):
                        discovery["metric_categories"]["energy"].add(key)
                        categorized = True
                    elif any(x in key_lower for x in ["tempo", "rhythm", "beat"]):
                        discovery["metric_categories"]["rhythm"].add(key)
                        categorized = True
                    elif any(
                        x in key_lower for x in ["distance", "euclidean", "cosine"]
                    ):
                        discovery["metric_categories"]["distance"].add(key)
                        categorized = True
                    elif any(x in key_lower for x in ["correlation", "corr"]):
                        discovery["metric_categories"]["correlation"].add(key)
                        categorized = True
                    elif any(x in key_lower for x in ["wer", "cer"]) and any(
                        x in key_lower for x in ["delete", "insert", "replace", "equal"]
                    ):
                        discovery["metric_categories"]["asr_wer_cer"].add(key)
                        categorized = True
                    elif any(
                        x in key_lower for x in ["pesq", "stoi", "sdr", "sir", "sar"]
                    ):
                        discovery["metric_categories"]["speech_enhancement"].add(key)
                        categorized = True
                    elif any(x in key_lower for x in ["mos", "quality"]):
                        discovery["metric_categories"]["audio_quality"].add(key)
                        categorized = True

                    if not categorized:
                        discovery["metric_categories"]["other"].add(key)

    return discovery


def estimate_metric_quality(metric_name: str, values: List[float]) -> Dict[str, Any]:
    """
    Estimate the quality and characteristics of a metric based on its values and name.
    Returns insights about the metric's behavior and interpretation.
    """
    if not values:
        return {
            "quality": "insufficient_data",
            "recommendation": "Not enough data to analyze",
        }

    clean_values = [
        v for v in values if not (v == float("inf") or v == float("-inf") or v != v)
    ]
    if not clean_values:
        return {
            "quality": "invalid_data",
            "recommendation": "All values are invalid (inf/nan)",
        }

    analysis = {
        "count": len(clean_values),
        "mean": statistics.mean(clean_values),
        "median": statistics.median(clean_values),
        "std": safe_stdev(values),
        "min": min(clean_values),
        "max": max(clean_values),
        "range": max(clean_values) - min(clean_values),
        "cv": (
            safe_stdev(values) / statistics.mean(clean_values)
            if statistics.mean(clean_values) != 0
            else float("inf")
        ),
    }

    # Metric-specific analysis based on known metric types
    metric_lower = metric_name.lower()

    # Determine if higher or lower is better based on specific metric knowledge
    if any(x in metric_lower for x in ["similarity", "sim", "correlation", "corr"]):
        analysis["higher_is_better"] = True
        analysis["interpretation"] = (
            "Higher values indicate better similarity/correlation"
        )
    elif any(
        x in metric_lower
        for x in [
            "mos",
            "quality",
            "nisqa",
            "utmos",
            "singmos",
            "pesq",
            "stoi",
            "dnsmos_pro_bvcc",
            "dnsmos_pro_nisqa",
            "dnsmos_pro_vcc2018",
        ]
    ):
        analysis["higher_is_better"] = True
        analysis["interpretation"] = "Higher values indicate better audio quality"
    elif any(
        x in metric_lower
        for x in ["sdr", "sir", "sar", "si-snr", "ci-sdr", "si_snr", "ci_sdr"]
    ):
        analysis["higher_is_better"] = True
        analysis["interpretation"] = (
            "Higher values indicate better signal separation/enhancement"
        )
    elif any(x in metric_lower for x in ["wer", "cer", "error", "rmse"]):
        analysis["higher_is_better"] = False
        analysis["interpretation"] = (
            "Lower values indicate better recognition/lower error"
        )
    elif (
        any(x in metric_lower for x in ["distance", "dtw"])
        and "cosine" not in metric_lower
    ):
        analysis["higher_is_better"] = False
        analysis["interpretation"] = (
            "Lower values indicate better similarity (distance metric)"
        )
    elif "mcd" in metric_lower:
        analysis["higher_is_better"] = False
        analysis["interpretation"] = (
            "Lower MCD indicates better mel-cepstral similarity"
        )
    elif any(x in metric_lower for x in ["asvspoof", "spoof"]):
        analysis["higher_is_better"] = None  # Depends on threshold
        analysis["interpretation"] = (
            "Spoofing detection score - interpretation depends on threshold"
        )
    elif "clap" in metric_lower:
        analysis["higher_is_better"] = True
        analysis["interpretation"] = (
            "Higher CLAP score indicates better audio-text alignment"
        )
    else:
        analysis["higher_is_better"] = None
        analysis["interpretation"] = "Direction unclear from metric name"

    # Set best/worst values
    if analysis["higher_is_better"] is True:
        analysis["best_value"] = analysis["max"]
        analysis["worst_value"] = analysis["min"]
    elif analysis["higher_is_better"] is False:
        analysis["best_value"] = analysis["min"]
        analysis["worst_value"] = analysis["max"]
    else:
        analysis["best_value"] = None
        analysis["worst_value"] = None

    # Quality assessment
    if analysis["cv"] < 0.1:
        analysis["variability"] = "low"
        analysis["quality_note"] = (
            "Low variability - metric might not be discriminative"
        )
    elif analysis["cv"] > 2.0:
        analysis["variability"] = "very_high"
        analysis["quality_note"] = (
            "Very high variability - check for outliers or measurement issues"
        )
    elif analysis["cv"] > 0.5:
        analysis["variability"] = "high"
        analysis["quality_note"] = "High variability - good discriminative power"
    else:
        analysis["variability"] = "moderate"
        analysis["quality_note"] = (
            "Moderate variability - reasonable discriminative power"
        )

    # Range analysis with metric-specific validation
    if analysis["range"] == 0:
        analysis["range_note"] = "All values identical - no discriminative power"
    elif "similarity" in metric_lower and (analysis["min"] < 0 or analysis["max"] > 1):
        analysis["range_note"] = "Unusual range for similarity metric (expected 0-1)"
    elif "correlation" in metric_lower and (
        analysis["min"] < -1 or analysis["max"] > 1
    ):
        analysis["range_note"] = (
            "Unusual range for correlation metric (expected -1 to 1)"
        )
    elif any(x in metric_lower for x in ["mos", "nisqa", "utmos"]) and (
        analysis["min"] < 1 or analysis["max"] > 5
    ):
        analysis["range_note"] = "Unusual range for MOS-scale metric (expected 1-5)"
    elif "pesq" in metric_lower and (analysis["min"] < -0.5 or analysis["max"] > 4.5):
        analysis["range_note"] = "Unusual range for PESQ metric (expected -0.5 to 4.5)"
    elif "stoi" in metric_lower and (analysis["min"] < 0 or analysis["max"] > 1):
        analysis["range_note"] = "Unusual range for STOI metric (expected 0-1)"
    elif any(x in metric_lower for x in ["wer", "cer"]) and analysis["min"] < 0:
        analysis["range_note"] = "Negative error rates detected - check calculation"

    return analysis


def create_visualizations(metrics_stats: Dict, discovery: Dict, output_dir: str = None):
    """Create visualizations organized by metric categories"""
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create category-based visualizations
    for category, metrics in discovery["metric_categories"].items():
        if not metrics or len(metrics) == 0:
            continue

        print(f"Creating visualization for {category} metrics ({len(metrics)} metrics)")

        # Collect data for this category
        category_data = {}
        for metric_name in metrics:
            if metric_name in metrics_stats and metrics_stats[metric_name]["values"]:
                category_data[metric_name] = metrics_stats[metric_name]

        if not category_data:
            continue

        metrics_in_category = list(category_data.keys())
        n_metrics = len(metrics_in_category)

        if n_metrics == 1:
            # Single metric - create detailed plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            metric = metrics_in_category[0]
            data = category_data[metric]

            # Histogram
            ax1.hist(data["clean_values"], bins=20, alpha=0.7, edgecolor="black")
            ax1.set_title(f"{metric} - Distribution")
            ax1.set_xlabel("Value")
            ax1.set_ylabel("Frequency")
            ax1.grid(True, alpha=0.3)

            # Add statistics text
            stats_text = f"Mean: {data['mean']:.4f}\nStd: {data['std']:.4f}\nMin: {data['min']:.4f}\nMax: {data['max']:.4f}"
            ax1.text(
                0.05,
                0.95,
                stats_text,
                transform=ax1.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            # Box plot
            ax2.boxplot(data["clean_values"])
            ax2.set_title(f"{metric} - Box Plot")
            ax2.set_ylabel("Value")
            ax2.grid(True, alpha=0.3)

        else:
            # Multiple metrics - grid layout
            n_cols = min(4, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()

            for i, metric in enumerate(metrics_in_category):
                if i >= len(axes):
                    break

                ax = axes[i]
                data = category_data[metric]

                # Histogram for each metric
                ax.hist(data["clean_values"], bins=15, alpha=0.7, edgecolor="black")
                ax.set_title(f"{metric}")
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)

                # Add mean line
                ax.axvline(
                    data["mean"],
                    color="red",
                    linestyle="--",
                    alpha=0.8,
                    label=f'Mean: {data["mean"]:.3f}',
                )
                ax.legend(fontsize=8)

            # Hide empty subplots
            for i in range(n_metrics, len(axes)):
                axes[i].set_visible(False)

        plt.suptitle(f"{category.title()} Metrics Analysis", fontsize=16)
        plt.tight_layout()
        if output_dir:
            plt.savefig(
                os.path.join(output_dir, f"{category}_metrics_analysis.png"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()

    # Create correlation heatmap if we have multiple metrics
    if len(discovery["all_metrics"]) > 1:
        print("Creating correlation heatmap...")

        # Prepare data for correlation analysis
        correlation_data = {}
        min_samples = float("inf")

        for metric_name in discovery["all_metrics"]:
            if metric_name in metrics_stats and metrics_stats[metric_name]["values"]:
                correlation_data[metric_name] = metrics_stats[metric_name][
                    "clean_values"
                ]
                min_samples = min(min_samples, len(correlation_data[metric_name]))

        if len(correlation_data) > 1 and min_samples > 1:
            # Truncate all arrays to same length for correlation
            for metric_name in correlation_data:
                correlation_data[metric_name] = correlation_data[metric_name][
                    :min_samples
                ]

            # Create DataFrame and compute correlation
            df_corr = pd.DataFrame(correlation_data)
            correlation_matrix = df_corr.corr()

            # Plot heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(
                correlation_matrix,
                mask=mask,
                annot=True,
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
            )
            plt.title("Metric Correlation Matrix")
            plt.tight_layout()
            if output_dir:
                plt.savefig(
                    os.path.join(output_dir, "correlation_heatmap.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
            plt.show()


def print_metric_analysis(metrics_stats: Dict, discovery: Dict):
    """Print detailed analysis of all metrics with quality estimates"""
    print(f"\n{'='*120}")
    print("DETAILED METRIC ANALYSIS")
    print(f"{'='*120}")

    for category, metrics in discovery["metric_categories"].items():
        if not metrics:
            continue

        print(f"\n{category.upper()} METRICS ({len(metrics)} metrics):")
        print("-" * 80)

        for metric in sorted(metrics):
            if metric not in metrics_stats:
                continue

            data = metrics_stats[metric]
            analysis = estimate_metric_quality(metric, data["values"])

            print(f"\n  {metric}:")
            print(f"    Samples: {analysis['count']}")
            print(f"    Mean: {analysis['mean']:.4f}, Median: {analysis['median']:.4f}")
            print(
                f"    Std: {analysis['std']:.4f}, Range: [{analysis['min']:.4f}, {analysis['max']:.4f}]"
            )
            print(
                f"    CV: {analysis['cv']:.4f} ({analysis['variability']} variability)"
            )
            print(f"    Quality: {analysis['quality_note']}")

            if analysis.get("range_note"):
                print(f"    Range note: {analysis['range_note']}")
            if analysis.get("higher_is_better") is not None:
                direction = "higher" if analysis["higher_is_better"] else "lower"
                best_val = analysis["best_value"]
                print(
                    f"    Interpretation: {direction} values are better (best: {best_val:.4f})"
                )


def export_results_to_csv(metrics_stats: Dict, discovery: Dict, output_file: str):
    """Export comprehensive results to CSV files"""
    # Main results file
    results_data = []

    for metric_name in sorted(discovery["all_metrics"]):
        if metric_name not in metrics_stats:
            continue

        data = metrics_stats[metric_name]
        analysis = estimate_metric_quality(metric_name, data["values"])

        # Find category
        category = "other"
        for cat, metrics in discovery["metric_categories"].items():
            if metric_name in metrics:
                category = cat
                break

        row = {
            "metric_name": metric_name,
            "category": category,
            "count": analysis["count"],
            "mean": analysis["mean"],
            "median": analysis["median"],
            "std": analysis["std"],
            "min": analysis["min"],
            "max": analysis["max"],
            "range": analysis["range"],
            "cv": analysis["cv"],
            "variability": analysis["variability"],
            "quality_note": analysis["quality_note"],
        }

        if analysis.get("higher_is_better") is not None:
            row["higher_is_better"] = analysis["higher_is_better"]
            row["best_value"] = analysis["best_value"]
            row["worst_value"] = analysis["worst_value"]

        results_data.append(row)

    df = pd.DataFrame(results_data)
    df.to_csv(output_file, index=False)
    print(f"Results exported to: {output_file}")

    # Per-utterance detailed results
    detailed_file = output_file.replace(".csv", "_detailed.csv")
    # This would require restructuring the data, keeping it simple for now
    print(f"Summary results saved. Use --output-jsonl for per-utterance details.")


def main():
    parser = argparse.ArgumentParser("Standalone Metrics Analysis Tool")
    parser.add_argument(
        "input", type=str, help="Input JSONL file or directory containing JSONL files"
    )
    parser.add_argument(
        "--per-utt", action="store_true", help="Show per-utterance results in console"
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="Output per-utterance results to JSONL file with computed statistics",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=None,
        help="Specific metrics to analyze (if not specified, analyzes all discovered metrics)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization plots organized by metric categories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots and CSV export",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export results to CSV files with metric analysis",
    )
    parser.add_argument(
        "--show-discovery",
        action="store_true",
        help="Show what metrics were discovered in the data",
    )
    parser.add_argument(
        "--analyze-metrics",
        action="store_true",
        help="Show detailed analysis of metric quality and characteristics",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help="Specific metric categories to analyze (similarity, chroma, dtw, spectral, etc.)",
    )

    args = parser.parse_args()

    # Read all data from file(s)
    info = read_input_files(args.input)

    if not info:
        print("No data found in input file")
        return

    # Discover all metrics and categorize them
    discovery = discover_metrics(info)

    if args.show_discovery:
        print(f"\n{'='*80}")
        print("DISCOVERY RESULTS")
        print(f"{'='*80}")
        print(f"Found {len(discovery['all_metrics'])} total metrics")
        print(
            f"Ignored {len(discovery['ignored_fields'])} non-numeric fields: {sorted(discovery['ignored_fields'])}"
        )

        print(f"\nMetric categories:")
        for category, metrics in discovery["metric_categories"].items():
            if metrics:
                print(f"  {category}: {len(metrics)} metrics")
                for metric in sorted(metrics):
                    print(f"    - {metric}")

    # Filter by categories if specified
    if args.categories:
        filtered_categories = {}
        for cat in args.categories:
            cat_lower = cat.lower()
            for existing_cat, metrics in discovery["metric_categories"].items():
                if cat_lower in existing_cat.lower():
                    filtered_categories[existing_cat] = metrics
        discovery["metric_categories"] = filtered_categories
        print(
            f"Filtering to {len(filtered_categories)} categories: {list(filtered_categories.keys())}"
        )

    # Filter metrics if specified
    if args.metrics:
        filtered_metrics = set()
        for specified_metric in args.metrics:
            for metric in discovery["all_metrics"]:
                if specified_metric.lower() in metric.lower():
                    filtered_metrics.add(metric)
        discovery["all_metrics"] = filtered_metrics
        # Update categories
        for category in discovery["metric_categories"]:
            discovery["metric_categories"][category] = discovery["metric_categories"][
                category
            ].intersection(filtered_metrics)

    # Collect and analyze all metrics
    metrics_stats = {}
    per_utt_results = []

    for metric_name in discovery["all_metrics"]:
        metrics_stats[metric_name] = {
            "values": [],
            "clean_values": [],
            "mean": 0,
            "std": 0,
            "min": 0,
            "max": 0,
            "count": 0,
        }

    # Process each sample
    for i, item in enumerate(info):
        # Start with original data
        utt_result = item.copy()

        # Collect all metric values
        for metric_name in discovery["all_metrics"]:
            if metric_name in item and isinstance(item[metric_name], (int, float)):
                value = item[metric_name]
                metrics_stats[metric_name]["values"].append(value)

                # Track clean values (no inf/nan)
                if not (
                    value == float("inf") or value == float("-inf") or value != value
                ):
                    metrics_stats[metric_name]["clean_values"].append(value)

        per_utt_results.append(utt_result)

    # Calculate statistics for each metric
    for metric_name in metrics_stats:
        data = metrics_stats[metric_name]
        if data["clean_values"]:
            data["mean"] = statistics.mean(data["clean_values"])
            data["std"] = safe_stdev(data["values"])
            data["min"] = min(data["clean_values"])
            data["max"] = max(data["clean_values"])
            data["count"] = len(data["clean_values"])

    # Print per-utterance results if requested
    if args.per_utt:
        print(f"\n{'='*120}")
        print("PER-UTTERANCE RESULTS")
        print(f"{'='*120}")

        # Show all metrics (limit display for readability)
        display_metrics = sorted(discovery["all_metrics"])[:15]  # Show top 15 metrics

        # Header
        header = f"{'Utterance ID':<30}"
        for metric in display_metrics:
            header += f" {metric:<18}"
        print(header)
        print("-" * len(header))

        # Data rows (show first 20 utterances)
        display_count = min(20, len(per_utt_results))
        for i in range(display_count):
            result = per_utt_results[i]
            row = f"{result.get('key', f'utt_{i+1}'):<30}"
            for metric in display_metrics:
                value = result.get(metric, 0)
                if isinstance(value, float):
                    if value == float("inf"):
                        value_str = "inf"
                    else:
                        value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                row += f" {value_str:>18}"
            print(row)

        if len(per_utt_results) > 20:
            print(f"\n... showing first 20 of {len(per_utt_results)} utterances")
            print("Use --output-jsonl to save all results to file")

        if len(discovery["all_metrics"]) > 15:
            print(f"\n... showing 15 of {len(discovery['all_metrics'])} metrics")
            remaining_metrics = sorted(
                list(discovery["all_metrics"] - set(display_metrics))
            )
            print("Remaining metrics:", remaining_metrics)

    # Output to JSONL if requested
    if args.output_jsonl:
        with open(args.output_jsonl, "w", encoding="utf-8") as jsonl_file:
            for result in per_utt_results:
                # Handle inf values for JSON serialization
                clean_result = {}
                for k, v in result.items():
                    if v == float("inf"):
                        clean_result[k] = "inf"
                    elif v == float("-inf"):
                        clean_result[k] = "-inf"
                    else:
                        clean_result[k] = v
                jsonl_file.write(json.dumps(clean_result, ensure_ascii=False) + "\n")
        print(f"\nPer-utterance results saved to: {args.output_jsonl}")
        print(f"Total utterances saved: {len(per_utt_results)}")

    # Print overall results
    print(f"\n{'='*120}")
    print(f"OVERALL RESULTS ({len(info)} utterances)")
    print(f"{'='*120}")

    # Print metrics organized by category
    for category, metrics in discovery["metric_categories"].items():
        if not metrics:
            continue

        category_metrics_with_data = []
        for metric in metrics:
            if metric in metrics_stats and metrics_stats[metric]["count"] > 0:
                category_metrics_with_data.append(metric)

        if not category_metrics_with_data:
            continue

        print(
            f"\n{category.upper()} METRICS ({len(category_metrics_with_data)} metrics):"
        )
        print("-" * 80)

        for metric_name in sorted(category_metrics_with_data):
            data = metrics_stats[metric_name]
            print(f"  {metric_name}:")
            print(
                f"    Count: {data['count']}, Mean: {data['mean']:.4f}, Std: {data['std']:.4f}"
            )
            print(f"    Range: [{data['min']:.4f}, {data['max']:.4f}]")

    # Summary table
    print(f"\n{'='*120}")
    print("SUMMARY TABLE")
    print(f"{'='*120}")

    # Create summary table
    header = f"{'Metric':<35} {'Category':<15} {'Count':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}"
    print(header)
    print("-" * len(header))

    for metric_name in sorted(discovery["all_metrics"]):
        if metric_name not in metrics_stats or metrics_stats[metric_name]["count"] == 0:
            continue

        data = metrics_stats[metric_name]

        # Find category
        category = "other"
        for cat, metrics in discovery["metric_categories"].items():
            if metric_name in metrics:
                category = cat
                break

        row = f"{metric_name:<35} {category:<15} {data['count']:<8} {data['mean']:<12.4f} {data['std']:<12.4f} {data['min']:<12.4f} {data['max']:<12.4f}"
        print(row)

    # Show detailed metric analysis if requested
    if args.analyze_metrics:
        print_metric_analysis(metrics_stats, discovery)

    # Create visualizations if requested
    if args.visualize:
        try:
            create_visualizations(metrics_stats, discovery, args.output_dir)
        except ImportError:
            print("Warning: matplotlib/seaborn not available. Skipping visualizations.")
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")

    # Export to CSV if requested
    if args.export_csv:
        try:
            output_file = (
                os.path.join(args.output_dir, "metrics_analysis.csv")
                if args.output_dir
                else "metrics_analysis.csv"
            )
            export_results_to_csv(metrics_stats, discovery, output_file)
        except ImportError:
            print("Warning: pandas not available. Skipping CSV export.")
        except Exception as e:
            print(f"Warning: Could not export to CSV: {e}")

    # Print final summary
    print(f"\n{'='*120}")
    print("SUMMARY")
    print(f"{'='*120}")
    print(f"Processed {len(info)} utterances")
    print(
        f"Analyzed {len([m for m in discovery['all_metrics'] if m in metrics_stats and metrics_stats[m]['count'] > 0])} metrics"
    )
    print(
        f"Metric categories: {[cat for cat, metrics in discovery['metric_categories'].items() if metrics]}"
    )
    print(
        f"Total data points: {sum(metrics_stats[m]['count'] for m in metrics_stats if metrics_stats[m]['count'] > 0)}"
    )


if __name__ == "__main__":
    main()

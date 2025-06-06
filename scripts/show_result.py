import json
import argparse
from collections import defaultdict
import statistics
import os
import glob

def calculate_wer_cer(delete, insert, replace, equal):
    """Calculate WER or CER from component counts"""
    total_ref = delete + replace + equal  # Total reference tokens/chars
    total_errors = delete + insert + replace
    
    if total_ref == 0:
        return 0.0 if total_errors == 0 else float('inf')
    
    return total_errors / total_ref


def read_input_files(input_path):
    """Read JSONL files from either a single file or directory"""
    files_to_process = []
    
    if os.path.isfile(input_path):
        files_to_process = [input_path]
    elif os.path.isdir(input_path):
        # Look for common JSONL file patterns
        patterns = ['*.jsonl', '*.json', '*.jl', '*.txt']  # Added .txt for your case
        for pattern in patterns:
            files_to_process.extend(glob.glob(os.path.join(input_path, pattern)))
        
        if not files_to_process:
            print(f"No JSONL files found in directory: {input_path}")
            return []
    else:
        print(f"Input path does not exist: {input_path}")
        return []
    
    print(f"Processing {len(files_to_process)} file(s): {[os.path.basename(f) for f in files_to_process]}")
    
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
                            data['_source_file'] = os.path.basename(file_path)
                            all_data.append(data)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Invalid JSON on line {line_num} in {file_path}: {e}")
                        except Exception as e:
                            print(f"Warning: Error parsing line {line_num} in {file_path}: {e}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    return all_data


def main():
    parser = argparse.ArgumentParser("WER/CER Aggregation Tool")
    parser.add_argument("input", type=str, help="Input JSONL file or directory containing JSONL files")
    parser.add_argument("--per-utt", action="store_true", 
                       help="Show per-utterance results in console")
    parser.add_argument("--output-jsonl", type=str, default=None,
                       help="Output per-utterance results to JSONL file with original data preserved")
    args = parser.parse_args()

    # Read all data from file(s)
    info = read_input_files(args.input)

    if not info:
        print("No data found in input file")
        return

    # Find all model prefixes (e.g., "whisper_", "espnet_", "owsm_")
    model_prefixes = set()
    for item in info:
        for key in item.keys():
            if "_wer_delete" in key or "_cer_delete" in key:
                prefix = key.split("_wer_delete")[0] + "_" if "_wer_delete" in key else key.split("_cer_delete")[0] + "_"
                model_prefixes.add(prefix)

    # Store per-utterance results with original data preserved
    per_utt_results = []
    
    # Aggregate statistics for each model (corpus-level)
    model_stats = defaultdict(lambda: {
        'wer': {'delete': 0, 'insert': 0, 'replace': 0, 'equal': 0},
        'cer': {'delete': 0, 'insert': 0, 'replace': 0, 'equal': 0},
        'per_utt_wer': [],
        'per_utt_cer': [],
        'other_metrics': defaultdict(list)
    })

    # Process each sample
    for i, item in enumerate(info):
        # Start with original data (preserve everything)
        utt_result = item.copy()
        
        for prefix in model_prefixes:
            # Check if this model's metrics exist in current item
            wer_delete_key = f"{prefix}wer_delete"
            if wer_delete_key not in item:
                continue
                
            # Aggregate WER/CER components for corpus-level calculation
            for component in ['delete', 'insert', 'replace', 'equal']:
                wer_key = f"{prefix}wer_{component}"
                cer_key = f"{prefix}cer_{component}"
                
                if wer_key in item:
                    model_stats[prefix]['wer'][component] += item[wer_key]
                if cer_key in item:
                    model_stats[prefix]['cer'][component] += item[cer_key]
            
            # Calculate per-utterance WER/CER
            utt_wer = calculate_wer_cer(
                item.get(f"{prefix}wer_delete", 0),
                item.get(f"{prefix}wer_insert", 0),
                item.get(f"{prefix}wer_replace", 0),
                item.get(f"{prefix}wer_equal", 0)
            )
            utt_cer = calculate_wer_cer(
                item.get(f"{prefix}cer_delete", 0),
                item.get(f"{prefix}cer_insert", 0),
                item.get(f"{prefix}cer_replace", 0),
                item.get(f"{prefix}cer_equal", 0)
            )
            
            # Add computed WER/CER to the result (with clear naming)
            model_name = prefix.rstrip('_')
            utt_result[f'{model_name}_computed_wer'] = utt_wer
            utt_result[f'{model_name}_computed_cer'] = utt_cer
            
            # Store for statistics
            if utt_wer != float('inf'):
                model_stats[prefix]['per_utt_wer'].append(utt_wer)
            if utt_cer != float('inf'):
                model_stats[prefix]['per_utt_cer'].append(utt_cer)
            
            # Collect other metrics for averaging (both prefixed and non-prefixed)
            for key, value in item.items():
                # Skip WER/CER components, key field, and text fields
                if "_wer_" in key or "_cer_" in key or key == "key" or "text" in key:
                    continue
                
                if isinstance(value, (int, float)):
                    # Store metric under the current model prefix for tracking
                    model_stats[prefix]['other_metrics'][key].append(value)
        
        per_utt_results.append(utt_result)

    # Print per-utterance results if requested
    if args.per_utt:
        print(f"\n{'='*120}")
        print("PER-UTTERANCE RESULTS")
        print(f"{'='*120}")
        
        # Show key metrics in a clean table format
        computed_metrics = []
        for prefix in sorted(model_prefixes):
            model_name = prefix.rstrip('_')
            computed_metrics.extend([f'{model_name}_computed_wer', f'{model_name}_computed_cer'])
        
        # Add other common metrics
        other_metrics = ['spk_similarity', 'emotion_similarity']
        for metric in other_metrics:
            if any(metric in result for result in per_utt_results):
                computed_metrics.append(metric)
        
        # Header
        header = f"{'Utterance ID':<25}"
        for metric in computed_metrics:
            header += f" {metric:<15}"
        print(header)
        print("-" * len(header))
        
        # Data rows (show first 20 utterances, then offer to continue)
        display_count = min(20, len(per_utt_results))
        for i in range(display_count):
            result = per_utt_results[i]
            row = f"{result.get('key', f'utt_{i+1}'):<25}"
            for metric in computed_metrics:
                value = result.get(metric, 0)
                if isinstance(value, float):
                    if value == float('inf'):
                        value_str = "inf"
                    else:
                        value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                row += f" {value_str:>15}"
            print(row)
        
        if len(per_utt_results) > 20:
            print(f"\n... showing first 20 of {len(per_utt_results)} utterances")
            print("Use --output-jsonl to save all results to file")

    # Output to JSONL if requested
    if args.output_jsonl:
        with open(args.output_jsonl, 'w', encoding='utf-8') as jsonl_file:
            for result in per_utt_results:
                # Handle inf values for JSON serialization
                clean_result = {}
                for k, v in result.items():
                    if v == float('inf'):
                        clean_result[k] = "inf"
                    elif v == float('-inf'):
                        clean_result[k] = "-inf"
                    else:
                        clean_result[k] = v
                jsonl_file.write(json.dumps(clean_result, ensure_ascii=False) + '\n')
        print(f"\nPer-utterance results with original data saved to: {args.output_jsonl}")
        print(f"Total utterances saved: {len(per_utt_results)}")

    # Print overall (corpus-level) results
    print(f"\n{'='*120}")
    print(f"OVERALL (CORPUS-LEVEL) RESULTS ({len(info)} utterances)")
    print(f"{'='*120}")

    for prefix in sorted(model_prefixes):
        stats = model_stats[prefix]
        model_name = prefix.rstrip('_')
        
        # Calculate corpus-level WER and CER
        corpus_wer = calculate_wer_cer(
            stats['wer']['delete'],
            stats['wer']['insert'], 
            stats['wer']['replace'],
            stats['wer']['equal']
        )
        
        corpus_cer = calculate_wer_cer(
            stats['cer']['delete'],
            stats['cer']['insert'],
            stats['cer']['replace'], 
            stats['cer']['equal']
        )
        
        # Calculate per-utterance statistics for all metrics
        per_utt_wer_stats = {}
        per_utt_cer_stats = {}
        other_metric_stats = {}
        

        # Safe standard deviation calculation
        def safe_stdev(values):
            try:
                if len(values) <= 1:
                    return 0.0
                
                # Filter out inf and nan values
                clean_values = [v for v in values if not (v == float('inf') or v == float('-inf') or v != v)]
                
                if len(clean_values) <= 1:
                    return 0.0
                
                # Check if all values are the same (or very close)
                if max(clean_values) - min(clean_values) < 1e-10:
                    return 0.0
                
                # Manual calculation to avoid statistics module issues
                mean_val = sum(clean_values) / len(clean_values)
                variance = sum((x - mean_val) ** 2 for x in clean_values) / (len(clean_values) - 1)
                
                if variance < 0:
                    return 0.0
                
                import math
                return math.sqrt(variance)
                
            except (statistics.StatisticsError, ValueError, ZeroDivisionError, OverflowError):
                return 0.0       
 
        if stats['per_utt_wer']:
            per_utt_wer_stats = {
                'mean': statistics.mean(stats['per_utt_wer']),
                'median': statistics.median(stats['per_utt_wer']),
                'std': safe_stdev(stats['per_utt_wer']),
                'min': min(stats['per_utt_wer']),
                'max': max(stats['per_utt_wer'])
            }
        
        if stats['per_utt_cer']:
            per_utt_cer_stats = {
                'mean': statistics.mean(stats['per_utt_cer']),
                'median': statistics.median(stats['per_utt_cer']),
                'std': safe_stdev(stats['per_utt_cer']),
                'min': min(stats['per_utt_cer']),
                'max': max(stats['per_utt_cer'])
            }
        
        # Calculate statistics for other metrics
        for metric_name, values in stats['other_metrics'].items():
            if values:
                # Filter out inf and nan values
                clean_values = [v for v in values if not (v == float('inf') or v == float('-inf') or v != v)]
               
                other_metric_stats[metric_name] = {
                    'mean': statistics.mean(clean_values),
                    'median': statistics.median(clean_values),
                    'std': safe_stdev(values),
                    'min': min(values),
                    'max': max(values)
                }
        
        print(f"\n{model_name.upper()} RESULTS:")
        print(f"  Corpus-level WER: {corpus_wer:.4f} ({corpus_wer*100:.2f}%)")
        print(f"  Corpus-level CER: {corpus_cer:.4f} ({corpus_cer*100:.2f}%)")
        
        if per_utt_wer_stats:
            print(f"  Per-utterance WER stats:")
            print(f"    Mean: {per_utt_wer_stats['mean']:.4f}, Median: {per_utt_wer_stats['median']:.4f}")
            print(f"    Std: {per_utt_wer_stats['std']:.4f}, Min: {per_utt_wer_stats['min']:.4f}, Max: {per_utt_wer_stats['max']:.4f}")
        
        if per_utt_cer_stats:
            print(f"  Per-utterance CER stats:")
            print(f"    Mean: {per_utt_cer_stats['mean']:.4f}, Median: {per_utt_cer_stats['median']:.4f}")
            print(f"    Std: {per_utt_cer_stats['std']:.4f}, Min: {per_utt_cer_stats['min']:.4f}, Max: {per_utt_cer_stats['max']:.4f}")
        
        # Print component breakdown
        print(f"  WER Components - Delete: {stats['wer']['delete']}, Insert: {stats['wer']['insert']}, Replace: {stats['wer']['replace']}, Equal: {stats['wer']['equal']}")
        print(f"  CER Components - Delete: {stats['cer']['delete']}, Insert: {stats['cer']['insert']}, Replace: {stats['cer']['replace']}, Equal: {stats['cer']['equal']}")
        
        # Print other metrics with full statistics
        if other_metric_stats:
            print(f"  Other metrics statistics:")
            for metric_name, metric_stats in sorted(other_metric_stats.items()):
                # Clean up metric name for display (remove model prefix if present)
                display_name = metric_name
                if display_name.startswith(prefix):
                    display_name = display_name[len(prefix):]
                
                print(f"    {display_name}:")
                print(f"      Mean: {metric_stats['mean']:.4f}, Median: {metric_stats['median']:.4f}")
                print(f"      Std: {metric_stats['std']:.4f}, Min: {metric_stats['min']:.4f}, Max: {metric_stats['max']:.4f}")

    # Summary comparison if multiple models
    if len(model_prefixes) > 1:
        print(f"\n{'='*120}")
        print("SUMMARY COMPARISON")
        print(f"{'='*120}")
        
        # Collect all metrics for comparison
        all_comparison_metrics = set()
        for prefix in model_prefixes:
            stats = model_stats[prefix]
            all_comparison_metrics.add('corpus_wer')
            all_comparison_metrics.add('mean_wer')
            all_comparison_metrics.add('corpus_cer')
            all_comparison_metrics.add('mean_cer')
            
            # Add all other metrics (they'll be the same across models for non-prefixed metrics)
            for metric_name in stats['other_metrics'].keys():
                # Clean metric name for comparison table
                clean_name = metric_name
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix):]
                all_comparison_metrics.add(f"mean_{clean_name}")
        
        all_comparison_metrics = sorted(all_comparison_metrics)
        
        # Print header
        header = f"{'Model':<15}"
        for metric in all_comparison_metrics:
            header += f" {metric:<15}"
        print(header)
        print("-" * len(header))
        
        # Print data for each model
        for prefix in sorted(model_prefixes):
            stats = model_stats[prefix]
            model_name = prefix.rstrip('_')
            
            corpus_wer = calculate_wer_cer(stats['wer']['delete'], stats['wer']['insert'], 
                                         stats['wer']['replace'], stats['wer']['equal'])
            corpus_cer = calculate_wer_cer(stats['cer']['delete'], stats['cer']['insert'],
                                         stats['cer']['replace'], stats['cer']['equal'])
            
            mean_wer = statistics.mean(stats['per_utt_wer']) if stats['per_utt_wer'] else 0
            mean_cer = statistics.mean(stats['per_utt_cer']) if stats['per_utt_cer'] else 0
            
            row = f"{model_name:<15}"
            for metric in all_comparison_metrics:
                if metric == 'corpus_wer':
                    value = corpus_wer
                elif metric == 'mean_wer':
                    value = mean_wer
                elif metric == 'corpus_cer':
                    value = corpus_cer
                elif metric == 'mean_cer':
                    value = mean_cer
                elif metric.startswith('mean_'):
                    clean_metric = metric[5:]  # Remove 'mean_' prefix
                    # Look for this metric in various forms (with/without model prefix)
                    value = 0
                    for original_metric in stats['other_metrics']:
                        metric_suffix = original_metric
                        if metric_suffix.startswith(prefix):
                            metric_suffix = metric_suffix[len(prefix):]
                        
                        if metric_suffix == clean_metric and stats['other_metrics'][original_metric]:
                            value = statistics.mean(stats['other_metrics'][original_metric])
                            break
                else:
                    value = 0
                
                row += f" {value:>15.4f}"
            print(row)

if __name__ == "__main__":
    main()

import json
import os


def compute_error_rates(data):
    """
    Compute WER and individual error rates from whisper error counts

    WER = (deletions + insertions + substitutions) / total_words_in_reference
    Insertion Error Rate = insertions / total_words_in_reference
    Deletion Error Rate = deletions / total_words_in_reference
    Substitution Error Rate = substitutions / total_words_in_reference
    """

    # Extract error counts
    deletions = data["whisper_wer_delete"]
    insertions = data["whisper_wer_insert"]
    substitutions = data["whisper_wer_replace"]
    correct_words = data["whisper_wer_equal"]

    # Total words in reference = correct + deletions + substitutions
    # (insertions don't count toward reference length)
    total_ref_words = correct_words + deletions + substitutions

    # Avoid division by zero
    if total_ref_words == 0:
        return {
            "wer": 0.0,
            "insertion_error": 0.0,
            "deletion_error": 0.0,
            "substitution_error": 0.0,
        }

    # Calculate error rates
    wer = (deletions + insertions + substitutions) / total_ref_words
    insertion_error = insertions / total_ref_words
    deletion_error = deletions / total_ref_words
    substitution_error = substitutions / total_ref_words

    return {
        "wer": wer,
        "insertion_error": insertion_error,
        "deletion_error": deletion_error,
        "substitution_error": substitution_error,
    }


def process_jsonl_file(input_file, output_dir="filtered_results"):
    """
    Process JSONL file and create filtered versions based on error rate thresholds
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read and process all lines
    processed_lines = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())

                # Compute error rates
                error_rates = compute_error_rates(data)

                # Add computed rates to the data
                enhanced_data = data.copy()
                enhanced_data.update(error_rates)

                processed_lines.append(enhanced_data)

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except KeyError as e:
                print(f"Missing required field in line {line_num}: {e}")
                continue

    print(f"Processed {len(processed_lines)} lines")

    # Define filtering criteria
    filters = {
        "wer_lt_0.05": lambda x: x["wer"] < 0.05,
        "insertion_error_lt_0.05": lambda x: x["insertion_error"] < 0.05,
        "deletion_error_lt_0.05": lambda x: x["deletion_error"] < 0.05,
        "deletion_inseration_error_lt_0.05": lambda x: x["insertion_error"] < 0.05 and x["deletion_error"] < 0.05, 
    }

    # Apply filters and save results
    filter_results = {}

    for filter_name, filter_func in filters.items():
        filtered_data = [line for line in processed_lines if filter_func(line)]
        filter_results[filter_name] = len(filtered_data)

        # Save filtered data to JSONL file
        output_file = os.path.join(output_dir, f"{filter_name}.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for item in filtered_data:
                f.write(json.dumps(item) + "\n")

        print(f"{filter_name}: {len(filtered_data)} lines saved to {output_file}")

    # Save all processed data (with computed error rates)
    all_processed_file = os.path.join(output_dir, "all_with_error_rates.jsonl")
    with open(all_processed_file, "w", encoding="utf-8") as f:
        for item in processed_lines:
            f.write(json.dumps(item) + "\n")

    print(f"All processed data saved to {all_processed_file}")

    # Print summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)

    if processed_lines:
        wer_values = [x["wer"] for x in processed_lines]
        insertion_values = [x["insertion_error"] for x in processed_lines]
        deletion_values = [x["deletion_error"] for x in processed_lines]
        substitution_values = [x["substitution_error"] for x in processed_lines]

        print(f"Total processed lines: {len(processed_lines)}")
        print(f"Average WER: {sum(wer_values)/len(wer_values):.4f}")
        print(
            f"Average Insertion Error: {sum(insertion_values)/len(insertion_values):.4f}"
        )
        print(
            f"Average Deletion Error: {sum(deletion_values)/len(deletion_values):.4f}"
        )
        print(
            f"Average Substitution Error: {sum(substitution_values)/len(substitution_values):.4f}"
        )

        print(f"\nMin WER: {min(wer_values):.4f}")
        print(f"Max WER: {max(wer_values):.4f}")

        print("\nFilter Results:")
        for filter_name, count in filter_results.items():
            percentage = (count / len(processed_lines)) * 100
            print(f"  {filter_name}: {count} lines ({percentage:.1f}%)")


def main():
    """
    Main function to run the analysis
    """
    input_file = input("Enter the path to your JSONL file: ").strip()

    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    try:
        process_jsonl_file(input_file)
        print(
            f"\nAnalysis complete! Check the 'filtered_results' directory for output files."
        )

    except Exception as e:
        print(f"Error processing file: {e}")


# Example usage for testing with sample data
def test_with_sample():
    """
    Test function with the provided sample data
    """
    sample_data = {
        "key": "Bones_04x04_000061_1842.83_1858.06",
        "whisper_hyp_text": "there are people around here who seem to like you very much people who are concerned for your happiness what is this",
        "ref_text": "there are people around here to seem to like you very much people who are concerned with your happiness what is this",
        "whisper_wer_delete": 0,
        "whisper_wer_insert": 0,
        "whisper_wer_replace": 2,
        "whisper_wer_equal": 20,
        "whisper_cer_delete": 1,
        "whisper_cer_insert": 1,
        "whisper_cer_replace": 4,
        "whisper_cer_equal": 111,
    }

    error_rates = compute_error_rates(sample_data)
    print("Sample computation:")
    print(f"Input data: {sample_data}")
    print(f"Computed error rates: {error_rates}")

    # Manual verification:
    # Total ref words = 20 (correct) + 0 (deleted) + 2 (substituted) = 22
    # WER = (0 + 0 + 2) / 22 = 2/22 ≈ 0.0909
    print(f"Manual verification - WER should be 2/22 = {2/22:.4f}")


if __name__ == "__main__":
    # Uncomment the next line to test with sample data first
    # test_with_sample()

    # Run main analysis
    main()

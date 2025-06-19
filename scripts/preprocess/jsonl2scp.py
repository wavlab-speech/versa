#!/usr/bin/env python3
import json
import argparse
import re
from pathlib import Path


def clean_transcript(text):
    """
    Clean the transcript text for Kaldi format.

    Args:
        text (str): Raw transcript text

    Returns:
        str: Cleaned transcript
    """
    if not text:
        return ""

    # Replace newlines with spaces
    text = text.replace("\n", " ")

    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    # Convert to uppercase (common in Kaldi)
    text = text.upper()

    # Remove special characters that might cause issues (optional)
    # Uncomment the next line if you want to remove punctuation
    # text = re.sub(r'[^\w\s]', '', text)

    return text


def get_nested_value(data, key_path):
    """
    Get value from nested dictionary using dot notation.

    Args:
        data (dict): Dictionary to search
        key_path (str): Key path (e.g., 'level1.level2.key')

    Returns:
        Any: Value at the key path, or None if not found
    """
    keys = key_path.split(".")
    value = data

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return None


def convert_jsonl2scp_text(
    input_file,
    output_file="text",
    id_key="segment_index",
    transcript_key="local_scene",
    clean_text=True,
):
    """
    Convert JSONL file to Kaldi text format.

    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output text file
        id_key (str): JSON key for utterance ID (supports dot notation for nested keys)
        transcript_key (str): JSON key for transcript text (supports dot notation)
        clean_text (bool): Whether to clean the transcript text
    """
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: Input file '{input_file}' does not exist")
        return

    entries = []

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # Extract required fields using configurable keys
                    utterance_id = get_nested_value(data, id_key)
                    transcript_text = get_nested_value(data, transcript_key)

                    if not utterance_id:
                        print(
                            f"Warning: Missing or empty '{id_key}' on line {line_num}"
                        )
                        continue

                    if not transcript_text:
                        print(
                            f"Warning: Missing or empty '{transcript_key}' for ID '{utterance_id}'"
                        )
                        continue

                    # Convert to string if not already
                    utterance_id = str(utterance_id)
                    transcript_text = str(transcript_text)

                    # Clean transcript if requested
                    if clean_text:
                        transcript = clean_transcript(transcript_text)
                    else:
                        transcript = transcript_text.replace("\n", " ").strip()

                    if transcript:  # Only add non-empty transcripts
                        entries.append((utterance_id, transcript))

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue

    except Exception as e:
        print(f"Error reading file '{input_file}': {e}")
        return

    if not entries:
        print("No valid entries found to convert")
        return

    # Sort entries by utterance_id for consistent output
    entries.sort(key=lambda x: x[0])

    # Write Kaldi text format
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for utterance_id, transcript in entries:
                f.write(f"{utterance_id} {transcript}\n")

        print(f"Successfully converted {len(entries)} entries to '{output_file}'")
        print(f"Using ID key: '{id_key}', Transcript key: '{transcript_key}'")

        # Show first few examples
        print("\nFirst few entries:")
        for i, (utterance_id, transcript) in enumerate(entries[:3]):
            print(
                f"  {utterance_id} {transcript[:60]}{'...' if len(transcript) > 60 else ''}"
            )

    except Exception as e:
        print(f"Error writing output file '{output_file}': {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL to Kaldi text format with configurable keys",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default usage (segment_index + local_scene)
  python jsonl2scp.py input.jsonl
  
  # Custom keys
  python jsonl2scp.py input.jsonl --id-key "utterance_id" --transcript-key "text"
  
  # Nested keys using dot notation
  python jsonl2scp.py input.jsonl --id-key "metadata.id" --transcript-key "content.transcript"
  
  # Multiple configurations
  python jsonl2scp.py input.jsonl -i "speaker_id" -t "dialogue" -o "speaker_text.txt"
        """,
    )

    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument(
        "-o", "--output", default="text", help="Output text file (default: text)"
    )
    parser.add_argument(
        "-i",
        "--id-key",
        default="scene_index",
        help="JSON key for utterance ID (default: scene_index)",
    )
    parser.add_argument(
        "-t",
        "--transcript-key",
        default="local_scene",
        help="JSON key for transcript text (default: local_scene)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip text cleaning (keep original formatting)",
    )

    args = parser.parse_args()

    convert_jsonl2scp_text(
        args.input, args.output, args.id_key, args.transcript_key, not args.no_clean
    )


if __name__ == "__main__":
    main()

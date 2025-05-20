#!/usr/bin/env python3

import os
import sys
import argparse


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Create a wav.scp file from audio files in a folder"
    )
    parser.add_argument("folder_path", help="Path to the folder containing audio files")
    parser.add_argument(
        "output_scp",
        nargs="?",
        default="wav.scp",
        help="Output SCP file path (default: wav.scp)",
    )
    args = parser.parse_args()

    folder_path = args.folder_path
    output_scp = args.output_scp

    # Check if folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found or is not a directory.")
        sys.exit(1)

    # Create or clear the output file
    with open(output_scp, "w") as f:
        pass

    # Find all wav and flac files and create entries
    with open(output_scp, "a") as f:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".flac") or file.endswith(".wav"):
                    # Get utterance ID (filename without extension)
                    utterance_id = os.path.splitext(file)[0]

                    # Get absolute path
                    audio_path = os.path.abspath(os.path.join(root, file))

                    # Write entry to scp file
                    f.write(f"{utterance_id} {audio_path}\n")

    print(
        f"Created wav.scp file '{output_scp}' with entries for all WAV and FLAC files in '{folder_path}'"
    )


if __name__ == "__main__":
    main()

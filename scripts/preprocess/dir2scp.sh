#!/bin/bash

# Check if folder path is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 folder_path [output_scp_file]"
    exit 1
fi

folder_path="$1"
output_scp="${2:-wav.scp}"  # Default to wav.scp if not specified

# Check if the folder exists
if [ ! -d "$folder_path" ]; then
    echo "Error: Folder '$folder_path' not found or is not a directory."
    exit 1
fi

# Clear or create the output SCP file
> "$output_scp"

# Find all wav files in the folder and create the wav.scp entries
find "$folder_path" -type f \( -name "*.flac" -o -name "*.wav" \) | while read -r audio_file; do
    # Get the basename without extension to use as utterance ID
    filename=$(basename "$wav_file")
    utterance_id="${filename%.*}"
    
    # Get the absolute path
    absolute_path=$(readlink -f "$wav_file")
    
    # Write to the scp file: utterance_id absolute_path
    echo "$utterance_id $absolute_path" >> "$output_scp"
done

echo "Created wav.scp file '$output_scp' with entries for all WAV files in '$folder_path'"

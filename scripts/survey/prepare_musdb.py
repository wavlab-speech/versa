import os
import argparse
import soundfile as sf
import numpy as np
from glob import glob
from tqdm import tqdm
import librosa


def split_audio(file_path, chunk_length, output_dir, parent_folder):
    """Split the audio file into non-overlapping chunks and save them."""
    # Load the audio file using soundfile
    print(f"Processing: {file_path}")
    waveform, sample_rate = librosa.load(file_path, mono=True)
    total_length = len(waveform) / sample_rate  # Total length in seconds

    # Calculate the number of chunks
    num_chunks = int(np.ceil(total_length / chunk_length))

    # Split and save each chunk
    for i in range(num_chunks):
        start_time = i * chunk_length
        end_time = min((i + 1) * chunk_length, total_length)

        # Find the sample indices for the chunk
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # Extract the chunk waveform
        chunk_waveform = waveform[start_sample:end_sample]

        # Build the output file path
        output_filename = f"{parent_folder}_{start_time}_{end_time}.wav"
        output_file_path = os.path.join(output_dir, output_filename)

        # Save the chunk to the output directory
        sf.write(output_file_path, chunk_waveform, sample_rate)
    print(f"Saved chunks for : {parent_folder}")


def process_directory(main_directory, chunk_length, output_dir):
    """Process all subdirectories in the main directory and split 'mix.wav' files."""
    for mix_wav_path in tqdm(
        glob(os.path.join(main_directory, "**/mixture.wav"), recursive=True)
    ):
        # Get the parent folder name
        parent_folder = mix_wav_path.split("/")[-2].replace(" ", "_")
        # Split the 'mix.wav' into chunks
        split_audio(mix_wav_path, chunk_length, output_dir, parent_folder)


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Split 'mix.wav' files into chunks.")
    parser.add_argument(
        "--main_directory",
        type=str,
        help="Path to the main directory containing subfolders.",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory where the chunks will be saved."
    )
    parser.add_argument(
        "--chunk_length", type=float, help="Length of each chunk in seconds."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Process the directory
    process_directory(args.main_directory, args.chunk_length, args.output_dir)


if __name__ == "__main__":
    main()

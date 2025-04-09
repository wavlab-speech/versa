from huggingface_hub import HfApi
import os, argparse
import tarfile

# Initialize the Hugging Face API
api = HfApi()


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Split 'mix.wav' files into chunks.")
    parser.add_argument(
        "--output_dir", type=str, help="Where AudioSet files will be saved."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Define the repository details
    repo_id = "agkphysics/AudioSet"  # Dataset repository
    repo_path = "data"
    local_save_dir = args.output_dir
    audio_dump = os.path.join(local_save_dir, "audio_files")

    # Create the local directory if it doesn't exist
    os.makedirs(local_save_dir, exist_ok=True)

    # List files in the dataset repository (specify repo_type="dataset")
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    # Filter files matching the desired pattern and path
    files_to_download = [
        file
        for file in repo_files
        if file.startswith(repo_path) and file.endswith(".tar") and "eval" in file
    ]

    print(f"Files to download: {files_to_download}")

    # Base URL for the dataset files
    base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/"

    # Download each file
    for file_path in files_to_download:
        file_url = base_url + file_path
        local_file_path = os.path.join(local_save_dir, os.path.basename(file_path))
        if os.path.exists(local_file_path):
            print(f"File {local_file_path} already exists, skipping download.")
        else:
            print(f"Downloading {file_url} to {local_file_path}...")

            # Download the file manually using requests
            import requests

            response = requests.get(file_url, stream=True)
            if response.status_code == 200:
                with open(local_file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
            else:
                print(
                    f"Failed to download {file_url}, status code: {response.status_code}"
                )

        # Extract the .tar file
        print(f"Extracting {local_file_path} to {audio_dump}...")
        try:
            with tarfile.open(local_file_path, "r") as tar:
                tar.extractall(
                    path=audio_dump,
                    members=[
                        member
                        for member in tar.getmembers()
                        if member.isfile()  # Only extract files, skip directories
                    ],
                )
        except Exception as e:
            print(f"Error extracting {local_file_path}: {e}")

        # Delete the .tar file after successful extraction
        print(f"Deleting {local_file_path}...")
        try:
            os.remove(local_file_path)
        except Exception as e:
            print(f"Error deleting {local_file_path}: {e}")

    print("All files downloaded and extracted.")


if __name__ == "__main__":
    main()

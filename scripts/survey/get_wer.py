import argparse


def calculate_average_wer(file_path):
    total_wer = 0
    line_count = 0

    total_delete, total_insert, total_replace, total_length = 0, 0, 0, 0
    with open(file_path, "r") as file:
        for line in file:
            data = eval(line.strip())

            if data.get("whisper_wer_equal", None):
                wer_delete = data["whisper_wer_delete"]
                wer_insert = data["whisper_wer_insert"]
                wer_replace = data["whisper_wer_replace"]
                ref_text_length = data["whisper_wer_equal"] + wer_delete + wer_replace

            else:
                continue
                wer_delete = data["espnet_wer_delete"]
                wer_insert = data["espnet_wer_insert"]
                wer_replace = data["espnet_wer_replace"]
                ref_text_length = data["espnet_wer_equal"] + wer_delete + wer_replace

            if ref_text_length > 0:
                total_delete += wer_delete
                total_insert += wer_insert
                total_replace += wer_replace
                total_length += ref_text_length
    print(
        "wer: {}".format((total_delete + total_insert + total_replace) / total_length)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate average WER from a text file."
    )
    parser.add_argument("--file_path", type=str, help="Path to the input text file.")
    args = parser.parse_args()
    calculate_average_wer(args.file_path)

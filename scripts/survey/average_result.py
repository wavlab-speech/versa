import argparse
import re
import numpy as np
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description="Calculate average values of metrics from a text file."
    )
    parser.add_argument("--file_path", type=str, help="Path to the input text file")
    args = parser.parse_args()

    metrics = defaultdict(list)

    with open(args.file_path, "r") as f:
        for line in f:
            if line.strip():
                matches = re.findall(r"'(\w+)': ([\d.]+|inf)", line)
                for key, value in matches:
                    if value != "inf":
                        metrics[key].append(float(value))

    for key, values in metrics.items():
        mean_value = np.mean(values)
        print(f"{key} avg: {mean_value}")


if __name__ == "__main__":
    main()

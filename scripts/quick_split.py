import os
import math


def split_file(input_file, num_parts=5):
    """
    Splits a file into a specified number of parts with approximately equal line counts.

    Args:
        input_file (str): Path to the input file
        num_parts (int): Number of parts to split the file into (default: 5)
    """
    # Get the total number of lines in the file
    with open(input_file, "r") as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"Total lines in file: {total_lines}")

    # Calculate lines per part (rounded up to ensure we cover all lines)
    lines_per_part = math.ceil(total_lines / num_parts)
    print(f"Approximate lines per part: {lines_per_part}")

    # Create base name for output files
    base_name, ext = os.path.splitext(input_file)

    # Split and write the files
    for i in range(num_parts):
        start_line = i * lines_per_part
        end_line = min((i + 1) * lines_per_part, total_lines)

        # Skip if this part would be empty
        if start_line >= total_lines:
            break

        output_file = f"{base_name}_part{i+1}{ext}"

        with open(output_file, "w") as f:
            f.writelines(lines[start_line:end_line])

        print(f"Created {output_file} with {end_line - start_line} lines")


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python split_file.py <filename> [number_of_parts]")
        sys.exit(1)

    filename = sys.argv[1]
    num_parts = 5  # Default

    if len(sys.argv) > 2:
        try:
            num_parts = int(sys.argv[2])
        except ValueError:
            print("Number of parts must be an integer")
            sys.exit(1)

    split_file(filename, num_parts)

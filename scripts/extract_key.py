import json

def extract_keys_from_jsonl(input_file, output_file):
    """
    Extract only the 'key' values from a JSONL file and save them to a text file.
    
    Args:
        input_file (str): Path to the input JSONL file
        output_file (str): Path to the output text file
    """
    keys = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    data = json.loads(line)
                    ref_length = len(data["ref_text"].split())
                    if ref_length < 16:
                        continue
                    if 'key' in data:
                        start, end = data["key"].split("_")[-2:]
                        if float(end) - float(start) > 30 or float(end) - float(start) < 10:
                            continue
                        keys.append(data['key'])
                    else:
                        print(f"Warning: No 'key' field found in line {line_num}")
                
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    continue
        
        # Write keys to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for key in keys:
                f.write(key + '\n')
        
        print(f"Successfully extracted {len(keys)} keys to '{output_file}'")
        return keys
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return []
    except Exception as e:
        print(f"Error processing file: {e}")
        return []

# Example usage:
if __name__ == "__main__":
    # Replace 'input.jsonl' with your actual input file path
    # Replace 'keys_only.txt' with your desired output file path
    input_filename = "filtered_results/deletion_error_lt_0.05.jsonl"
    output_filename = "deletion_error_lt0.05_300h.txt"
    
    extracted_keys = extract_keys_from_jsonl(input_filename, output_filename)
    
    # Optional: Print the first few keys as a preview
    if extracted_keys:
        print(f"\nFirst few keys extracted:")
        for i, key in enumerate(extracted_keys[:5]):
            print(f"{i+1}: {key}")
        if len(extracted_keys) > 5:
            print(f"... and {len(extracted_keys) - 5} more keys")

# Alternative one-liner approach using list comprehension:
def extract_keys_one_liner(input_file, output_file):
    """
    One-liner version to extract keys from JSONL file
    """
    try:
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            keys = [json.loads(line)['key'] for line in f_in if line.strip()]
            f_out.write('\n'.join(keys))
        print(f"Extracted {len(keys)} keys using one-liner approach")
    except Exception as e:
        print(f"Error: {e}")

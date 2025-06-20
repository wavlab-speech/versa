#!/usr/bin/env python3

"""
Qwen2-Audio JSONL Output Standardizer with Batch Inference

This script standardizes outputs from Qwen2-Audio that are stored in JSONL format
using batch inference with the Transformers library to improve performance.
"""

import ast
import json
import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Union, Any, Tuple
import re
from tqdm import tqdm
import torch
from collections import defaultdict

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except ImportError:
    logging.warning(
        "Transformers not found. Please install with: pip install transformers"
    )
    AutoTokenizer, AutoModelForCausalLM, pipeline = None, None, None

# Define the expected formats/categories for each metric
EXPECTED_FORMATS = {
    # Speaker Characteristics
    "qwen_speaker_count": {
        "type": "number",
        "description": "Number of distinct speakers (1-10)",
    },
    "qwen_speaker_gender": {
        "type": "category",
        "categories": [
            "Male",
            "Female",
            "Non-binary/unclear",
            "Multiple speakers with mixed genders",
        ],
    },
    "qwen_speaker_age": {
        "type": "category",
        "categories": ["Child", "Teen", "Young adult", "Middle-aged adult", "Senior"],
    },
    "qwen_speech_impairment": {
        "type": "category",
        "categories": [
            "No apparent impairment",
            "Stuttering/disfluency",
            "Articulation disorder",
            "Voice disorder",
            "Fluency disorder",
            "Foreign accent",
            "Dysarthria",
            "Apraxia",
            "Other impairment",
        ],
    },
    # Voice Properties
    "qwen_pitch_range": {
        "type": "category",
        "categories": ["Wide range", "Moderate range", "Narrow range", "Monotone"],
    },
    "qwen_voice_pitch": {
        "type": "category",
        "categories": ["Very high", "High", "Medium", "Low", "Very low"],
    },
    "qwen_voice_type": {
        "type": "category",
        "categories": [
            "Clear",
            "Breathy",
            "Creaky/vocal fry",
            "Hoarse",
            "Nasal",
            "Pressed/tense",
            "Resonant",
            "Whispered",
            "Tremulous",
        ],
    },
    "qwen_speech_volume_level": {
        "type": "category",
        "categories": [
            "Very quiet",
            "Quiet",
            "Moderate",
            "Loud",
            "Very loud",
            "Variable",
        ],
    },
    # Speech Content
    "qwen_language": {
        "type": "category",
        "categories": [
            "English",
            "Spanish",
            "Mandarin Chinese",
            "Hindi",
            "Arabic",
            "French",
            "Russian",
            "Portuguese",
            "German",
            "Japanese",
            "Other",
        ],
    },
    "qwen_speech_register": {
        "type": "category",
        "categories": [
            "Formal register",
            "Standard register",
            "Consultative register",
            "Casual register",
            "Intimate register",
            "Technical register",
            "Slang register",
        ],
    },
    "qwen_vocabulary_complexity": {
        "type": "category",
        "categories": ["Basic", "General", "Advanced", "Technical", "Academic"],
    },
    "qwen_speech_purpose": {
        "type": "category",
        "categories": [
            "Informative",
            "Persuasive",
            "Entertainment",
            "Narrative",
            "Conversational",
            "Instructional",
            "Emotional expression",
        ],
    },
    # Speech Delivery
    "qwen_speech_emotion": {
        "type": "category",
        "categories": [
            "Neutral",
            "Happy",
            "Sad",
            "Angry",
            "Fearful",
            "Surprised",
            "Disgusted",
            "Other",
        ],
    },
    "qwen_speech_clarity": {
        "type": "category",
        "categories": [
            "High clarity",
            "Medium clarity",
            "Low clarity",
            "Very low clarity",
        ],
    },
    "qwen_speech_rate": {
        "type": "category",
        "categories": ["Very slow", "Slow", "Medium", "Fast", "Very fast"],
    },
    "qwen_speaking_style": {
        "type": "category",
        "categories": [
            "Formal",
            "Professional",
            "Casual/conversational",
            "Animated/enthusiastic",
            "Deliberate",
            "Dramatic",
            "Authoritative",
            "Hesitant",
        ],
    },
    "qwen_laughter_crying": {
        "type": "category",
        "categories": [
            "No laughter or crying",
            "Contains laughter",
            "Contains crying",
            "Contains both",
            "Contains other emotional sounds",
            "Contains multiple emotional vocalizations",
        ],
    },
    # Interaction Patterns
    "qwen_overlapping_speech": {
        "type": "category",
        "categories": [
            "No overlap",
            "Minimal overlap",
            "Moderate overlap",
            "Significant overlap",
            "Constant overlap",
        ],
    },
    # Recording Environment
    "qwen_speech_background_environment": {
        "type": "category",
        "categories": [
            "Quiet indoor",
            "Noisy indoor",
            "Outdoor urban",
            "Outdoor natural",
            "Event/crowd",
            "Music background",
            "Multiple environments",
        ],
    },
    "qwen_recording_quality": {
        "type": "category",
        "categories": ["Professional", "Good", "Fair", "Poor", "Very poor"],
    },
    "qwen_channel_type": {
        "type": "category",
        "categories": [
            "Professional microphone",
            "Consumer microphone",
            "Smartphone",
            "Telephone/VoIP",
            "Webcam/computer mic",
            "Headset microphone",
            "Distant microphone",
            "Radio/broadcast",
            "Surveillance/hidden mic",
        ],
    },
}

# Translation dictionary for common non-English terms
TRANSLATION_DICT = {
    # Chinese to English translations for common speech properties
    "慢": "Slow",
    "中等": "Medium",
    "快": "Fast",
    "很快": "Very fast",
    "很慢": "Very slow",
    "男": "Male",
    "女": "Female",
    "高": "High",
    "低": "Low",
    "中": "Medium",
    "很高": "Very high",
    "很低": "Very low",
    "明确": "Clear",
    "清晰": "High clarity",
    "不清晰": "Low clarity",
    "普通": "Medium clarity",
    "一般": "Medium clarity",
    "专业": "Professional",
    "良好": "Good",
    "一个": "1",
    "两个": "2",
    "三个": "3",
    "四个": "4",
    "五个": "5",
    # Add more translations as needed
}


def json_processor(single_quoted_json):
    """Convert single-quoted JSON to double-quoted JSON."""
    try:
        # Method 1: Using ast.literal_eval to parse the Python dict-like syntax
        # This handles apostrophes well because it's a Python interpreter
        parsed_data = ast.literal_eval(single_quoted_json)

        # Use json.dumps to convert to proper JSON with double quotes
        return json.dumps(parsed_data)

    except (SyntaxError, ValueError) as e:
        print(f"Primary method failed: {e}")

        # Method 2: Regex-based fallback approach for more complex cases
        try:
            # Replace property names in single quotes with double quotes
            result = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', single_quoted_json)

            # Temporarily mark apostrophes in text with a unique marker
            result = re.sub(r"(\w)'(\w)", r"\1‖\2", result)

            # Replace value strings in single quotes with double quotes
            result = re.sub(r":(\s*)'([^']*)'", r':\1"\2"', result)

            # Restore apostrophes from the unique marker
            result = result.replace("‖", "'")

            # Validate by attempting to parse
            json.loads(result)

            return result

        except Exception as fallback_error:
            print(f"Fallback method also failed: {fallback_error}")

            # Method 3: Even more advanced fallback for complex JSON
            # This approach tries to handle nested structures better
            # First normalize all escaped quotes
            text = single_quoted_json.replace("\\'", "___APOSTROPHE___")

            # Replace all standalone single quotes with double quotes
            text = text.replace("'", '"')

            # Restore all apostrophes
            text = text.replace("___APOSTROPHE___", "'")

            # Validate
            json.loads(text)

            # Format it nicely
            parsed = json.loads(text)

            # NOTE(jiatong): if stil not fixed, raise corresponding error
            return json.dumps(parsed)


class BatchInferenceStandardizer:
    """
    A class for standardizing Qwen2-Audio outputs using batch inference.
    """

    def __init__(
        self, model_name="Qwen/Qwen2.5-7B-Instruct-1M", device="auto", batch_size=8
    ):
        """
        Initialize the standardizer with a text-only LLM.

        Args:
            model_name (str): HuggingFace model name for the text-only LLM
            device (str): Device to run the model on ("cpu", "cuda", "auto")
            batch_size (int): Batch size for inference
        """
        self.expected_formats = EXPECTED_FORMATS
        self.translation_dict = TRANSLATION_DICT
        self.llm_available = False
        self.batch_size = batch_size

        # Try to initialize the LLM
        if AutoTokenizer is not None and AutoModelForCausalLM is not None:
            try:
                logging.info(f"Initializing language model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    low_cpu_mem_usage=True,
                    torch_dtype="auto",
                    device_map=device,
                )

                # Use custom generation instead of pipeline for batch inference
                self.llm_available = True
                logging.info(f"LLM initialized successfully. Batch size: {batch_size}")
            except Exception as e:
                logging.warning(f"Failed to initialize LLM: {e}")
                self.llm_available = False

    def _translate_non_english(self, text):
        """
        Translate non-English terms to English.

        Args:
            text (str): Text that may contain non-English terms

        Returns:
            str: Text with non-English terms translated
        """
        if not isinstance(text, str):
            return text

        # For each non-English term in our dictionary
        for non_english, english in self.translation_dict.items():
            if non_english in text:
                # If it's a complete match, return the translation
                if text.strip() == non_english:
                    return english
                # Otherwise replace the term within the text
                text = text.replace(non_english, english)

        return text

    def _generate_prompt(self, metric_name, raw_output):
        """
        Generate a prompt for the text-only LLM to standardize the output.

        Args:
            metric_name (str): Name of the metric being standardized
            raw_output (str): Raw output from Qwen2-Audio

        Returns:
            str: Prompt for the text-only LLM
        """
        if metric_name not in self.expected_formats:
            return f"""
            I need to classify this speech property: "{raw_output}"
            Please extract the most relevant category mentioned.
            Provide only the category name without explanations. \n\n
            Here is the result: 
            """

        format_info = self.expected_formats[metric_name]

        if format_info["type"] == "number":
            return f"""
            Extract the number of speakers from this response: "{raw_output}"
            If multiple numbers are mentioned, choose the most definitive one.
            Provide only the number (1-10) without any other text. \n\n
            Here is the result:
            """

        elif format_info["type"] == "multi-category":
            categories = ", ".join([f'"{cat}"' for cat in format_info["categories"]])
            return f"""
            Classify this response into one or more of these categories: {categories}
            Response: "{raw_output}"
            Extract only the most relevant categories from the valid list.
            If 'Other' is chosen, specify what it is if possible (e.g., "Other: Polish").
            Provide only the category name(s) without explanations, separated by commas if multiple.
            \n\n
            Here is the result:
            """

        else:  # "category"
            categories = ", ".join([f'"{cat}"' for cat in format_info["categories"]])
            return f"""
            Classify this response into exactly one of these categories: {categories}
            Response: "{raw_output}"
            Find the closest match from the valid categories.
            Provide only the exact category name without explanations. \n\n
            Here is the result:
            """

    def _process_llm_output(self, metric_name, llm_output):
        """
        Process the output from the text-only LLM.

        Args:
            metric_name (str): Name of the metric
            llm_output (str): Output from the text-only LLM

        Returns:
            Union[str, int, List[str]]: Standardized output
        """
        # Clean up the output
        clean_output = llm_output.strip()

        # Remove common prefixes that LLMs tend to add
        prefixes_to_remove = [
            "The category is",
            "Category:",
            "Answer:",
            "Output:",
            "The speaker count is",
            "Number of speakers:",
            "The number is",
            "Classification:",
            "The classification is",
        ]

        for prefix in prefixes_to_remove:
            if clean_output.startswith(prefix):
                clean_output = clean_output[len(prefix) :].strip()

        # Process based on expected type
        if metric_name in self.expected_formats:
            format_type = self.expected_formats[metric_name]["type"]

            if format_type == "number":
                # Extract just the number
                match = re.search(r"\d+", clean_output)
                if match:
                    return int(match.group())
                return 1  # Default to 1 if no number found

            elif format_type == "multi-category":
                # Split by commas and clean up
                categories = [cat.strip() for cat in clean_output.split(",")]
                return categories

        # For all other cases, return the cleaned string
        return clean_output

    def _rules_based_standardize(self, metric_name, raw_output):
        """
        Standardize output using simple rules-based approach.

        Args:
            metric_name (str): Name of the metric
            raw_output (str): Raw output from Qwen2-Audio

        Returns:
            Union[str, int, List[str]]: Standardized output based on simple rules
        """
        # First translate any non-English terms
        translated_output = self._translate_non_english(raw_output)

        # Only process if we have format info
        if metric_name not in self.expected_formats:
            return translated_output

        format_info = self.expected_formats[metric_name]

        # Handle special case for language
        if metric_name == "qwen_language":
            for cat in format_info["categories"]:
                if cat.lower() in translated_output.lower():
                    return cat
            # Extract language name from common phrases
            match = re.search(
                r"(language|spoken) (?:is|in) ([\w\s]+)",
                translated_output,
                re.IGNORECASE,
            )
            if match:
                language = match.group(2).strip()
                # Check if this matches one of our categories
                for cat in format_info["categories"]:
                    if cat.lower() in language.lower():
                        return cat
                return language
            return "English"  # Default

        # Handle special case for overlapping speech
        if metric_name == "qwen_overlapping_speech":
            if "no overlap" in translated_output.lower():
                return "No overlap"
            for cat in format_info["categories"]:
                if cat.lower() in translated_output.lower():
                    return cat
            return "No overlap"  # Default

        # Handle numeric values (speaker count)
        if format_info["type"] == "number":
            # First check for digit
            match = re.search(r"\d+", translated_output)
            if match:
                num = int(match.group())
                return min(max(1, num), 10)  # Ensure between 1-10

            # Check for number words
            number_words = {
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
                "six": 6,
                "seven": 7,
                "eight": 8,
                "nine": 9,
                "ten": 10,
            }
            for word, num in number_words.items():
                if word in translated_output.lower():
                    return num

            return 1  # Default to 1

        # For regular categories
        if format_info["type"] == "category":
            # First check exact match
            for category in format_info["categories"]:
                if category.lower() in translated_output.lower():
                    return category

            # If no exact match, try to find the best match
            best_match = None
            best_score = 0

            for category in format_info["categories"]:
                # Simple matching score based on word presence
                category_lower = category.lower()
                words = category_lower.split()

                # Count how many words from the category appear in the output
                score = sum(1 for word in words if word in translated_output.lower())

                if score > best_score:
                    best_score = score
                    best_match = category

            return best_match if best_match else format_info["categories"][0]

        return translated_output

    def batch_generate(self, prompts):
        """
        Generate responses for a batch of prompts.

        Args:
            prompts (List[str]): List of prompts

        Returns:
            List[str]: List of generated responses
        """
        if not self.llm_available:
            return [""] * len(prompts)

        try:
            # Tokenize inputs
            inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(
                self.model.device
            )

            # Generate outputs
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,  # Short responses sufficient for classification
                    do_sample=False,  # Deterministic for consistent classification
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Extract generated text
            generated_texts = []
            for i, output in enumerate(outputs):
                # Skip input tokens to get only the generated part
                input_length = inputs.input_ids[i].shape[0]
                generated_tokens = output[input_length:]
                generated_text = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                generated_texts.append(generated_text)

            return generated_texts

        except Exception as e:
            logging.error(f"Error in batch generation: {e}")
            return [""] * len(prompts)

    def batch_standardize(self, items):
        """
        Standardize a batch of metrics using LLM.

        Args:
            items (List[Tuple[str, str, str]]): List of (id, metric_name, raw_output) tuples

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of standardized outputs
        """
        if not self.llm_available or not items:
            # Fall back to rules-based approach for each item
            return {
                item_id: self._rules_based_standardize(metric_name, raw_output)
                for item_id, metric_name, raw_output in items
            }

        # First translate non-English terms
        translated_items = [
            (item_id, metric_name, self._translate_non_english(raw_output))
            for item_id, metric_name, raw_output in items
        ]

        # Generate prompts for each item
        prompts = [
            self._generate_prompt(metric_name, raw_output)
            for _, metric_name, raw_output in translated_items
        ]

        # Process in batches
        all_responses = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]
            batch_responses = self.batch_generate(batch_prompts)
            all_responses.extend(batch_responses)

        # Process LLM outputs
        results = {}
        for i, (item_id, metric_name, _) in enumerate(translated_items):
            llm_output = all_responses[i]
            if llm_output:
                results[item_id] = self._process_llm_output(metric_name, llm_output)
            else:
                # Fall back to rules-based approach if LLM failed
                results[item_id] = self._rules_based_standardize(
                    metric_name, translated_items[i][2]
                )

        return results

    def standardize_jsonl_batch(self, json_objects):
        """
        Standardize a batch of JSON objects.

        Args:
            json_objects (List[Dict]): List of JSON objects to standardize

        Returns:
            List[Dict]: List of standardized JSON objects
        """
        if not json_objects:
            return []

        # Group metrics from all objects for batch processing
        batch_items = []
        item_map = {}  # Maps (object_idx, metric_name) to batch_item_idx

        for obj_idx, obj in enumerate(json_objects):
            # Keep track of original values
            obj["original_values"] = {}

            # Process each metric in the object
            for key, value in list(obj.items()):
                # Skip non-qwen keys
                if not key.startswith("qwen_"):
                    continue

                # Store original value
                obj["original_values"][key] = value

                # Create a unique ID for this item
                item_id = f"{obj_idx}_{key}"
                item_map[(obj_idx, key)] = len(batch_items)

                # Add to batch items
                batch_items.append((item_id, key, value))

        # Batch standardize all metrics
        standardized_values = self.batch_standardize(batch_items)

        # Update objects with standardized values
        for obj_idx, obj in enumerate(json_objects):
            for key in list(obj.keys()):
                if key.startswith("qwen_"):
                    item_id = f"{obj_idx}_{key}"
                    if item_id in standardized_values:
                        obj[key] = standardized_values[item_id]

        return json_objects

    def process_jsonl_file(self, input_path, output_path, batch_size=100):
        """
        Process a JSONL file containing Qwen2-Audio outputs.

        Args:
            input_path (str): Path to input JSONL file
            output_path (str): Path to output JSONL file
            batch_size (int): Number of JSON objects to process at once

        Returns:
            dict: Processing statistics
        """
        # Count lines in file for progress tracking
        with open(input_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        processed = 0
        successful = 0
        failed = 0

        # Process file in batches
        with open(input_path, "r", encoding="utf-8") as infile, open(
            output_path, "w", encoding="utf-8"
        ) as outfile:

            # Read and process the file in batches
            batch = []
            batch_line_numbers = []

            for line_num, line in enumerate(
                tqdm(infile, total=total_lines, desc="Processing JSONL")
            ):
                try:
                    # Parse JSON object
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(json_processor(line))
                    processed += 1

                    # Add to current batch
                    batch.append(data)
                    batch_line_numbers.append(line_num)

                    # Process batch if it reaches the batch size
                    if len(batch) >= batch_size:
                        try:
                            standardized_batch = self.standardize_jsonl_batch(batch)

                            # Write standardized objects to output file
                            for obj in standardized_batch:
                                outfile.write(
                                    json.dumps(obj, ensure_ascii=False) + "\n"
                                )
                                outfile.flush()
                                successful += 1

                        except Exception as e:
                            # Write original objects on batch failure
                            logging.error(f"Error processing batch: {e}")
                            for original_obj in batch:
                                outfile.write(
                                    json.dumps(original_obj, ensure_ascii=False) + "\n"
                                )
                                outfile.flush()
                                failed += 1

                        # Reset batch
                        batch = []
                        batch_line_numbers = []

                except Exception as e:
                    failed += 1
                    logging.error(f"Error parsing line {line_num+1}: {e}")
                    # Write original line to avoid data loss
                    if line:
                        outfile.write(line + "\n")

            # Process remaining batch
            if batch:
                try:
                    standardized_batch = self.standardize_jsonl_batch(batch)

                    # Write standardized objects to output file
                    for obj in standardized_batch:
                        outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        successful += 1

                except Exception as e:
                    # Write original objects on batch failure
                    logging.error(f"Error processing final batch: {e}")
                    for original_obj in batch:
                        outfile.write(
                            json.dumps(original_obj, ensure_ascii=False) + "\n"
                        )
                        failed += 1

        return {
            "total_lines": total_lines,
            "processed": processed,
            "successful": successful,
            "failed": failed,
        }


def process_directory(input_dir, output_dir, file_pattern="*.jsonl", batch_size=100):
    """
    Process all JSONL files in a directory.

    Args:
        input_dir (str): Input directory containing JSONL files
        output_dir (str): Output directory
        file_pattern (str): File pattern to match
        batch_size (int): Number of JSON objects to process at once

    Returns:
        dict: Processing statistics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create standardizer
    standardizer = BatchInferenceStandardizer(batch_size=batch_size)

    # Find all matching files
    input_path = Path(input_dir)
    jsonl_files = list(input_path.glob(file_pattern))

    if not jsonl_files:
        logging.warning(
            f"No files found in {input_dir} matching pattern '{file_pattern}'"
        )
        return {"error": f"No files found matching pattern '{file_pattern}'"}

    # Process each file
    results = {}
    totals = {
        "total_files": len(jsonl_files),
        "total_lines": 0,
        "processed": 0,
        "successful": 0,
        "failed": 0,
    }

    for jsonl_file in jsonl_files:
        # Generate output path
        rel_path = (
            jsonl_file.relative_to(input_path)
            if jsonl_file.is_relative_to(input_path)
            else jsonl_file.name
        )
        output_path = Path(output_dir) / rel_path

        # Ensure output directory exists
        os.makedirs(output_path.parent, exist_ok=True)

        # Process the file
        logging.info(f"Processing {jsonl_file}")
        try:
            stats = standardizer.process_jsonl_file(
                str(jsonl_file), str(output_path), batch_size
            )

            # Add to results
            results[str(jsonl_file)] = {
                "output_file": str(output_path),
                "success": True,
                "stats": stats,
            }

            # Update totals
            totals["total_lines"] += stats["total_lines"]
            totals["processed"] += stats["processed"]
            totals["successful"] += stats["successful"]
            totals["failed"] += stats["failed"]

        except Exception as e:
            logging.error(f"Error processing file {jsonl_file}: {e}")
            results[str(jsonl_file)] = {
                "output_file": str(output_path),
                "success": False,
                "error": str(e),
            }

    # Save summary
    summary_path = Path(output_dir) / "processing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {"totals": totals, "results": results}, f, indent=2, ensure_ascii=False
        )

    return totals


def main():
    parser = argparse.ArgumentParser(
        description="Standardize Qwen2-Audio JSONL outputs with batch inference"
    )
    parser.add_argument("input", help="Input JSONL file or directory")
    parser.add_argument(
        "--output",
        "-o",
        help="Output file or directory (default: adds '_standardized' to input name)",
    )
    parser.add_argument(
        "--pattern",
        default="*.jsonl",
        help="File pattern for directory mode (default: *.jsonl)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="Qwen/Qwen2.5-7B-Instruct-1M",
        help="HuggingFace model to use (default: Qwen/Qwen2.5-7B-Instruct-1M)",
    )
    parser.add_argument(
        "--device",
        "-d",
        default="auto",
        help="Device to run on ('cpu', 'cuda', 'auto') (default: auto)",
    )
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Setup logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            (
                logging.FileHandler(args.log_file)
                if args.log_file
                else logging.NullHandler()
            ),
        ],
    )

    input_path = Path(args.input)

    # Set default output path if not provided
    if args.output is None:
        if input_path.is_file():
            output_path = input_path.with_stem(f"{input_path.stem}_standardized")
        else:
            output_path = input_path.parent / f"{input_path.name}_standardized"
    else:
        output_path = Path(args.output)

    # Process based on input type
    if input_path.is_file():
        logging.info(f"Processing single JSONL file: {input_path}")
        standardizer = BatchInferenceStandardizer(
            model_name=args.model, device=args.device, batch_size=args.batch_size
        )
        stats = standardizer.process_jsonl_file(
            str(input_path), str(output_path), args.batch_size
        )
        logging.info(
            f"Processing complete: {stats['successful']}/{stats['total_lines']} lines processed successfully"
        )

    elif input_path.is_dir():
        logging.info(f"Processing directory: {input_path}")
        totals = process_directory(
            str(input_path), str(output_path), args.pattern, args.batch_size
        )

        logging.info(
            f"Processing complete: {totals['successful']}/{totals['total_lines']} lines processed successfully across {totals['total_files']} files"
        )

    else:
        logging.error(f"Input path {input_path} does not exist")


if __name__ == "__main__":
    main()

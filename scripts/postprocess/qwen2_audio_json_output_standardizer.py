#!/usr/bin/env python3

# Copyright 2025 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Qwen2-Audio JSON Output Standardizer

This script standardizes the existing JSON outputs from Qwen2-Audio
by applying LLM-based or rules-based refinement.
"""

import json
import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Union, Any
import re
from tqdm import tqdm

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


class JsonOutputStandardizer:
    """
    A class for standardizing existing JSON outputs from Qwen2-Audio.
    """

    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2", device="auto"):
        """
        Initialize the standardizer with a text-only LLM.

        Args:
            model_name (str): HuggingFace model name for the text-only LLM
            device (str): Device to run the model on ("cpu", "cuda", "auto")
        """
        self.expected_formats = EXPECTED_FORMATS
        self.translation_dict = TRANSLATION_DICT
        self.llm_available = False

        # Try to initialize the LLM
        if AutoTokenizer is not None and AutoModelForCausalLM is not None:
            try:
                logging.info(f"Initializing language model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, low_cpu_mem_usage=True, torch_dtype="auto"
                )

                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=device,
                    max_new_tokens=512,
                )

                self.llm_available = True
                logging.info("LLM initialized successfully")
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
            Provide only the category name without explanations.
            """

        format_info = self.expected_formats[metric_name]

        if format_info["type"] == "number":
            return f"""
            Extract the number of speakers from this response: "{raw_output}"
            If multiple numbers are mentioned, choose the most definitive one.
            Provide only the number (1-10) without any other text.
            """

        elif format_info["type"] == "multi-category":
            categories = ", ".join([f'"{cat}"' for cat in format_info["categories"]])
            return f"""
            Classify this response into one or more of these categories: {categories}
            Response: "{raw_output}"
            Extract only the most relevant categories from the valid list.
            If 'Other' is chosen, specify what it is if possible (e.g., "Other: Polish").
            Provide only the category name(s) without explanations, separated by commas if multiple.
            """

        else:  # "category"
            categories = ", ".join([f'"{cat}"' for cat in format_info["categories"]])
            return f"""
            Classify this response into exactly one of these categories: {categories}
            Response: "{raw_output}"
            Find the closest match from the valid categories.
            Provide only the exact category name without explanations.
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

    def standardize(self, metric_name, raw_output):
        """
        Standardize the output from Qwen2-Audio.

        Args:
            metric_name (str): Name of the metric
            raw_output (Union[str, int]): Raw output from Qwen2-Audio

        Returns:
            Union[str, int, List[str]]: Standardized output
        """
        # Convert int to str if needed
        if isinstance(raw_output, int):
            raw_output = str(raw_output)

        # First translate any non-English terms
        translated_output = self._translate_non_english(raw_output)

        # If LLM is available, use it
        if self.llm_available:
            try:
                prompt = self._generate_prompt(metric_name, translated_output)

                # Generate response from the text-only LLM
                response = self.pipeline(
                    prompt, do_sample=False, return_full_text=False
                )[0]["generated_text"]

                # Process the LLM's output
                return self._process_llm_output(metric_name, response)

            except Exception as e:
                logging.warning(
                    f"LLM processing failed for {metric_name}: {e}. Falling back to rules-based approach."
                )
                return self._rules_based_standardize(metric_name, translated_output)
        else:
            # Fall back to rules-based approach
            return self._rules_based_standardize(metric_name, translated_output)

    def standardize_json(self, qwen_output):
        """
        Standardize an entire JSON output from Qwen2-Audio.

        Args:
            qwen_output (dict): Dictionary containing Qwen2-Audio outputs

        Returns:
            dict: Dictionary with standardized outputs
        """
        # Create a copy to not modify the original
        standardized = qwen_output.copy()

        # Keep track of the original values
        original_values = {}

        # Process each metric in the output
        for key, value in qwen_output.items():
            # Skip non-qwen keys
            if not key.startswith("qwen_"):
                continue

            # Store original value
            original_values[key] = value

            # Standardize the value
            standardized[key] = self.standardize(key, value)

        # Add the original values for reference
        standardized["original_values"] = original_values

        return standardized


def process_json_file(file_path, output_path=None, use_llm=True):
    """
    Process a single JSON file containing Qwen2-Audio outputs.

    Args:
        file_path (str): Path to the JSON file
        output_path (str): Path to save the standardized JSON (default: original_path_standardized.json)
        use_llm (bool): Whether to use LLM-based standardization

    Returns:
        dict: Standardized output
    """
    # Set default output path if not provided
    if output_path is None:
        file_path_obj = Path(file_path)
        output_path = file_path_obj.with_stem(f"{file_path_obj.stem}_standardized")

    # Load the JSON file
    with open(file_path, "r", encoding="utf-8") as f:
        qwen_output = json.load(f)

    # Initialize the standardizer
    standardizer = JsonOutputStandardizer()

    # Standardize the output
    standardized = standardizer.standardize_json(qwen_output)

    # Save the standardized output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(standardized, f, indent=2, ensure_ascii=False)

    return standardized


def process_directory(input_dir, output_dir=None, use_llm=True, file_pattern="*.json"):
    """
    Process all JSON files in a directory.

    Args:
        input_dir (str): Directory containing JSON files
        output_dir (str): Directory to save standardized JSON files
        use_llm (bool): Whether to use LLM-based standardization
        file_pattern (str): Glob pattern for JSON files

    Returns:
        dict: Summary of processing results
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(input_dir), f"{os.path.basename(input_dir)}_standardized"
        )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of JSON files
    input_path = Path(input_dir)
    json_files = list(input_path.glob(file_pattern))

    if not json_files:
        logging.warning(
            f"No JSON files found in {input_dir} matching pattern {file_pattern}"
        )
        return {"error": "No JSON files found"}

    logging.info(f"Found {len(json_files)} JSON files to process")

    # Process each file
    results = {}
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        try:
            # Generate output path
            rel_path = json_file.relative_to(input_path)
            output_file = Path(output_dir) / rel_path.with_stem(
                f"{json_file.stem}_standardized"
            )

            # Ensure output directory exists
            os.makedirs(output_file.parent, exist_ok=True)

            # Process the file
            standardized = process_json_file(json_file, output_file, use_llm)

            results[str(json_file)] = {"output_file": str(output_file), "success": True}

        except Exception as e:
            logging.error(f"Error processing {json_file}: {e}")
            results[str(json_file)] = {"success": False, "error": str(e)}

    # Create summary
    summary = {
        "total_files": len(json_files),
        "processed": sum(1 for r in results.values() if r.get("success", False)),
        "failed": sum(1 for r in results.values() if not r.get("success", False)),
        "use_llm": use_llm,
    }

    # Save summary
    summary_path = Path(output_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "results": results}, f, indent=2, ensure_ascii=False
        )

    logging.info(f"Processing complete. Summary saved to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Standardize Qwen2-Audio JSON outputs")
    parser.add_argument("input", help="Input JSON file or directory")
    parser.add_argument(
        "--output",
        "-o",
        help="Output file or directory (default: adds '_standardized' to input name)",
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="Use rules-based approach instead of LLM"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Setup logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    logging.basicConfig(
        level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    input_path = Path(args.input)

    # If input is a directory, process all JSON files
    if input_path.is_dir():
        summary = process_directory(input_path, args.output, use_llm=not args.no_llm)

        print("\nProcessing Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")

    # If input is a file, process single file
    elif input_path.is_file():
        try:
            standardized = process_json_file(
                input_path, args.output, use_llm=not args.no_llm
            )
            print(f"Successfully standardized {input_path}")

        except Exception as e:
            logging.error(f"Error processing {input_path}: {e}")
            print(f"Failed to standardize {input_path}: {e}")

    else:
        logging.error(f"Input path {input_path} does not exist")
        print(f"Error: Input path {input_path} does not exist")


if __name__ == "__main__":
    main()

# Copyright 2025 Jiatong Shi
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""Shared, model-free Qwen output normalization rules."""

import re

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


def translate_non_english(text, translation_dict=TRANSLATION_DICT):
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
    for non_english, english in translation_dict.items():
        if non_english in text:
            # If it's a complete match, return the translation
            if text.strip() == non_english:
                return english
            # Otherwise replace the term within the text
            text = text.replace(non_english, english)

    return text


def _phrase_pattern(phrase):
    """Regex for `phrase` as whole words (lowercase).

    Word boundaries are ASCII-only: translated labels can sit right next to
    CJK characters ("Female性"), which Unicode word matching would treat as
    part of the same word."""
    return re.compile(r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)", re.ASCII)


def _contains_phrase(text, phrase):
    """Whether `phrase` occurs in `text` as whole words (case-insensitive).

    Plain substring tests confuse overlapping labels, e.g. "male" inside
    "female" or "one" inside "someone"."""
    return _phrase_pattern(phrase).search(text.lower()) is not None


def _match_category(text, categories):
    """Return the category named in `text`, or None.

    Categories are matched as whole words. A match lying inside a longer
    matched category is ignored ("low" inside "very low"); among the rest,
    the earliest mention wins."""
    text_lower = text.lower()
    spans = []
    for category in categories:
        for match in _phrase_pattern(category).finditer(text_lower):
            spans.append((match.start(), match.end(), category))
    outer = [
        (start, end, category)
        for start, end, category in spans
        if not any(
            s <= start and end <= e and (e - s) > (end - start) for s, e, _ in spans
        )
    ]
    if not outer:
        return None
    return min(outer, key=lambda span: span[0])[2]


def process_llm_output(metric_name, llm_output, expected_formats=EXPECTED_FORMATS):
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
    if metric_name in expected_formats:
        format_type = expected_formats[metric_name]["type"]

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


def rules_based_standardize(
    metric_name,
    raw_output,
    expected_formats=EXPECTED_FORMATS,
    translation_dict=TRANSLATION_DICT,
):
    """
    Standardize output using simple rules-based approach.

    Args:
        metric_name (str): Name of the metric
        raw_output (str): Raw output from Qwen2-Audio

    Returns:
        Union[str, int, List[str]]: Standardized output based on simple rules
    """
    # First translate any non-English terms
    translated_output = translate_non_english(raw_output, translation_dict)

    # Only process if we have format info
    if metric_name not in expected_formats:
        return translated_output

    format_info = expected_formats[metric_name]

    # Handle special case for language
    if metric_name == "qwen_language":
        matched = _match_category(translated_output, format_info["categories"])
        if matched is not None:
            return matched
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
        matched = _match_category(translated_output, format_info["categories"])
        if matched is not None:
            return matched
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
            if _contains_phrase(translated_output, word):
                return num

        return 1  # Default to 1

    # For regular categories
    if format_info["type"] == "category":
        # First check for a category named in full (whole words)
        matched = _match_category(translated_output, format_info["categories"])
        if matched is not None:
            return matched

        # If no exact match, try to find the best match
        best_match = None
        best_score = 0

        for category in format_info["categories"]:
            # Simple matching score based on word presence
            category_lower = category.lower()
            words = category_lower.split()

            # Count how many words from the category appear in the output
            score = sum(
                1 for word in words if _contains_phrase(translated_output, word)
            )

            if score > best_score:
                best_score = score
                best_match = category

        return best_match if best_match else format_info["categories"][0]

    return translated_output

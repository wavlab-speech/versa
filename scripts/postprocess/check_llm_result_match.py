import json
import sys
from typing import Dict, Any, List

# Define expected formats
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


def filter_json_object(json_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter a JSON object to keep only the 'key' and keys matching EXPECTED_FORMATS
    """
    if not isinstance(json_obj, dict):
        return {}

    filtered_obj = {}

    # Always keep the "key" key if it exists
    if "key" in json_obj:
        filtered_obj["key"] = json_obj["key"]

    # Only keep other keys if they match one in EXPECTED_FORMATS
    for key, value in json_obj.items():
        if key in EXPECTED_FORMATS:
            if EXPECTED_FORMATS[key]["type"] == "category":
                if value in EXPECTED_FORMATS[key]["categories"]:
                    filtered_obj[key] = value
            elif EXPECTED_FORMATS[key]["type"] == "number":
                try:
                    filtered_obj[key] = int(value)
                except:
                    continue

    return filtered_obj


def process_jsonl_file(input_file: str, output_file: str) -> None:
    """
    Process a JSONL file and create a filtered version
    """
    filtered_objects = []

    # Read and process each line
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                json_obj = json.loads(line.strip())
                filtered_obj = filter_json_object(json_obj)
                filtered_objects.append(filtered_obj)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line[:50]}...")

    # Write filtered objects to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        for obj in filtered_objects:
            f.write(json.dumps(obj) + "\n")

    print(f"Processed {len(filtered_objects)} objects")
    print(f"Filtered JSONL written to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_jsonl.py input.jsonl output.jsonl")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    process_jsonl_file(input_file, output_file)

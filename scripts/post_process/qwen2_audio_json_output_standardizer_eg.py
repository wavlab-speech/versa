#!/usr/bin/env python3

# Copyright 2025 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Example Usage of the JSON Output Standardizer

This script demonstrates how to use the JSON Output Standardizer
to process a single JSON output from Qwen2-Audio.
"""

from qwen_output_standardizer import JsonOutputStandardizer

# Example Qwen2-Audio output (similar to what you provided)
example_output = {
    'key': 'DEMAND_p298_172', 
    'qwen_speaker_count': '1', 
    'qwen_speaker_gender': 'Male', 
    'qwen_speaker_age': 'Middle-aged adult', 
    'qwen_speech_impairment': 'No apparent impairment: typical speech patterns', 
    'qwen_voice_pitch': 'Low', 
    'qwen_pitch_range': 'Narrow range', 
    'qwen_voice_type': 'Creaky/vocal fry', 
    'qwen_speech_volume_level': 'Loud', 
    'qwen_language': 'The language spoken in the audio is English.', 
    'qwen_speech_register': 'Intimate register', 
    'qwen_vocabulary_complexity': 'General', 
    'qwen_speech_purpose': 'Conversational', 
    'qwen_speech_emotion': 'Sad', 
    'qwen_speech_clarity': 'Low clarity', 
    'qwen_speech_rate': '慢',  # This is Chinese for "slow"
    'qwen_speaking_style': 'Casual', 
    'qwen_laughter_crying': 'No laughter or crying: speech only', 
    'qwen_overlapping_speech': 'There is no overlap in the audio.', 
    'qwen_speech_background_environment': 'Noisy indoor', 
    'qwen_recording_quality': 'Good', 
    'qwen_channel_type': 'Professional microphone'
}

def main():
    print("Original Qwen2-Audio Output:")
    for key, value in example_output.items():
        if key.startswith("qwen_"):
            print(f"  {key}: {value}")
    
    # Initialize the standardizer
    # Note: If you don't want to use an LLM, the standardizer will automatically 
    # fall back to rules-based approach if LLM initialization fails
    standardizer = JsonOutputStandardizer()
    
    # Standardize the output
    standardized = standardizer.standardize_json(example_output)
    
    print("\nStandardized Output:")
    for key, value in standardized.items():
        if key.startswith("qwen_") and key != "qwen_original_values":
            # Find any differences
            if key in example_output and example_output[key] != value:
                print(f"  {key}: {value} (was: {example_output[key]})")
            else:
                print(f"  {key}: {value}")
    
    # You can also save the standardized output to a file
    import json
    with open("standardized_example.json", "w", encoding="utf-8") as f:
        json.dump(standardized, f, indent=2, ensure_ascii=False)
    
    print("\nStandardized output saved to standardized_example.json")


if __name__ == "__main__":
    main()

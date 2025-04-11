#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
from typing import Dict, Optional, Union, Any

import librosa
import numpy as np
import torch
from Levenshtein import opcodes

logger = logging.getLogger(__name__)

# Handle optional whisper dependency
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    logger.warning(
        "Whisper is not properly installed. "
        "Please install following https://github.com/openai/whisper"
    )
    whisper = None
    WHISPER_AVAILABLE = False

from espnet2.text.cleaner import TextCleaner

# Constants
TARGET_FS = 16000
CHUNK_SIZE = 30  # seconds

class WhisperNotAvailableError(RuntimeError):
    """Exception raised when Whisper is required but not available."""
    pass

def asr_match_setup(
    model_tag: str = "default", 
    beam_size: int = 5, 
    text_cleaner: str = "whisper_basic", 
    use_gpu: bool = True
) -> Dict[str, Any]:
    """
    Set up ASR matching utilities.

    Args:
        model_tag: Whisper model tag. Options include "tiny", "base", "small", 
                 "medium", "large", or "large-v2". Defaults to "large".
        beam_size: Beam size for decoding.
        text_cleaner: Text cleaner type for post-processing.
        use_gpu: Whether to use GPU for computation.

    Returns:
        Dictionary containing the model, text cleaner, and beam size.

    Raises:
        WhisperNotAvailableError: If Whisper is not installed but is required.
        RuntimeError: If model loading fails.
    """
    if not WHISPER_AVAILABLE:
        raise WhisperNotAvailableError(
            "Whisper WER is used for evaluation while openai-whisper is not installed"
        )
    
    # Use the large model by default
    if model_tag == "default":
        model_tag = "large"
    
    # Set device based on availability and user preference
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    try:
        # Load the Whisper model
        logger.info(f"Loading Whisper model '{model_tag}' on {device}")
        model = whisper.load_model(model_tag, device=device)
        
        # Initialize text cleaner
        textcleaner = TextCleaner(text_cleaner)
        
        # Return utilities dictionary
        return {
            "model": model, 
            "cleaner": textcleaner, 
            "beam_size": beam_size
        }
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Whisper model: {str(e)}") from e

def asr_match_metric(
    wer_utils: Dict[str, Any],
    pred_x: np.ndarray, 
    gt_x: np.ndarray, 
    cache_pred_text: Optional[str] = None, 
    fs: int = 16000
) -> Dict[str, Union[float, str]]:
    """
    Calculate the ASR match error rate and related metrics.

    This function compares the ASR transcription of the predicted audio
    with the transcription of the ground truth audio to compute character-level
    edit distance metrics.

    Args:
        wer_utils: A utility dict for WER calculation including:
                  - model: whisper model
                  - cleaner: text cleaner
                  - beam_size: beam size for decoding
        pred_x: Predicted/test signal as a numpy array (time,)
        gt_x: Ground truth signal as a numpy array (time,)
        cache_pred_text: Optional pre-computed transcription for pred_x
        fs: Sampling rate of the input audio in Hz

    Returns:
        Dictionary containing:
        - asr_match_error_rate: The character error rate
        - whisper_hyp_text: The transcription of the predicted audio

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If transcription fails
    """
    # Validate inputs
    if pred_x is None or gt_x is None:
        raise ValueError("Both predicted and ground truth signals must be provided")
    
    # Make sure inputs are numpy arrays
    pred_x = np.asarray(pred_x)
    gt_x = np.asarray(gt_x)
    
    # Process the speech to be evaluated
    if cache_pred_text is not None:
        inf_text = cache_pred_text
    else:
        try:
            # Resample if necessary
            if fs != TARGET_FS:
                pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=TARGET_FS)
            
            # Convert to tensor and transcribe
            with torch.no_grad():
                transcription = wer_utils["model"].transcribe(
                    torch.tensor(pred_x).float(), 
                    beam_size=wer_utils["beam_size"]
                )
                inf_text = transcription["text"]
        except Exception as e:
            raise RuntimeError(f"Failed to transcribe predicted signal: {str(e)}") from e

    # Process the ground truth speech
    try:
        # Resample if necessary
        if fs != TARGET_FS:
            gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=TARGET_FS)
        
        # Convert to tensor and transcribe
        with torch.no_grad():
            transcription = wer_utils["model"].transcribe(
                torch.tensor(gt_x).float(), 
                beam_size=wer_utils["beam_size"]
            )
            gt_text = transcription["text"]
    except Exception as e:
        raise RuntimeError(f"Failed to transcribe ground truth signal: {str(e)}") from e

    # Clean the text using the provided cleaner
    ref_text = wer_utils["cleaner"](gt_text)
    pred_text = wer_utils["cleaner"](inf_text)

    # Convert texts to character lists for edit distance calculation
    ref_chars = list(ref_text)
    pred_chars = list(pred_text)
    
    # Initialize result dictionary with operation counts
    result = {
        "asr_match_delete": 0,  # Deletions: chars in reference but not in prediction
        "asr_match_insert": 0,  # Insertions: chars in prediction but not in reference
        "asr_match_replace": 0, # Substitutions: chars that differ between ref and pred
        "asr_match_equal": 0,   # Matches: chars that are the same in ref and pred
    }
    
    # Calculate edit operations using Levenshtein
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_chars, pred_chars):
        if op == "insert":
            result["asr_match_" + op] += inf_et - inf_st
        else:
            result["asr_match_" + op] += ref_et - ref_st
    
    # Validate operation counts
    total_ref = (
        result["asr_match_delete"]
        + result["asr_match_replace"]
        + result["asr_match_equal"]
    )
    if total_ref != len(ref_chars):
        logger.warning(
            f"Reference operation count mismatch: {total_ref} vs {len(ref_chars)}"
        )
    
    total_pred = (
        result["asr_match_insert"]
        + result["asr_match_replace"]
        + result["asr_match_equal"]
    )
    if total_pred != len(pred_chars):
        logger.warning(
            f"Prediction operation count mismatch: {total_pred} vs {len(pred_chars)}"
        )
    
    # Calculate error rate
    if len(ref_chars) == 0:
        # Handle empty reference case
        asr_match_error_rate = 1.0
        logger.warning("Reference text is empty, setting error rate to 1.0")
    else:
        # Calculate character error rate
        asr_match_error_rate = (
            result["asr_match_delete"]
            + result["asr_match_insert"]
            + result["asr_match_replace"]
        ) / len(ref_chars)
    
    # Return results
    return {
        "asr_match_error_rate": asr_match_error_rate, 
        "whisper_hyp_text": inf_text,
        # Additional metrics that might be useful
        "ref_text_length": len(ref_chars),
        "pred_text_length": len(pred_chars),
        "match_details": result
    }

def is_whisper_available():
    """
    Check if the Whisper package is available.
    
    Returns:
        bool: True if Whisper is available, False otherwise.
    """
    return WHISPER_AVAILABLE

if __name__ == "__main__":
    # Example usage
    try:
        # Generate random test audio (1 second at 16kHz)
        test_audio = np.random.random(TARGET_FS)
        
        # Set up ASR matching utilities
        wer_utils = asr_match_setup(model_tag="tiny", use_gpu=torch.cuda.is_available())
        
        # Calculate metrics
        metrics = asr_match_metric(wer_utils, test_audio, test_audio, None, TARGET_FS)
        
        # Print results
        print(f"ASR Match Error Rate: {metrics['asr_match_error_rate']:.4f}")
        print(f"Transcription: '{metrics['whisper_hyp_text']}'")
    except WhisperNotAvailableError:
        print("This script requires the Whisper package. Please install it first.")
    except Exception as e:
        print(f"Error running ASR match: {str(e)}")

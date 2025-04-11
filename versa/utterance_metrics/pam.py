#!/usr/bin/env python3

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional, Union, List, Any

warnings.filterwarnings("ignore")
import argparse
import os
import re
import sys
from dataclasses import dataclass

import torch
import torchaudio
import torchaudio.transforms as T
import yaml
from huggingface_hub.file_download import hf_hub_download
from transformers import AutoTokenizer, logging

from versa.utterance_metrics.pam_utils.clap import CLAP

logging.set_verbosity_error()
import collections

import numpy as np
import torch.nn.functional as F

# Constants
HF_REPO = "microsoft/msclap"
CLAP_VERSION = "CLAP_weights_2023.pth"
PAM_PROMPTS = [
    "the sound is clear and clean.",
    "the sound is noisy and with artifacts.",
]
RESAMPLE_RATE = 44100
AUDIO_DURATION = 7
SAMPLES = RESAMPLE_RATE * AUDIO_DURATION


@dataclass
class PAMConfig:
    """Configuration class for PAM model."""
    audioenc_name: str
    sampling_rate: int
    window_size: int
    hop_size: int
    mel_bins: int
    fmin: int
    fmax: int
    num_classes: int
    out_emb: int
    text_model: str
    transformer_embed_dim: int
    d_proj: int
    text_len: int


class PAM:
    """
    A class for Perceptual Audio Metric (PAM).
    PAM evaluates audio quality using text prompts and CLAP embeddings.
    """

    def __init__(
        self, 
        model_fp: Optional[Union[Path, str]] = None, 
        model_config: Optional[Dict[str, Any]] = None, 
        use_cuda: bool = False
    ):
        """
        Initialize PAM model.
        
        Args:
            model_fp: Path to the model weights file
            model_config: Model configuration dictionary
            use_cuda: Whether to use CUDA for computation
        """
        self.file_path = os.path.realpath(__file__)
        self.config = model_config
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

        # Automatically download model if not provided
        if not model_fp:
            model_fp = hf_hub_download(HF_REPO, CLAP_VERSION)

        self.model_fp = model_fp
        self.use_cuda = use_cuda
        self.clap, self.tokenizer, self.args = self._load_clap()

        # Two prompt strategy
        self.pam_prompts = PAM_PROMPTS
        self._get_text_embeddings()

    def _parse_config(self, config_dict: Dict[str, Any]) -> argparse.Namespace:
        """Parse configuration dictionary into argparse namespace."""
        return argparse.Namespace(**config_dict)

    def _load_clap(self) -> tuple:
        """Load CLAP model with args from config file."""
        args = self._parse_config(self.config)

        self.token_keys = ["input_ids", "attention_mask"]

        clap = CLAP(
            audioenc_name=args.audioenc_name,
            sample_rate=args.sampling_rate,
            window_size=args.window_size,
            hop_size=args.hop_size,
            mel_bins=args.mel_bins,
            fmin=args.fmin,
            fmax=args.fmax,
            classes_num=args.num_classes,
            out_emb=args.out_emb,
            text_model=args.text_model,
            transformer_embed_dim=args.transformer_embed_dim,
            d_proj=args.d_proj,
        )

        # Load pretrained weights for model
        checkpoint = torch.load(self.model_fp, map_location=self.device)
        model_state_dict = checkpoint["model"]

        # Load state dict
        clap.load_state_dict(model_state_dict, strict=False)
        clap.eval()  # set clap in eval mode
        
        # Setup tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.text_model)
        tokenizer.add_special_tokens({"pad_token": "!"})

        # Move model to appropriate device
        clap = clap.to(self.device)

        return clap, tokenizer, args

    def _preprocess_text(self, text_queries: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize text queries.
        
        Args:
            text_queries: List of text prompts
            
        Returns:
            Dictionary of tokenized text tensors
        """
        tokenized_texts = []
        for text in text_queries:
            if "gpt" in self.args.text_model:
                text = text + " <|endoftext|>"
            
            tok = self.tokenizer.encode_plus(
                text=text,
                add_special_tokens=True,
                max_length=self.args.text_len,
                padding="max_length",
                return_tensors="pt",
            )
            
            for key in self.token_keys:
                tok[key] = tok[key].reshape(-1).to(self.device)
            
            tokenized_texts.append(tok)

        # Batch tokenized texts
        tokenized_texts_batch = {
            key: torch.cat([d[key].reshape(1, -1) for d in tokenized_texts])
            for key in tokenized_texts[0]
        }
        
        return tokenized_texts_batch

    def _get_text_embeddings(self) -> None:
        """Compute and store text embeddings for PAM prompts."""
        preprocessed_text = self._preprocess_text(self.pam_prompts)
        with torch.no_grad():
            self.pam_embeddings = self.clap.caption_encoder(preprocessed_text)

    def _get_audio_embeddings(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute audio embeddings.
        
        Args:
            audio_tensor: Audio input tensor
            
        Returns:
            Audio embeddings
        """
        with torch.no_grad():
            return self.clap.audio_encoder(audio_tensor)[0]

    def _compute_similarity(self, audio_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between text and audio embeddings.
        
        Args:
            audio_embeddings: Audio embeddings tensor
            
        Returns:
            Similarity scores
        """
        # Normalize embeddings
        audio_embeddings = F.normalize(audio_embeddings, dim=-1)
        text_embeddings = F.normalize(self.pam_embeddings, dim=-1)

        # Compute similarity with temperature scaling
        logit_scale = self.clap.logit_scale.exp()
        similarity = logit_scale * text_embeddings @ audio_embeddings.T
        
        return similarity.T

    def evaluate(self, audio_tensor: torch.Tensor) -> float:
        """
        Compute PAM score for input audio tensor.
        
        Args:
            audio_tensor: Audio input tensor
            
        Returns:
            PAM score (quality score between 0 and 1)
        """
        # Move audio to device
        audio_tensor = audio_tensor.to(self.device)

        # Get embeddings and compute similarity
        audio_embeddings = self._get_audio_embeddings(audio_tensor)
        similarity = self._compute_similarity(audio_embeddings)
        
        # Apply softmax to get probability distribution
        probabilities = F.softmax(similarity, dim=1)
        
        # PAM score is the probability of "clear and clean" prompt
        pam_score = probabilities[:, 0].detach().mean().item()

        return pam_score


def load_audio(
    audio_file: Union[str, torch.Tensor], 
    sample_rate: int, 
    repro: bool = True
) -> torch.Tensor:
    """
    Load and preprocess audio file.
    
    Args:
        audio_file: Path to audio file or audio tensor
        sample_rate: Sample rate of the input audio
        repro: If True, use reproducible processing (taking first 7 seconds)
        
    Returns:
        Processed audio tensor
    """
    # Load audio file if path is provided
    if isinstance(audio_file, str):
        audio, sample_rate = torchaudio.load(audio_file)
    else:
        audio = audio_file.clone()  # Create a copy to avoid modifying the original
    
    # Ensure audio is a FloatTensor
    audio = torch.FloatTensor(audio)

    # Resample audio if needed
    if sample_rate != RESAMPLE_RATE:
        resampler = T.Resample(sample_rate, RESAMPLE_RATE)
        audio = resampler(audio)

    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # Reshape to 1D
    audio = audio.reshape(-1)
    
    # Process audio to be exactly AUDIO_DURATION seconds
    if SAMPLES >= audio.shape[0]:
        # Audio is shorter than required duration, repeat to match
        repeat_factor = int(np.ceil(SAMPLES / audio.shape[0]))
        audio = audio.repeat(repeat_factor)
        # Trim to exact length
        audio = audio[:SAMPLES]
    else:
        # Audio is longer than required duration
        if repro:
            # Take first AUDIO_DURATION seconds
            audio = audio[:SAMPLES]
        else:
            # Take chunks of AUDIO_DURATION seconds plus remaining portion
            cutoff = int(np.floor(audio.shape[0] / SAMPLES))
            initial_audio = audio[:cutoff * SAMPLES]
            
            remaining = audio[cutoff * SAMPLES:]
            if remaining.shape[0] > 0:
                # If remaining is non-empty, take the last AUDIO_DURATION seconds
                remaining = audio[-SAMPLES:] if remaining.shape[0] <= SAMPLES else remaining[:SAMPLES]
                audio = torch.cat([initial_audio, remaining])
            else:
                audio = initial_audio

    return audio


def pam_model_setup(model_config: Dict[str, Any], use_gpu: bool = False) -> PAM:
    """
    Initialize PAM model with given configuration.
    
    Args:
        model_config: Model configuration dictionary
        use_gpu: Whether to use GPU for computation
        
    Returns:
        Initialized PAM model
    """
    model = PAM(model_config=model_config, use_cuda=use_gpu)
    return model


def pam_metric(
    model: PAM, 
    pred_x: Union[str, torch.Tensor, np.ndarray], 
    gt_x: Optional[Union[str, torch.Tensor, np.ndarray]] = None, 
    fs: int = 16000
) -> Dict[str, float]:
    """
    Compute PAM metric for given audio.
    
    Args:
        model: PAM model
        pred_x: Predicted audio (file path or tensor)
        gt_x: Ground truth audio (unused, kept for API compatibility)
        fs: Sample rate of the input audio
        
    Returns:
        Dictionary containing PAM score
    """
    # Convert numpy array to tensor if needed
    if isinstance(pred_x, np.ndarray):
        pred_x = torch.FloatTensor(pred_x)
    
    # Load and preprocess audio
    audio = load_audio(pred_x, fs, repro=True)
    
    # Ensure audio has batch dimension
    if len(audio.shape) < 2:
        audio = audio.unsqueeze(0)
    
    # Compute PAM score
    pam_score = model.evaluate(audio)

    return {"pam_score": pam_score}


if __name__ == "__main__":
    # Example usage
    a = np.random.random(16000)
    
    # Load configuration from YAML file
    try:
        with open("egs/separate_metrics/pam.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)[0]
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Initialize model and compute metric
    try:
        model = pam_model_setup(config, use_gpu=torch.cuda.is_available())
        result = pam_metric(model, a, fs=16000)
        print(f"PAM score: {result['pam_score']:.4f}")
    except Exception as e:
        print(f"Error computing PAM metric: {e}")
        sys.exit(1)

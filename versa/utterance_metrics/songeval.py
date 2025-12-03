#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import librosa
import torch
import numpy as np
from muq import MuQ
from omegaconf import OmegaConf
from safetensors.torch import load_file
import requests
import logging
import subprocess
from hydra.utils import instantiate


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def songeval_model_setup(cache_dir="versa_cache", use_gpu=False):
    """
    Setup SongEval model for evaluation.
    Auto-downloads model if not present.
    """
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    repo_url = "https://github.com/ASLP-lab/SongEval.git"

    songeval_dir = cache_dir / "SongEval"

    if not songeval_dir.exists():
        logger.info(f"Cloning SongEval repository into {cache_dir}")
        subprocess.run(["git", "clone", repo_url, str(songeval_dir)], check=True)
    else:
        logger.info(f"Using existing SongEval repository in {cache_dir}")
    
    import sys
    sys.path.insert(0, str(songeval_dir)) 
    model_path = songeval_dir / "ckpt" / "model.safetensors"
    config_path = songeval_dir / "config.yaml"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found in {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found in {config_path}")

    # load classifier model weights
    with torch.no_grad():
        train_config = OmegaConf.load(config_path)
        # model = instantiate(train_config.generator).to(device).eval()
        model = instantiate(train_config.generator)
        state_dict = load_file(model_path, device="cpu")
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device).eval()

    # load MuQ as encoder
    muq_model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    muq_model = muq_model.to(device).eval()

    model_dict = {"model": model, "muq": muq_model, "device": device}
    return model_dict


def songeval_metric(model_dict, pred, fs):
    """
    pred: np.ndarray, original waveform
    fs: original sampling rate
    return: dict, metric results (5 dimensions)
    """
    device = model_dict["device"]
    model = model_dict["model"]
    muq_model = model_dict["muq"]

    # resample to 24kHz
    pred = librosa.resample(pred, orig_sr=fs, target_sr=24000)

    audio = torch.tensor(pred).unsqueeze(0).to(device)
    with torch.no_grad():
        output = muq_model(audio, output_hidden_states=True)
        hidden = output["hidden_states"][6]
        scores_g = model(hidden).squeeze(0)

    values = {
        "Coherence": round(scores_g[0].item(), 4),
        "Musicality": round(scores_g[1].item(), 4),
        "Memorability": round(scores_g[2].item(), 4),
        "Clarity": round(scores_g[3].item(), 4),
        "Naturalness": round(scores_g[4].item(), 4),
    }

    return values


if __name__ == "__main__":
    # tese example
    a = np.random.rand(24000).astype(np.float32)
    model = songeval_model_setup(use_gpu=True)
    print("metrics:", songeval_metric(model, a, 24000))

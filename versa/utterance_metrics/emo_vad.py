#!/usr/bin/env python3

# Copyright 2025 BoHao Su
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Module for dimensional emotion prediction metrics using w2v2-how-to."""

import logging
import os
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
        self,
        input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits


def w2v2_emo_dim_setup(
    model_tag="default", model_path=None, model_config=None, use_gpu=False
):
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    if model_path is not None and model_config is not None:
        model = EmotionModel.from_pretrained(
            pretrained_model_name_or_path=model_path, config=model_config
        ).to(device)
    else:
        if model_tag == "default":
            model_tag = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        model = EmotionModel.from_pretrained(model_tag).to(device)
    processor = Wav2Vec2Processor.from_pretrained(
        "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    )
    emo_utils = {"model": model, "processor": processor, "device": device}
    return emo_utils


def dim_emo_pred(emo_utils, pred_x, fs):
    """Calculate dimensional emotion (arousal, dominance, valence) of input audio samples.

    Args:
        model (w2v2-how-to): The loaded EMO2VEC model.
        pred_x (np.ndarray): Predicted audio signal.
        fs (int): Sampling rate.

    Returns:
        dict: Dictionary containing the dimensional emotion predictions.
    """
    # NOTE(jiatong): only work for 16000 Hz
    if fs != 16000:
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)
    pred_x = emo_utils["processor"](pred_x, sampling_rate=16000)
    pred_x = pred_x["input_values"][0]
    pred_x = pred_x.reshape(1, -1)
    pred_x = torch.from_numpy(pred_x).to(emo_utils["device"])
    with torch.no_grad():
        avd_emo = emo_utils["model"](pred_x)[1].squeeze(0).cpu().numpy()

    return {"aro_val_dom_emo": avd_emo}


if __name__ == "__main__":
    a = np.random.random(16000)
    emo_utils = w2v2_emo_dim_setup()
    print(f"metrics: {dim_emo_pred(emo_utils, a, 16000)}")

#!/usr/bin/env python3


from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import argparse
import os
import re
import sys

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

HF_REPO = "microsoft/msclap"
CLAP_VERSION = "CLAP_weights_2023.pth"
PAM_PROMPTS = [
    "the sound is clear and clean.",
    "the sound is noisy and with artifacts.",
]
RESAMPLE_RATE = 44100
AUDIO_DURATION = 7
SAMPLES = RESAMPLE_RATE * AUDIO_DURATION


class PAM:
    """
    A class for PAM metric.
    """

    def __init__(
        self, model_fp: Path | str | None = None, model_config=None, use_cuda=False
    ):
        self.np_str_obj_array_pattern = re.compile(r"[SaUO]")
        self.file_path = os.path.realpath(__file__)
        self.default_collate_err_msg_format = (
            "default_collate: batch must contain tensors, numpy arrays, numbers, "
            "dicts or lists; found {}"
        )
        self.config = model_config

        # Automatically download model if not provided
        if not model_fp:
            model_fp = hf_hub_download(HF_REPO, CLAP_VERSION)

        self.model_fp = model_fp
        self.use_cuda = use_cuda
        self.clap, self.tokenizer, self.args = self.load_clap()

        # Two prompt strategy
        self.pam_prompts = PAM_PROMPTS
        self.get_text_embeddings()

    def read_config_as_args(self, yml_config, args=None, is_config_str=False):
        return_dict = {}

        if args != None:
            for k, v in yml_config.items():
                if k in args.__dict__:
                    args.__dict__[k] = v
                else:
                    sys.stderr.write(
                        "Ignored unknown parameter {} in yaml.\n".format(k)
                    )
        else:
            for k, v in yml_config.items():
                return_dict[k] = v

        args = args if args != None else return_dict
        return argparse.Namespace(**args)

    def load_clap(self):
        r"""Load CLAP model with args from config file"""

        args = self.read_config_as_args(self.config, is_config_str=True)

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
        model_state_dict = torch.load(self.model_fp, map_location=torch.device("cpu"))[
            "model"
        ]

        # We unwrap the DDP model and save. If the model is not unwrapped and saved, then the model needs to unwrapped before `load_state_dict`:
        # Reference link: https://discuss.pytorch.org/t/how-to-load-dataparallel-model-which-trained-using-multiple-gpus/146005
        clap.load_state_dict(model_state_dict, strict=False)

        clap.eval()  # set clap in eval mode
        tokenizer = AutoTokenizer.from_pretrained(args.text_model)
        tokenizer.add_special_tokens({"pad_token": "!"})

        if self.use_cuda and torch.cuda.is_available():
            clap = clap.cuda()

        return clap, tokenizer, args

    def default_collate(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif (
            elem_type.__module__ == "numpy"
            and elem_type.__name__ != "str_"
            and elem_type.__name__ != "string_"
        ):
            if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
                # array of string classes and object
                if self.np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(
                        self.default_collate_err_msg_format.format(elem.dtype)
                    )

                return self.default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, collections.abc.Mapping):
            return {key: self.default_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
            return elem_type(
                *(self.default_collate(samples) for samples in zip(*batch))
            )
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError(
                    "each element in list of batch should be of equal size"
                )
            transposed = zip(*batch)
            return [self.default_collate(samples) for samples in transposed]

        raise TypeError(self.default_collate_err_msg_format.format(elem_type))

    def preprocess_text(self, text_queries):
        r"""Load list of class labels and return tokenized text"""
        tokenized_texts = []
        for ttext in text_queries:
            if "gpt" in self.args.text_model:
                ttext = ttext + " <|endoftext|>"
            tok = self.tokenizer.encode_plus(
                text=ttext,
                add_special_tokens=True,
                max_length=self.args.text_len,
                padding="max_length",
                return_tensors="pt",
            )
            for key in self.token_keys:
                tok[key] = (
                    tok[key].reshape(-1).cuda()
                    if self.use_cuda and torch.cuda.is_available()
                    else tok[key].reshape(-1)
                )
            tokenized_texts.append(tok)

        tokenized_texts_batch = {
            key: torch.cat([d[key].reshape(1, -1) for d in tokenized_texts])
            for key in tokenized_texts[0]
        }
        return tokenized_texts_batch

    def get_text_embeddings(self):
        r"""Save text embeddings of PAM prompts"""
        preprocessed_text = self.preprocess_text(self.pam_prompts)
        self.pam_embeddings = self._get_text_embeddings(preprocessed_text)

    def _get_text_embeddings(self, preprocessed_text):
        r"""Load preprocessed text and return text embeddings"""
        with torch.no_grad():
            return self.clap.caption_encoder(preprocessed_text)

    def _get_audio_embeddings(self, preprocessed_audio):
        r"""Load preprocessed audio and return a audio embeddings"""
        with torch.no_grad():
            return self.clap.audio_encoder(preprocessed_audio)[0]

    def compute_similarity(self, audio_embeddings):
        r"""Compute similarity between text and audio embeddings"""
        audio_embeddings = audio_embeddings / torch.norm(
            audio_embeddings, dim=-1, keepdim=True
        )
        text_embeddings = self.pam_embeddings / torch.norm(
            self.pam_embeddings, dim=-1, keepdim=True
        )

        logit_scale = self.clap.logit_scale.exp()
        similarity = logit_scale * text_embeddings @ audio_embeddings.T
        return similarity.T

    def evaluate(self, audio_tensor):
        r"""Compute PAM score using audio tensors"""
        if self.use_cuda and torch.cuda.is_available():
            audio_tensor = audio_tensor.cuda()

        audio_embedddings = self._get_audio_embeddings(audio_tensor)
        sim = self.compute_similarity(audio_embedddings)
        prob = F.softmax(sim, dim=1)
        pam_score = prob[:, 0]

        pam_score = pam_score.detach().mean().item()

        return pam_score


def load_audio(audio_file, sample_rate, repro=True):
    r"""Loads audio file and returns raw audio."""
    if type(audio_file) == str:
        audio_file, sample_rate = torchaudio.load(audio_file)
    else:
        audio = audio_file
    audio = torch.FloatTensor(audio)

    # Resample audio if needed
    resampler = T.Resample(sample_rate, RESAMPLE_RATE)
    audio = resampler(audio)

    # process audio to be a multiple of 7 seconds
    audio = audio.reshape(-1)
    if SAMPLES >= audio.shape[0]:
        repeat_factor = int(np.ceil((SAMPLES) / audio.shape[0]))
        # Repeast audio_time_series by repeat_factor to match audio_duration
        audio = audio.repeat(repeat_factor)
        # remove excess part of audio_time_series
        audio = audio[0:SAMPLES]
    else:
        if repro:
            # retain only first 7 seconds
            start_index = 0
            audio = audio[start_index : start_index + SAMPLES]
        else:
            cutoff = int(np.floor(audio.shape[0] / SAMPLES))
            # cutoff audio
            initial_audio_series = audio[0 : cutoff * SAMPLES]
            # remaining audio repeat and cut off
            remaining = audio[cutoff * SAMPLES :]
            if remaining.shape[0] != 0:
                remaining = audio[-SAMPLES:]
                audio = torch.cat([initial_audio_series, remaining])
            else:
                audio = initial_audio_series

    return audio


def pam_model_setup(model_config, use_gpu=False):
    model = PAM(model_config=model_config, use_cuda=use_gpu)
    return model


def pam_metric(model, pred_x, gt_x=None, fs=None):
    # NOTE(jiatong): only work for 16000 Hz

    audio = load_audio(pred_x, fs, repro=True)
    if len(audio.shape) < 2:
        audio = audio[None, :]
    pam_score = model.evaluate(audio)

    return {"pam_score": pam_score}


if __name__ == "__main__":
    a = np.random.random(16000)
    with open("egs/separate_metrics/pam.yaml", "r", encoding="utf-8") as f:
        config = yaml.full_load(f)[0]
    model = pam_model_setup(config, True)
    print("metrics: {}".format(pam_metric(model, a, fs=16000)))

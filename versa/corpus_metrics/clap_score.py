#!/usr/bin/env python3

# Copyright 2025 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import os

import librosa
import numpy as np

from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType
from versa.scorer_shared import audio_loader_setup
from versa.utils_shared import load_audio, wav_normalize

try:
    from frechet_audio_distance import CLAPScore
except ImportError:
    CLAPScore = None


def clap_score_setup(
    submodel_name="630k-audioset",
    ckpt_dir=None,
    verbose=False,
    audio_load_worker=8,
    enable_fusion=False,
    cache_dir="versa_cache/clap_score",
    cache_embeddings=False,
    io="kaldi",
):
    if CLAPScore is None:
        raise ModuleNotFoundError(
            "frechet_audio_distance is not installed. "
            "Please install it with `pip install frechet-audio-distance`."
        )

    clap = CLAPScore(
        ckpt_dir=ckpt_dir,
        submodel_name=submodel_name,
        verbose=verbose,
        audio_load_worker=audio_load_worker,
        enable_fusion=enable_fusion,
    )

    return {
        "module": clap,
        "cache_dir": cache_dir,
        "cache_embeddings": cache_embeddings,
        "io": io,
        "verbose": verbose,
    }


def _load_audio_entry(audio_info):
    if isinstance(audio_info, tuple):
        return load_audio(audio_info, "kaldi")
    return load_audio(audio_info, "soundfile")


def _load_audio_list(audio_files, target_sample_rate):
    audio_data = []
    for key in sorted(audio_files):
        sample_rate, wav = _load_audio_entry(audio_files[key])
        wav = wav_normalize(wav).astype(np.float32)
        if sample_rate != target_sample_rate:
            wav = librosa.resample(
                wav, orig_sr=sample_rate, target_sr=target_sample_rate
            )
        audio_data.append(wav.astype(np.float32))
    return audio_data


def clap_score_scoring(
    pred_x,
    clap_info,
    text_info=None,
    key_info="clap_score",
    batch_size=10,
):
    if text_info is None:
        raise ValueError("CLAP score requires text references via --text.")

    clap = clap_info["module"]
    if isinstance(pred_x, dict):
        eval_files = pred_x
    else:
        eval_files = audio_loader_setup(pred_x, clap_info["io"])
    keys = sorted(eval_files)
    missing_text = [key for key in keys if key not in text_info]
    if missing_text:
        raise ValueError(
            "Missing text references for CLAP score: {}".format(
                ", ".join(missing_text[:5])
            )
        )

    cache_embeddings = clap_info["cache_embeddings"]
    if cache_embeddings:
        os.makedirs(clap_info["cache_dir"], exist_ok=True)
        text_embds_path = os.path.join(clap_info["cache_dir"], "text_embeddings.npy")
        audio_embds_path = os.path.join(clap_info["cache_dir"], "audio_embeddings.npy")
    else:
        text_embds_path = None
        audio_embds_path = None

    if text_embds_path is not None and os.path.exists(text_embds_path):
        logging.info("[CLAP score] Loading text embeddings from %s", text_embds_path)
        text_embds = np.load(text_embds_path)
    else:
        text_data = [text_info[key] for key in keys]
        text_embds = clap.get_text_embeddings(text_data)
        if text_embds_path is not None:
            np.save(text_embds_path, text_embds)

    if audio_embds_path is not None and os.path.exists(audio_embds_path):
        logging.info("[CLAP score] Loading audio embeddings from %s", audio_embds_path)
        audio_embds = np.load(audio_embds_path)
    else:
        audio_data = _load_audio_list(eval_files, clap.sample_rate)
        audio_embds = clap.get_audio_embeddings(audio_data, sr=clap.sample_rate)
        if audio_embds_path is not None:
            np.save(audio_embds_path, audio_embds)

    if len(text_embds) == 0:
        raise ValueError("CLAP score text embeddings are empty.")
    if len(audio_embds) == 0:
        raise ValueError("CLAP score audio embeddings are empty.")
    if text_embds.shape != audio_embds.shape:
        raise ValueError(
            "CLAP score text and audio embeddings have different shapes: "
            f"{text_embds.shape} vs. {audio_embds.shape}."
        )

    clap_score_mean, _ = clap.calculate_clap_score(
        text_embds, audio_embds, batch_size=batch_size
    )
    return {key_info: float(clap_score_mean)}


class ClapScoreMetric(BaseMetric):
    """Corpus-level CLAP score for paired text/audio alignment."""

    def _setup(self):
        self.io = self.config.get("io", "kaldi")
        self.batch_size = self.config.get("batch_size", 10)
        self.clap_info = clap_score_setup(
            submodel_name=self.config.get("submodel_name", "630k-audioset"),
            ckpt_dir=self.config.get("ckpt_dir", None),
            verbose=self.config.get("verbose", False),
            audio_load_worker=self.config.get("audio_load_worker", 8),
            enable_fusion=self.config.get("enable_fusion", False),
            cache_dir=self.config.get("cache_dir", "versa_cache/clap_score"),
            cache_embeddings=self.config.get("cache_embeddings", False),
            io=self.io,
        )

    def compute(self, predictions, references=None, metadata=None):
        metadata = metadata or {}
        text_info = metadata.get("text_info")
        if text_info is None:
            raise ValueError("CLAP score requires text references via --text.")

        scores = clap_score_scoring(
            predictions,
            self.clap_info,
            text_info=text_info,
            batch_size=self.batch_size,
        )
        return scores["clap_score"]

    def get_metadata(self):
        return _clap_score_metadata()


def _clap_score_metadata():
    return MetricMetadata(
        name="clap_score",
        category=MetricCategory.DISTRIBUTIONAL,
        metric_type=MetricType.FLOAT,
        requires_reference=False,
        requires_text=True,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["frechet_audio_distance", "librosa", "numpy"],
        description="CLAP score for paired text/audio alignment.",
        paper_reference="https://arxiv.org/abs/2301.12661",
        implementation_source="https://github.com/gudgud96/frechet-audio-distance",
    )


def register_clap_score_metric(registry):
    registry.register(
        ClapScoreMetric,
        _clap_score_metadata(),
        aliases=["clap", "clapscore"],
    )


if __name__ == "__main__":
    clap_info = clap_score_setup()
    print(
        clap_score_scoring(
            "test/test_samples/test2.scp",
            clap_info,
            {"test.wav": "speech audio"},
        )
    )

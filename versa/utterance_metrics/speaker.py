#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np

from versa.audio_utils import resample_audio

try:
    from espnet2.bin.spk_inference import Speech2Embedding
except ImportError:
    Speech2Embedding = None

from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType


def speaker_model_setup(
    model_tag="default",
    model_path=None,
    model_config=None,
    use_gpu=False,
    cache_dir=None,
):
    if Speech2Embedding is None:
        raise ImportError("speaker requires espnet. Please install espnet and retry")

    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    if model_path is not None and model_config is not None:
        model = Speech2Embedding(
            model_file=model_path, train_config=model_config, device=device
        )
    else:
        if model_tag == "default":
            model_tag = "espnet/voxcelebs12_rawnet3"
        if cache_dir is None:
            model = Speech2Embedding.from_pretrained(model_tag=model_tag, device=device)
        else:
            try:
                from espnet_model_zoo.downloader import ModelDownloader
            except ImportError:
                raise ImportError(
                    "speaker requires espnet_model_zoo. Please install it and retry"
                )
            model_kwargs = ModelDownloader(cachedir=cache_dir).download_and_unpack(
                model_tag
            )
            model = Speech2Embedding(device=device, **model_kwargs)
    return model


def speaker_metric(model, pred_x, gt_x, fs):
    # NOTE(jiatong): only work for 16000 Hz
    if fs != 16000:
        gt_x = resample_audio(gt_x, fs, 16000)
        pred_x = resample_audio(pred_x, fs, 16000)

    embedding_gen = model(pred_x).squeeze(0).cpu().numpy()
    embedding_gt = model(gt_x).squeeze(0).cpu().numpy()
    similarity = np.dot(embedding_gen, embedding_gt) / (
        np.linalg.norm(embedding_gen) * np.linalg.norm(embedding_gt)
    )
    return {"spk_similarity": similarity}


class SpeakerMetric(BaseMetric):
    """Speaker embedding cosine similarity."""

    def _setup(self):
        self.model_tag = self.config.get("model_tag", "default")
        self.model_path = self.config.get("model_path")
        self.model_config = self.config.get("model_config")
        self.use_gpu = self.config.get("use_gpu", False)
        self.cache_dir = self.config.get("cache_dir", "versa_cache/espnet_model_zoo")
        self.model = speaker_model_setup(
            model_tag=self.model_tag,
            model_path=self.model_path,
            model_config=self.model_config,
            use_gpu=self.use_gpu,
            cache_dir=self.cache_dir,
        )

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("Predicted signal must be provided")
        if references is None:
            raise ValueError("Reference signal must be provided")

        fs = metadata.get("sample_rate", 16000) if metadata else 16000
        return speaker_metric(
            self.model, np.asarray(predictions), np.asarray(references), fs
        )

    def get_metadata(self):
        return _speaker_metadata()


def _speaker_metadata():
    return MetricMetadata(
        name="speaker",
        category=MetricCategory.NON_MATCH,
        metric_type=MetricType.FLOAT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["espnet2", "librosa", "numpy"],
        description="Speaker embedding cosine similarity",
        paper_reference="https://arxiv.org/abs/2401.17230",
        implementation_source="https://github.com/espnet/espnet",
    )


def register_speaker_metric(registry):
    """Register speaker similarity with the registry."""
    registry.register(
        SpeakerMetric,
        _speaker_metadata(),
        aliases=["spk_similarity", "speaker_similarity"],
    )


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    metric = SpeakerMetric()
    print("metrics: {}".format(metric.compute(a, b, metadata={"sample_rate": 16000})))

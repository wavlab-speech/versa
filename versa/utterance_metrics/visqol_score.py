#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os

import numpy as np

from versa.audio_utils import resample_audio
from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType

try:
    from visqol import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2
except ImportError:
    visqol_lib_py = None
    visqol_config_pb2 = None


def visqol_setup(model):
    # model name related to
    # https://github.com/google/visqol/tree/master/model
    if visqol_lib_py is None or visqol_config_pb2 is None:
        raise ImportError(
            "visqol is not installed. Please install visqol following "
            "https://github.com/google/visqol and retry"
        )

    config = visqol_config_pb2.VisqolConfig()
    config.audio.sample_rate = 48000

    if model == "default":
        model_tag = "libsvm_nu_svr_model.txt"
    elif model == "general":
        model_tag = "tcdaudio14_aacvopus_coresv_svrnsim_n.68_g.01_c1.model"
    elif model == "grid-search":
        model_tag = (
            "tcdaudio_aacvopus_coresv_grid_nu0.3_c126.237786175_g0.204475514639.model"
        )
    elif model == "speech":
        model_tag = "tcdvoip_nu.568_c5.31474325639_g3.17773760038_model.txt"
        config.options.use_speech_scoring = True
        config.audio.sample_rate = 16000
    else:
        raise NotImplementedError(
            "Not a valid tag for model, check "
            "https://github.com/google/visqol/tree/master/model for details"
        )

    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", model_tag
    )

    api = visqol_lib_py.VisqolApi()

    api.Create(config)

    return api, config.audio.sample_rate


def visqol_metric(api, api_fs, pred_x, gt_x, fs):
    if api_fs != fs:
        gt_x = resample_audio(gt_x, fs, api_fs)
        pred_x = resample_audio(pred_x, fs, api_fs)

    similarity_result = api.Measure(gt_x, pred_x)

    return {"visqol": similarity_result.moslqo}


class VisqolMetric(BaseMetric):
    """Virtual Speech Quality Objective Listener metric."""

    def _setup(self):
        self.model = self.config.get("model", "default")
        self.api, self.api_fs = visqol_setup(self.model)

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("Predicted signal must be provided")
        if references is None:
            raise ValueError("Reference signal must be provided")

        fs = metadata.get("sample_rate", 16000) if metadata else 16000
        return visqol_metric(
            self.api,
            self.api_fs,
            np.asarray(predictions),
            np.asarray(references),
            fs,
        )

    def get_metadata(self):
        return _visqol_metadata()


def _visqol_metadata():
    return MetricMetadata(
        name="visqol",
        category=MetricCategory.DEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=False,
        auto_install=False,
        dependencies=["visqol", "librosa", "numpy"],
        description="Virtual Speech Quality Objective Listener MOS-LQO metric",
        paper_reference="https://arxiv.org/abs/2004.09584",
        implementation_source="https://github.com/google/visqol",
    )


def register_visqol_metric(registry):
    """Register VISQOL with the registry."""
    registry.register(
        VisqolMetric,
        _visqol_metadata(),
        aliases=["visqol_metric", "VISQOL"],
    )


if __name__ == "__main__":
    a = np.random.random(int(16000 * 1))
    b = np.random.random(int(16000 * 1))
    metric = VisqolMetric()
    print(metric.compute(a, b, metadata={"sample_rate": 16000}))

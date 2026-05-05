#!/usr/bin/env python3

# Copyright 2025 Wangyou Zhang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
from pathlib import Path
import sys
import yaml

import numpy as np
import torch

from versa.audio_utils import resample_audio
from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType

logger = logging.getLogger(__name__)


vqscore_dir = str(Path(__file__).parent / "VQscore")
sys.path.append(vqscore_dir)
try:
    from models.VQVAE_models import VQVAE_QE
except ImportError:
    logger.info(
        "After cloning this repository, please run the following command to"
        "initialize the submodule 'VQscore':"
        "```bash"
        "git submodule update --init --recursive"
        "```"
    )
    VQVAE_QE = None


def vqscore_setup(use_gpu=False):
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    if VQVAE_QE is None:
        raise ModuleNotFoundError(
            "After cloning this repository, please run the following command to"
            "initialize the submodule 'VQscore':"
            "```bash"
            "./tools/install_vqscore.sh"
            "```"
        )

    vqscore_conf = str(
        Path(vqscore_dir)
        / (
            "config/"
            "QE_cbook_size_2048_1_32_IN_input_encoder_z_Librispeech_clean_github.yaml"
        )
    )
    vqscore_model = str(
        Path(vqscore_dir)
        / (
            "exp/QE_cbook_size_2048_1_32_IN_input_encoder_z_Librispeech_clean_github/"
            "checkpoint-dnsmos_ovr_CC=0.835.pkl"
        )
    )

    with open(vqscore_conf, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    model = VQVAE_QE(**config["VQVAE_params"]).to(device=device).eval()
    model.load_state_dict(
        torch.load(vqscore_model, map_location=device)["model"]["VQVAE"]
    )
    model.input_transform = config["input_transform"]
    model.device = device
    return model


# ported from VQscore/inference.py
def stft_magnitude(x, hop_size, fft_size=512, win_length=512):
    if x.is_cuda:
        x_stft = torch.stft(
            x,
            fft_size,
            hop_size,
            win_length,
            window=torch.hann_window(win_length).to("cuda"),
            return_complex=False,
        )
    else:
        x_stft = torch.stft(
            x,
            fft_size,
            hop_size,
            win_length,
            window=torch.hann_window(win_length),
            return_complex=False,
        )
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


def cos_similarity(SP_noisy, SP_y_noisy, eps=1e-5):
    SP_noisy_norm = torch.norm(SP_noisy, p=2, dim=-1, keepdim=True) + eps
    SP_y_noisy_norm = torch.norm(SP_y_noisy, p=2, dim=-1, keepdim=True) + eps
    Cos_frame = torch.sum(
        SP_noisy / SP_noisy_norm * SP_y_noisy / SP_y_noisy_norm, dim=-1
    )  # torch.Size([B, T, 1]

    return torch.mean(Cos_frame)


def vqscore_metric(model, pred_x, fs):
    # NOTE(wangyou): current model only have 16k options
    if fs != 16000:
        pred_x = resample_audio(pred_x, fs, 16000)

    with torch.no_grad():
        audio = torch.as_tensor(
            pred_x, dtype=torch.float32, device=model.device
        ).unsqueeze(0)
        SP_input = stft_magnitude(audio, hop_size=256)
        if model.input_transform == "log1p":
            SP_input = torch.log1p(SP_input)
        z = model.CNN_1D_encoder(SP_input)
        zq, indices, vqloss, distance = model.quantizer(
            z, stochastic=False, update=False
        )
        VQScore_cos_z = cos_similarity(z.transpose(2, 1).cpu(), zq.cpu()).numpy()

    return {"vqscore": float(VQScore_cos_z)}


class VqscoreMetric(BaseMetric):
    """VQScore speech quality assessment metric."""

    def _setup(self):
        self.use_gpu = self.config.get("use_gpu", False)
        self.model = vqscore_setup(use_gpu=self.use_gpu)

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("Predicted signal must be provided")

        fs = metadata.get("sample_rate", 16000) if metadata else 16000
        return vqscore_metric(self.model, np.asarray(predictions), fs)

    def get_metadata(self):
        return _vqscore_metadata()


def _vqscore_metadata():
    return MetricMetadata(
        name="vqscore",
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=False,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["torch", "librosa", "numpy", "yaml"],
        description=(
            "VQScore speech quality assessment from encoded and quantized features"
        ),
        paper_reference="https://arxiv.org/abs/2402.16321",
        implementation_source="https://github.com/JasonSWFu/VQscore",
    )


def register_vqscore_metric(registry):
    """Register VQScore with the registry."""
    registry.register(
        VqscoreMetric,
        _vqscore_metadata(),
        aliases=["vq_score", "vqscore_metric"],
    )


if __name__ == "__main__":
    a = np.random.random(16000)
    metric = VqscoreMetric()
    print(metric.compute(a, metadata={"sample_rate": 16000}))

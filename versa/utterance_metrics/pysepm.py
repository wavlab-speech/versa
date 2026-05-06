#!/usr/bin/env python3
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import numpy as np

from versa.audio_utils import resample_audio
from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType

logger = logging.getLogger(__name__)

try:
    import pysepm  # Import the pysepm package for speech quality metrics
except ImportError:
    logger.info(
        "pysepm is not installed. Please use `tools/install_pysepm.sh` to install"
    )
    pysepm = None


def is_pysepm_available():
    return pysepm is not None


def fwsegsnr(pred_x, gt_x, fs, frame_len=0.03, overlap=0.75):
    """
    Compute Frequency-Weighted Segmental SNR.

    Args:
        pred_x (np.array): Audio signal to be evaluated signal.
        gt_x (np.array): Ground truth (clean) signal.
        fs (int): Sampling rate in Hz.
        frame_len (float): Frame length in seconds.
        overlap (float): Overlap ratio between frames.

    Returns:
        float: fwSNRseg score.
    """
    fwsegsnr_score = pysepm.fwSNRseg(
        cleanSig=gt_x, enhancedSig=pred_x, fs=fs, frameLen=frame_len, overlap=overlap
    )
    return fwsegsnr_score


def llr(pred_x, gt_x, fs, frame_len=0.03, overlap=0.75):
    """
    Compute the Log-Likelihood Ratio (LLR) between predicted and ground truth signals.

    Args:
        pred_x (np.array): Audio signal to be evaluated signal.
        gt_x (np.array): Ground truth (clean) signal.
        fs (int): Sampling rate in Hz.
        frame_len (float): Frame length in seconds.
        overlap (float): Overlap ratio between frames.

    Returns:
        float: LLR score.
    """
    llr_score = pysepm.llr(
        clean_speech=gt_x,
        processed_speech=pred_x,
        fs=fs,
        used_for_composite=False,
        frameLen=frame_len,
        overlap=overlap,
    )
    return llr_score


def wss(pred_x, gt_x, fs, frame_len=0.03, overlap=0.75):
    """
    Compute the Weighted Spectral Slope (WSS) measure.

    Args:
        pred_x (np.array): Audio signal to be evaluated signal.
        gt_x (np.array): Ground truth (clean) signal.
        fs (int): Sampling rate in Hz.
        frame_len (float): Frame length in seconds.
        overlap (float): Overlap ratio between frames.

    Returns:
        float: WSS score.
    """
    wss_score = pysepm.wss(
        clean_speech=gt_x,
        processed_speech=pred_x,
        fs=fs,
        frameLen=frame_len,
        overlap=overlap,
    )
    return wss_score


def cd(pred_x, gt_x, fs):
    """
    Compute the Cepstral Distance (CD) between predicted and ground truth signals.

    Args:
        pred_x (np.array): Audio signal to be evaluated signal.
        gt_x (np.array): Ground truth (clean) signal.
        fs (int): Sampling rate in Hz.

    Returns:
        float: Cepstral Distance score.
    """
    cep_dist_score = pysepm.cepstrum_distance(
        clean_speech=gt_x,
        processed_speech=pred_x,
        fs=fs,
        frameLen=0.03,
        overlap=0.75,
    )
    return cep_dist_score


def composite(pred_x, gt_x, fs):
    """
    Compute composite objective speech quality scores.

    Args:
        pred_x (np.array): Audio signal to be evaluated signal.
        gt_x (np.array): Ground truth (clean) signal.
        fs (int): Sampling rate in Hz.

    Returns:
        tuple: (Csig, Cbak, Covl) composite scores.
    """
    composite_score = pysepm.composite(
        clean_speech=gt_x,
        processed_speech=pred_x,
        fs=fs,
    )
    return composite_score


def csii(pred_x, gt_x, fs):
    """
    Compute the Coherence Speech Intelligibility Index (CSII).

    Args:
        pred_x (np.array): Audio signal to be evaluated signal.
        gt_x (np.array): Ground truth (clean) signal.
        fs (int): Sampling rate in Hz.

    Returns:
        tuple: CSII scores for high, mid, and low frequencies.
    """
    csii_score = pysepm.csii(
        clean_speech=gt_x,
        processed_speech=pred_x,
        sample_rate=fs,
    )
    return csii_score


def ncm(pred_x, gt_x, fs):
    """
    Compute the Normalized Covariance Measure (NCM).

    Args:
        pred_x (np.array): Audio signal to be evaluated signal.
        gt_x (np.array): Ground truth (clean) signal.
        fs (int): Sampling rate in Hz.

    Returns:
        float: NCM score.
    """
    ncm_score = pysepm.ncm(
        clean_speech=gt_x,
        processed_speech=pred_x,
        fs=fs,
    )
    return ncm_score


def pysepm_metric(pred_x, gt_x, fs, frame_len=0.03, overlap=0.75):
    if pysepm is None:
        raise ImportError(
            "pysepm is not installed. Please use `tools/install_pysepm.sh` to install"
        )
    fwsegsnr_score = fwsegsnr(pred_x, gt_x, fs, frame_len, overlap)
    llr_score = llr(pred_x, gt_x, fs, frame_len, overlap)
    wss_score = wss(pred_x, gt_x, fs, frame_len, overlap)
    cep_dist_score = cd(pred_x, gt_x, fs)

    if fs == 8000:
        composite_score = composite(pred_x, gt_x, 8000)
        ncm_score = ncm(pred_x, gt_x, 8000)
    elif fs < 16000:
        logging.info("not support fs {}, resample to 8khz".format(fs))
        new_gt_x = resample_audio(gt_x, fs, 8000)
        new_pred_x = resample_audio(pred_x, fs, 8000)
        composite_score = composite(new_pred_x, new_gt_x, 8000)
        ncm_score = ncm(new_pred_x, new_gt_x, 8000)
    elif fs == 16000:
        composite_score = composite(pred_x, gt_x, 16000)
        ncm_score = ncm(pred_x, gt_x, 16000)
    else:
        logging.info("not support fs {}, resample to 16khz".format(fs))
        new_gt_x = resample_audio(gt_x, fs, 16000)
        new_pred_x = resample_audio(pred_x, fs, 16000)
        composite_score = composite(new_pred_x, new_gt_x, 16000)
        ncm_score = ncm(new_pred_x, new_gt_x, 16000)

    csii_score = csii(pred_x, gt_x, fs)

    return {
        "pysepm_fwsegsnr": fwsegsnr_score,
        "pysepm_llr": llr_score,
        "pysepm_wss": wss_score,
        "pysepm_cd": cep_dist_score,
        "pysepm_c_sig": composite_score[0],
        "pysepm_c_bak": composite_score[1],
        "pysepm_c_ovl": composite_score[2],
        "pysepm_csii_high": csii_score[0],
        "pysepm_csii_mid": csii_score[1],
        "pysepm_csii_low": csii_score[2],
        "pysepm_ncm": ncm_score,
    }


class PysepmMetric(BaseMetric):
    """Composite pysepm reference-based speech quality metrics."""

    def _setup(self):
        if pysepm is None:
            raise ImportError(
                "pysepm is not installed. "
                "Please use `tools/install_pysepm.sh` to install"
            )
        self.frame_len = self.config.get("frame_len", 0.03)
        self.overlap = self.config.get("overlap", 0.75)

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("Predicted signal must be provided")
        if references is None:
            raise ValueError("Reference signal must be provided")
        fs = metadata.get("sample_rate", 16000) if metadata else 16000
        return pysepm_metric(
            np.asarray(predictions),
            np.asarray(references),
            fs,
            frame_len=self.frame_len,
            overlap=self.overlap,
        )

    def get_metadata(self):
        return _pysepm_metadata()


def _pysepm_metadata():
    return MetricMetadata(
        name="pysepm",
        category=MetricCategory.DEPENDENT,
        metric_type=MetricType.DICT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=False,
        auto_install=False,
        dependencies=["pysepm", "librosa", "numpy"],
        description="pysepm composite reference-based speech quality metrics",
        implementation_source="https://github.com/schmiph2/pysepm",
    )


def register_pysepm_metric(registry):
    """Register pysepm metrics with the registry."""
    registry.register(
        PysepmMetric,
        _pysepm_metadata(),
        aliases=["pysepm_metric"],
    )


if __name__ == "__main__":

    a = np.random.random(16000)
    b = np.random.random(16000)
    metric = PysepmMetric()
    score = metric.compute(a, b, metadata={"sample_rate": 16000})
    print(score)

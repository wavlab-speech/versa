#!/usr/bin/env python3
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import librosa
import logging

logger = logging.getLogger(__name__)
import numpy as np


try:
    import pysepm  # Import the pysepm package for speech quality metrics
except ImportError:
    logger.info(
        "pysepm is not installed. Please use `tools/install_pysepm.sh` to install"
    )
    pysepm = None


def fwsegsnr(pred_x, gt_x, fs, frame_len=0.03, overlap=0.75):
    """
    Compute the Frequency-Weighted Segmental SNR (fwsegSNR) between predicted and ground truth signals.

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
    Compute the Weighted Spectral Slope (WSS) measure between predicted and ground truth signals.

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
        clean_speech=gt_x, processed_speech=pred_x, fs=fs, frameLen=0.03, overlap=0.75
    )
    return cep_dist_score


def composite(pred_x, gt_x, fs):
    """
    Compute the composite objective measure scores (c_sig, c_bak, c_ovl) for speech quality.

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
    Compute the Coherence Speech Intelligibility Index (CSII) between predicted and ground truth signals.

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
    Compute the Normalized Covariance Measure (NCM) between predicted and ground truth signals.

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
            # Error message if pysepm is not installed
        )
    fwsegsnr_score = fwsegsnr(pred_x, gt_x, fs, frame_len, overlap)
    llr_score = llr(pred_x, gt_x, fs, frame_len, overlap)
    wss_score = wss(pred_x, gt_x, fs, frame_len, overlap)
    cep_dist_score = cd(pred_x, gt_x, fs)

    if fs == 8000:
        composite_score = composite(pred_x, gt_x, fs)
    elif fs < 16000:
        logging.info("not support fs {}, resample to 8khz".format(fs))
        new_gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=8000)
        new_pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=8000)
    elif fs == 16000:
        composite_score = composite(pred_x, gt_x, fs)
    else:
        logging.info("not support fs {}, resample to 16khz".format(fs))
        new_gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=16000)
        new_pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)
        composite_score = composite(pred_x, gt_x, fs)

    csii_score = csii(pred_x, gt_x, fs)
    ncm_score = ncm(pred_x, gt_x, fs)

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


if __name__ == "__main__":

    a = np.random.random(16000)
    b = np.random.random(16000)
    score = pysepm_metric(a, b, 16000)
    print(score)

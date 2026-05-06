#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
# Adapted/Inspired by ESPnet/S3PRL-VC from Wen-Chin Huang and Tomoki Hayashi
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import numpy as np

from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType

try:
    import pysptk
    import pyworld as pw
    import scipy
    from fastdtw import fastdtw
    from scipy.signal import firwin, lfilter
except ImportError:
    pysptk = None
    pw = None
    scipy = None
    fastdtw = None
    firwin = None
    lfilter = None


def _ensure_mcd_f0_dependencies():
    if any(
        dependency is None
        for dependency in (pysptk, pw, scipy, fastdtw, firwin, lfilter)
    ):
        raise ImportError(
            "mcd_f0 requires pysptk, pyworld, scipy, and fastdtw. "
            "Please install these dependencies and retry"
        )


def low_cut_filter(x, fs, cutoff=70):
    """Function to apply low cut filter

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter

    Return:
        (ndarray): Low cut filtered waveform sequence
    """

    _ensure_mcd_f0_dependencies()
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def spc2npow(spectrogram):
    """Calculate normalized power sequence from spectrogram

    Parameters
    ----------
    spectrogram : array, shape (T, `fftlen / 2 + 1`)
        Array of spectrum envelope

    Return
    ------
    npow : array, shape (`T`, `1`)
        Normalized power sequence

    """

    # frame based processing
    npow = np.apply_along_axis(_spvec2pow, 1, spectrogram)

    meanpow = np.mean(npow)
    npow = 10.0 * np.log10(npow / meanpow)

    return npow


def _spvec2pow(specvec):
    """Convert a spectrum envelope into a power

    Parameters
    ----------
    specvec : vector, shape (`fftlen / 2 + 1`)
        Vector of specturm envelope |H(w)|^2

    Return
    ------
    power : scala,
        Power of a frame

    """

    # set FFT length
    fftl2 = len(specvec) - 1
    fftl = fftl2 * 2

    # specvec is not amplitude spectral |H(w)| but power spectral |H(w)|^2
    power = specvec[0] + specvec[fftl2]
    for k in range(1, fftl2):
        power += 2.0 * specvec[k]
    power /= fftl

    return power


def extfrm(data, npow, power_threshold=-20):
    """Extract frame over the power threshold

    Parameters
    ----------
    data: array, shape (`T`, `dim`)
        Array of input data
    npow : array, shape (`T`)
        Vector of normalized power sequence.
    power_threshold : float, optional
        Value of power threshold [dB]
        Default set to -20

    Returns
    -------
    data: array, shape (`T_ext`, `dim`)
        Remaining data after extracting frame
        `T_ext` <= `T`

    """

    T = data.shape[0]
    if T != len(npow):
        raise ("Length of two vectors is different.")

    valid_index = np.where(npow > power_threshold)
    extdata = data[valid_index]
    assert extdata.shape[0] <= T

    return extdata


def world_extract(
    x,
    fs,
    f0min,
    f0max,
    mcep_shift=5,
    mcep_fftl=1024,
    mcep_dim=39,
    mcep_alpha=0.466,
    filter_cutoff=70,
):
    _ensure_mcd_f0_dependencies()
    # scale from [-1, 1] to [-32768, 32767]
    x = x * np.iinfo(np.int16).max

    if x.ndim > 1:
        x = x[:, 0]
        logging.warning(
            "detect multi-channel data for mcd-f0 caluclation, use first channel"
        )

    x = np.array(x, dtype=np.float64)
    x = low_cut_filter(x, fs, cutoff=filter_cutoff)

    # extract features
    f0, time_axis = pw.harvest(
        x.astype(np.double), fs, f0_floor=f0min, f0_ceil=f0max, frame_period=mcep_shift
    )
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=mcep_fftl)
    ap = pw.d4c(x, f0, time_axis, fs, fft_size=mcep_fftl)
    mcep = pysptk.sp2mc(sp, mcep_dim, mcep_alpha)
    npow = spc2npow(sp)

    return {
        "sp": sp,
        "mcep": mcep,
        "ap": ap,
        "f0": f0,
        "npow": npow,
    }


def mcd_f0(
    pred_x,
    gt_x,
    fs,
    f0min,
    f0max,
    mcep_shift=5,
    mcep_fftl=1024,
    mcep_dim=39,
    mcep_alpha=0.466,
    seq_mismatch_tolerance=0.1,
    power_threshold=-20,
    dtw=False,
):
    _ensure_mcd_f0_dependencies()

    pred_feats = world_extract(
        pred_x, fs, f0min, f0max, mcep_shift, mcep_fftl, mcep_dim, mcep_alpha
    )
    gt_feats = world_extract(
        gt_x, fs, f0min, f0max, mcep_shift, mcep_fftl, mcep_dim, mcep_alpha
    )

    if dtw:
        # VAD & DTW based on power
        pred_mcep_nonsil_pow = extfrm(
            pred_feats["mcep"], pred_feats["npow"], power_threshold=power_threshold
        )
        gt_mcep_nonsil_pow = extfrm(
            gt_feats["mcep"], gt_feats["npow"], power_threshold=power_threshold
        )
        _, path = fastdtw(
            pred_mcep_nonsil_pow,
            gt_mcep_nonsil_pow,
            dist=scipy.spatial.distance.euclidean,
        )
        twf_pow = np.array(path).T

        # MCD using power-based DTW
        pred_mcep_dtw_pow = pred_mcep_nonsil_pow[twf_pow[0]]
        gt_mcep_dtw_pow = gt_mcep_nonsil_pow[twf_pow[1]]
        diff2sum = np.sum((pred_mcep_dtw_pow - gt_mcep_dtw_pow) ** 2, 1)
        mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)

        # VAD & DTW based on f0
        gt_nonsil_f0_idx = np.where(gt_feats["f0"] > 0)[0]
        pred_nonsil_f0_idx = np.where(pred_feats["f0"] > 0)[0]
        try:
            gt_mcep_nonsil_f0 = gt_feats["mcep"][gt_nonsil_f0_idx]
            pred_mcep_nonsil_f0 = pred_feats["mcep"][pred_nonsil_f0_idx]
            _, path = fastdtw(
                pred_mcep_nonsil_f0,
                gt_mcep_nonsil_f0,
                dist=scipy.spatial.distance.euclidean,
            )
            twf_f0 = np.array(path).T

            # f0RMSE, f0CORR using f0-based DTW
            pred_f0_dtw = pred_feats["f0"][pred_nonsil_f0_idx][twf_f0[0]]
            gt_f0_dtw = gt_feats["f0"][gt_nonsil_f0_idx][twf_f0[1]]
            f0rmse = np.sqrt(np.mean((pred_f0_dtw - gt_f0_dtw) ** 2))
            f0corr = scipy.stats.pearsonr(pred_f0_dtw, gt_f0_dtw)[0]
        except ValueError:
            logging.warning(
                "No nonzero f0 is found. Skip f0rmse f0corr computation and "
                "set them to NaN. This might due to unconverge training. "
                "Please tune the training time and hypers."
            )
            f0rmse = np.nan
            f0corr = np.nan

    else:
        # Use shorter sequence
        pred_seq_len = len(pred_feats["f0"])
        gt_seq_len = len(gt_feats["f0"])
        min_len = min(pred_seq_len, gt_seq_len)
        mismatch_ratio = (pred_seq_len + gt_seq_len - 2 * min_len) / (
            pred_seq_len + gt_seq_len
        )
        assert mismatch_ratio < seq_mismatch_tolerance, (
            "two input sequence mismatch ratio over threshold "
            f"{seq_mismatch_tolerance}"
        )
        diff2sum = np.sum(
            (pred_feats["mcep"][:min_len] - gt_feats["mcep"][:min_len]) ** 2, 1
        )
        mcd = np.mean(10 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)
        f0rmse = np.sqrt(
            np.mean((pred_feats["f0"][:min_len] - gt_feats["f0"][:min_len]) ** 2)
        )
        f0corr = scipy.stats.pearsonr(
            pred_feats["f0"][:min_len], gt_feats["f0"][:min_len]
        )[0]

    return {
        "mcd": mcd,
        "f0rmse": f0rmse,
        "f0corr": f0corr,
    }


class McdF0Metric(BaseMetric):
    """Mel cepstral distortion and F0 metrics."""

    def _setup(self):
        _ensure_mcd_f0_dependencies()
        self.f0min = self.config.get("f0min", 40)
        self.f0max = self.config.get("f0max", 800)
        self.mcep_shift = self.config.get("mcep_shift", 5)
        self.mcep_fftl = self.config.get("mcep_fftl", 1024)
        self.mcep_dim = self.config.get("mcep_dim", 39)
        self.mcep_alpha = self.config.get("mcep_alpha", 0.466)
        self.seq_mismatch_tolerance = self.config.get("seq_mismatch_tolerance", 0.1)
        self.power_threshold = self.config.get("power_threshold", -20)
        self.dtw = self.config.get("dtw", False)

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("Predicted signal must be provided")
        if references is None:
            raise ValueError("Reference signal must be provided")

        fs = metadata.get("sample_rate", 16000) if metadata else 16000
        return mcd_f0(
            np.asarray(predictions),
            np.asarray(references),
            fs,
            self.f0min,
            self.f0max,
            mcep_shift=self.mcep_shift,
            mcep_fftl=self.mcep_fftl,
            mcep_dim=self.mcep_dim,
            mcep_alpha=self.mcep_alpha,
            seq_mismatch_tolerance=self.seq_mismatch_tolerance,
            power_threshold=self.power_threshold,
            dtw=self.dtw,
        )

    def get_metadata(self):
        return _mcd_f0_metadata()


def _mcd_f0_metadata():
    return MetricMetadata(
        name="mcd_f0",
        category=MetricCategory.DEPENDENT,
        metric_type=MetricType.DICT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=False,
        auto_install=False,
        dependencies=["pysptk", "pyworld", "scipy", "fastdtw", "numpy"],
        description="Mel cepstral distortion, F0 RMSE, and F0 correlation",
        paper_reference="https://ieeexplore.ieee.org/document/407206",
        implementation_source=(
            "https://github.com/espnet/espnet and "
            "https://github.com/unilight/s3prl-vc"
        ),
    )


def register_mcd_f0_metric(registry):
    """Register MCD/F0 metrics with the registry."""
    registry.register(
        McdF0Metric,
        _mcd_f0_metadata(),
        aliases=["mcd", "mcd_f0_metric"],
    )


# debug code
if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    metric = McdF0Metric({"dtw": True, "f0min": 1, "f0max": 8000})
    print("metrics: {}".format(metric.compute(a, b, metadata={"sample_rate": 16000})))

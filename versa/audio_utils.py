#!/usr/bin/env python3

# Copyright 2026 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Small audio helpers shared by metric wrappers."""

from math import gcd

from scipy.signal import resample_poly


def resample_audio(audio, orig_sr, target_sr):
    """Resample 1-D audio without importing librosa's numba-heavy audio module."""
    if orig_sr == target_sr:
        return audio
    divisor = gcd(int(orig_sr), int(target_sr))
    return resample_poly(audio, int(target_sr) // divisor, int(orig_sr) // divisor)

#!/usr/bin/env python3

# Copyright 2025 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import fnmatch
import logging
import os
import numpy as np
from typing import Dict, List
import soundfile as sf

def find_files(
    root_dir: str, query: List[str] = ["*.flac", "*.wav"], include_root_dir: bool = True
) -> Dict[str, str]:
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (List[str]): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        Dict[str]: List of found filenames.

    """
    files = {}
    for root, _, filenames in os.walk(root_dir, followlinks=True):
        for q in query:
            for filename in fnmatch.filter(filenames, q):
                value = os.path.join(root, filename)
                if not include_root_dir:
                    value = value.replace(root_dir + "/", "")
                files[filename] = value
    return files


def check_all_same(array):
    try:
        return np.all(array == array[0])
    except IndexError:
        logging.warning("Detect an empty audio")
        return True


def load_audio(info, io):
    if io == "kaldi":
        gen_sr, gen_wav = info
    elif io == "soundfile" or io == "dir":
        gen_wav, gen_sr = sf.read(info)
    else:
        raise NotImplementedError(f"Unknown io type: {io}")
    return gen_sr, gen_wav


def wav_normalize(wave_array):
    if wave_array.ndim > 1:
        wave_array = wave_array[:, 0]
        logging.warning(
            "detect multi-channel data for mcd-f0 caluclation, use first channel"
        )
    if wave_array.dtype != np.int16:
        return np.ascontiguousarray(copy.deepcopy(wave_array.astype(np.float64)))
    # Convert the integer samples to floating-point numbers
    data_float = wave_array.astype(np.float64)

    # Normalize the floating-point numbers to the range [-1.0, 1.0]
    max_int16 = np.iinfo(np.int16).max
    normalized_data = data_float / max_int16
    return np.ascontiguousarray(copy.deepcopy(normalized_data))


def check_minimum_length(length, key_info):
    if "stoi" in key_info:
        # NOTE(jiatong): explicitly 0.256s as in https://github.com/mpariente/pystoi/pull/24
        if length < 0.3:
            return False
    if "pesq" in key_info:
        # NOTE(jiatong): check https://github.com/ludlows/PESQ/blob/master/pesq/cypesq.pyx#L37-L46
        if length < 0.25:
            return False
    if "visqol" in key_info:
        # NOTE(jiatong): check https://github.com/google/visqol/blob/master/src/image_patch_creator.cc#L50-L72
        if length < 1.0:
            return False
    if "sheet" in key_info:
        # NOTE(jiatong): check https://github.com/unilight/sheet/blob/main/hubconf.py#L13-L15
        if length < 0.065:
            return False
    if "squim_ref" in key_info or "squim_no_ref" in key_info:
        # NOTE(jiatong): a fix related to kernel size
        if length < 0.1:
            return False
    return True



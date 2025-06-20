import os
import scipy
import librosa

import numpy as np
import onnxruntime as ort
from enum import Enum
import logging


__all__ = ["SigMOS", "Version"]


class Version(Enum):
    V1 = "v1"  # 15.10.2023

# Adapted from https://github.com/microsoft/SIG-Challenge/blob/main/ICASSP2024/sigmos/sigmos.py
class SigMOS:
    '''
    MOS Estimator for the P.804 standard.
    See https://arxiv.org/pdf/2309.07385.pdf
    '''
    def __init__(self, model_dir, model_version=Version.V1):
        assert model_version in [v for v in Version]

        model_path_history = {
            Version.V1: os.path.join(model_dir, 'model-sigmos_1697718653_41d092e8-epo-200.onnx')
        }

        self.sampling_rate = 48_000
        self.resample_type = 'fft'
        self.model_version = model_version

        # STFT params
        self.dft_size = 960
        self.frame_size = 480
        self.window_length = 960
        self.window = np.sqrt(np.hanning(int(self.window_length) + 1)[:-1]).astype(np.float32)

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(model_path_history[model_version], options)

    def stft(self, signal):
        last_frame = len(signal) % self.frame_size
        if last_frame == 0:
            last_frame = self.frame_size

        padded_signal = np.pad(signal, ((self.window_length - self.frame_size, self.window_length - last_frame),))
        frames = librosa.util.frame(padded_signal, frame_length=len(self.window), hop_length=self.frame_size, axis=0)
        spec = scipy.fft.rfft(frames * self.window, n=self.dft_size)
        return spec.astype(np.complex64)

    @staticmethod
    def compressed_mag_complex(x: np.ndarray, compress_factor=0.3):
        x = x.view(np.float32).reshape(x.shape + (2,)).swapaxes(-1, -2)
        x2 = np.maximum((x * x).sum(axis=-2, keepdims=True), 1e-12)
        if compress_factor == 1:
            mag = np.sqrt(x2)
        else:
            x = np.power(x2, (compress_factor - 1) / 2) * x
            mag = np.power(x2, compress_factor / 2)

        features = np.concatenate((mag, x), axis=-2)
        features = np.transpose(features, (1, 0, 2))
        return np.expand_dims(features, 0)

    def run(self, audio: np.ndarray, sr=None):
        if sr is not None and sr != self.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate, res_type=self.resample_type)
            logging.info(f"Audio file resampled from {sr} to {self.sampling_rate}!")

        features = self.stft(audio)
        features = self.compressed_mag_complex(features)

        onnx_inputs = {inp.name: features for inp in self.session.get_inputs()}
        output = self.session.run(None, onnx_inputs)[0][0]

        result = {
            'SIGMOS_COL': float(output[0]), 'SIGMOS_DISC': float(output[1]), 'SIGMOS_LOUD': float(output[2]),
            'SIGMOS_REVERB': float(output[4]), 'SIGMOS_SIG': float(output[5]),
            'SIGMOS_OVRL': float(output[6])
        }
        return result

def sigmos_setup(model_dir='./versa_cache/sigmos_model'):

    # Permalink to the onnx file:
    # https://github.com/microsoft/SIG-Challenge/blob/bf4525153b6ed998f19d9e79ff1fd00f55dec42b/ICASSP2024/sigmos/model-sigmos_1697718653_41d092e8-epo-200.onnx

    # Check if the model directory exists
    if not os.path.exists(model_dir):
        # If the directory does not exist, create it
        os.makedirs(model_dir, exist_ok=True)

    # Check if the model file already exists
    # If it does not exist, download it
    if not os.path.exists(os.path.join(model_dir, 'model-sigmos_1697718653_41d092e8-epo-200.onnx')):
        logging.info(f"Model file not found in {model_dir}. Downloading the model file...") 
        # Download the model file from the web url
        import requests
        model_url = "https://github.com/microsoft/SIG-Challenge/blob/bf4525153b6ed998f19d9e79ff1fd00f55dec42b/ICASSP2024/sigmos/model-sigmos_1697718653_41d092e8-epo-200.onnx"
        model_file = os.path.join(model_dir, 'model-sigmos_1697718653_41d092e8-epo-200.onnx')
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_file, 'wb') as f:
                f.write(response.content)
        else:
            raise RuntimeError(f"Failed to download the model file from {model_url}. Status code: {response.status_code}")
    

    sigmos_estimator = SigMOS(model_dir=model_dir)
    return sigmos_estimator


def sigmos_calculate(model, pred_x, gen_sr):
    """
    Reference:
    https://github.com/microsoft/SIG-Challenge/blob/main/ICASSP2024/sigmos/sigmos.py
    
    """

    result = model.run(pred_x, sr=gen_sr)

    return result



if __name__ == '__main__':
    ''' 
        Sample code to run the SigMOS estimator. 
        V1 (current model) is an alpha version and should be used in accordance.
    '''
    # model_dir = r"."
    sigmos_estimator = SigMOS()

    # input data must have sr=48kHz, otherwise please specify the sr (it will be resampled to 48kHz internally)
    sampling_rate = 48_000
    dummy_data = np.random.rand(5 * sampling_rate)
    dummy_result = sigmos_estimator.run(dummy_data, sr=sampling_rate)
    print(dummy_result)
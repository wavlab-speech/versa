#!/usr/bin/env python3

# Copyright 2023 Takaaki Saeki
# Copyright 2024 Jiatong Shi
# Copyright 2025 Jionghao Han
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

logger = logging.getLogger(__name__)

import librosa
import numpy as np
import torch
import requests
from pathlib import Path
from typing import Optional

try:
    import utmosv2
    from utmosv2.dataset.multi_spec import process_audio_only_versa
except ImportError:
    logger.info(
        "utmosv2 is not installed, please install via `tools/install_utmosv2.sh`"
    )
    utmosv2 = None


def pseudo_mos_setup(
    predictor_types, predictor_args, cache_dir="versa_cache", use_gpu=False
):
    # Supported predictor types: utmos, dnsmos, aecmos, plcmos
    # Predictor args: predictor specific args
    predictor_dict = {}
    predictor_fs = {}
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    # first import utmos to resolve cross-import from the same model
    if "utmos" in predictor_types:
        torch.hub.set_dir(cache_dir)
        utmos = torch.hub.load("ftshijt/SpeechMOS:main", "utmos22_strong").to(device)
        predictor_dict["utmos"] = utmos.float()
        predictor_fs["utmos"] = 16000
    if "utmosv2" in predictor_types:
        if utmosv2 is None:
            raise RuntimeError(
                "utmosv2 is not installed. Please follow `tools/install_utmosv2.sh` to install"
            )
        # NOTE(jiatong): if you have an error of `_pickle.UnpicklingError: invalid load key, 'v'.`
        # It is likely that you did not have `git lfs` properly setup. Please check
        # https://github.com/sarulab-speech/UTMOSv2?tab=readme-ov-file#---quick-prediction--------
        utmos_v2 = utmosv2.create_model(pretrained=True)
        # _cfg = importlib.import_module(f"utmosv2.config.fusion_stage3")
        # cfg = SimpleNamespace(
        #     **{k: v for k, v in _cfg.__dict__.items() if not k.startswith("__")}
        # )
        # utmosv2._settings.configure_execution(cfg)
        predictor_dict["utmosv2"] = utmos_v2.to(device)
        predictor_fs["utmosv2"] = 16000

    if (
        "aecmos" in predictor_types
        or "dnsmos" in predictor_types
        or "plcmos" in predictor_types
    ):
        try:
            import onnxruntime  # NOTE(jiatong): a requirement of aecmos but not in requirements
            from speechmos import dnsmos, plcmos
        except ImportError:
            raise ImportError(
                "Please install speechmos for dnsmos, and plcmos: pip install speechmos onnxruntime"
            )

    for predictor in predictor_types:
        if predictor == "dnsmos":
            predictor_dict["dnsmos"] = dnsmos
            if "dnsmos" not in predictor_args:
                predictor_fs["dnsmos"] = 16000
            else:
                predictor_fs["dnsmos"] = predictor_args["dnsmos"]["fs"]
        elif predictor == "plcmos":
            predictor_dict["plcmos"] = plcmos
            if "plcmos" not in predictor_args:
                predictor_fs["plcmos"] = 16000
            else:
                predictor_fs["plcmos"] = predictor_args["plcmos"]["fs"]
        elif predictor == "utmos" or predictor == "utmosv2":
            continue  # already initialized
        elif predictor == "singmos":
            torch.hub.set_dir(cache_dir)
            singmos = torch.hub.load(
                "South-Twilight/SingMOS:v0.2.0", "singing_ssl_mos", trust_repo=True
            ).to(device)
            predictor_dict["singmos"] = singmos
            predictor_fs["singmos"] = 16000
        elif predictor.startswith("dnsmos_pro_"):
            variant = predictor[len("dnsmos_pro_") :]
            model_path = Path(cache_dir) / f"dnsmos_pro_{variant}.pt"
            if not model_path.exists():
                url = f"https://github.com/fcumlin/DNSMOSPro/raw/refs/heads/main/runs/{variant.upper()}/model_best.pt"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f'Downloading: "{url}" to {model_path}')
                response = requests.get(url)
                with open(model_path, "wb") as f:
                    f.write(response.content)
            else:
                logger.info(f"Using cached model: {model_path}")
            predictor_dict[predictor] = torch.jit.load(model_path, map_location=device)
            predictor_fs[predictor] = 16000
        else:
            raise NotImplementedError("Not supported {}".format(predictor))

    return predictor_dict, predictor_fs


def pseudo_mos_metric(pred, fs, predictor_dict, predictor_fs, use_gpu=False):
    scores = {}
    for predictor in predictor_dict.keys():
        if predictor == "utmos":
            if fs != predictor_fs["utmos"]:
                pred_utmos = librosa.resample(
                    pred, orig_sr=fs, target_sr=predictor_fs["utmos"]
                )
            else:
                pred_utmos = pred
            pred_tensor = torch.from_numpy(pred_utmos).unsqueeze(0)
            if use_gpu:
                pred_tensor = pred_tensor.to("cuda")
            score = predictor_dict["utmos"](pred_tensor.float(), predictor_fs["utmos"])[
                0
            ].item()
            scores.update(utmos=score)

        elif predictor == "utmosv2":
            if fs != predictor_fs["utmosv2"]:
                pred_utmosv2 = librosa.resample(
                    pred, orig_sr=fs, target_sr=predictor_fs["utmosv2"]
                )
            else:
                pred_utmosv2 = pred

            if utmosv2 is not None:
                cfg = predictor_dict["utmosv2"].cfg
                spec_info = process_audio_only_versa(pred_utmosv2, cfg)
                spec_info = torch.tensor(spec_info).float().unsqueeze(0)

                # magic number of data types
                # defined at https://github.com/ftshijt/UTMOSv2/blob/main/utmosv2/dataset/_utils.py#L8
                data_type = np.zeros(10)
                # we use general version dataset label: sarulab (1)
                data_type[1] = 0
                d = torch.tensor(data_type, dtype=torch.float32).unsqueeze(0)
                if use_gpu:
                    spec_info = spec_info.to("cuda")
                    d = d.to("cuda")
            else:
                raise RuntimeError(
                    "utmosv2 is not installed. Use tools/install_utmosv2.sh to install."
                )

            pred_tensor = torch.from_numpy(pred_utmosv2).unsqueeze(0)
            if use_gpu:
                pred_tensor = pred_tensor.to("cuda")

            NUM_REPETITIONS = 5
            with torch.no_grad():
                score_info = []
                for i in range(NUM_REPETITIONS):
                    score_info.append(
                        predictor_dict["utmosv2"](pred_tensor.float(), spec_info, d)
                        .squeeze(1)
                        .cpu()
                        .numpy()[0]
                    )
            scores.update(utmosv2=sum(score_info) / NUM_REPETITIONS)

        elif predictor == "dnsmos":
            if fs != predictor_fs["dnsmos"]:
                pred_dnsmos = librosa.resample(
                    pred, orig_sr=fs, target_sr=predictor_fs["dnsmos"]
                )
                fs = predictor_fs["dnsmos"]
            else:
                pred_dnsmos = pred

            max_val = np.max(np.abs(pred_dnsmos))
            score = predictor_dict["dnsmos"].run(pred_dnsmos / max_val, sr=fs)

            scores.update(dns_overall=score["ovrl_mos"], dns_p808=score["p808_mos"])
        elif predictor == "plcmos":
            if fs != predictor_fs["plcmos"]:
                pred_plcmos = librosa.resample(
                    pred, orig_sr=fs, target_sr=predictor_fs["plcmos"]
                )
                fs = predictor_fs["plcmos"]
            else:
                pred_plcmos = pred

            max_val = np.max(np.abs(pred_plcmos))
            score = predictor_dict["plcmos"].run(pred_plcmos / max_val, sr=fs)
            scores.update(plcmos=score["plcmos"])
        elif predictor == "singmos":
            if fs != predictor_fs["singmos"]:
                pred_singmos = librosa.resample(
                    pred, orig_sr=fs, target_sr=predictor_fs["singmos"]
                )
            else:
                pred_singmos = pred
            pred_tensor = torch.from_numpy(pred_singmos).unsqueeze(0)
            length_tensor = torch.tensor([pred_tensor.size(1)]).int()
            if use_gpu:
                pred_tensor = pred_tensor.to("cuda")
                length_tensor = length_tensor.to("cuda")
            score = predictor_dict["singmos"](pred_tensor.float(), length_tensor)[
                0
            ].item()
            scores.update(singmos=score)
        elif predictor.startswith("dnsmos_pro_"):
            if fs != predictor_fs[predictor]:
                pred_dnsmos_pro = librosa.resample(
                    pred, orig_sr=fs, target_sr=predictor_fs[predictor]
                )
            else:
                pred_dnsmos_pro = pred

            def stft(
                samples: np.ndarray,
                win_length: int = 320,
                hop_length: int = 160,
                n_fft: int = 320,
                use_log: bool = True,
                use_magnitude: bool = True,
                n_mels: Optional[int] = None,
            ) -> np.ndarray:
                if use_log and not use_magnitude:
                    raise ValueError(
                        "Log is only available if the magnitude is to be computed."
                    )
                if n_mels is None:
                    spec = librosa.stft(
                        y=samples,
                        win_length=win_length,
                        hop_length=hop_length,
                        n_fft=n_fft,
                    )
                else:
                    spec = librosa.feature.melspectrogram(
                        y=samples,
                        win_length=win_length,
                        hop_length=hop_length,
                        n_fft=n_fft,
                        n_mels=n_mels,
                    )
                spec = spec.T
                if use_magnitude:
                    spec = np.abs(spec)
                if use_log:
                    spec = np.clip(spec, 10 ** (-7), 10**7)
                    spec = np.log10(spec)
                return spec

            spec = torch.FloatTensor(stft(pred_dnsmos_pro))
            if use_gpu:
                spec = spec.to("cuda")
            with torch.no_grad():
                prediction = predictor_dict[predictor](spec[None, None, ...])
            scores[predictor] = prediction[0, 0].item()
        else:
            raise NotImplementedError("Not supported {}".format(predictor))

    return scores


if __name__ == "__main__":
    a = np.random.random(16000)
    print(a)
    predictor_dict, predictor_fs = pseudo_mos_setup(
        [
            "utmos",
            "dnsmos",
            "plcmos",
            "singmos",
            "dnsmos_pro_bvcc",
            "dnsmos_pro_nisqa",
            "dnsmos_pro_vcc2018",
        ],
        predictor_args={
            "dnsmos": {"fs": 16000},
            "plcmos": {"fs": 16000},
        },
    )
    scores = pseudo_mos_metric(
        a, fs=16000, predictor_dict=predictor_dict, predictor_fs=predictor_fs
    )
    print("metrics: {}".format(scores))

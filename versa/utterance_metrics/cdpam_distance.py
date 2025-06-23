import torch
import librosa
import numpy as np
from functools import partial
import cdpam

TARGET_FS = 22050


def cdpam_model_setup(use_gpu=False):
    device = "cpu" if not use_gpu else "cuda"
    _original_torch_load = torch.load
    torch.load = partial(torch.load, weights_only=False)
    model = cdpam.CDPAM(dev=device)
    torch.load = _original_torch_load
    return model


def cdpam_metric(model, pred_x, gt_x, fs):
    if fs != TARGET_FS:
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=TARGET_FS)
        gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=TARGET_FS)
    pred_x = (torch.from_numpy(pred_x).unsqueeze(0) * 32768).round()
    gt_x = (torch.from_numpy(gt_x).unsqueeze(0) * 32768).round()
    dist = model.forward(gt_x, pred_x)
    return {"cdpam_distance": dist.detach().cpu().numpy().item()}


if __name__ == "__main__":
    a = np.random.random(22050)
    b = np.random.random(22050)
    model = cdpam_model_setup()
    print("metrics: {}".format(cdpam_metric(model, a, b, 22050)))

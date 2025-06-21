import torch
import torch.nn as nn
import librosa
import numpy as np
import urllib.request
import logging
import filelock
from pathlib import Path

TARGET_FS = 22050
MODEL_URL = "https://raw.githubusercontent.com/adrienchaton/PerceptualAudio_Pytorch/refs/heads/master/pretrained/dataset_combined_linear_tshrink.pth"


class lossnet(nn.Module):
    def __init__(self, nconv=14, nchan=32, dp=0.1, dist_act="no"):
        super(lossnet, self).__init__()
        self.nconv = nconv
        self.dist_act = dist_act
        self.convs = nn.ModuleList()
        self.chan_w = nn.ParameterList()
        for iconv in range(nconv):
            if iconv == 0:
                chin = 1
            else:
                chin = nchan
            if (iconv + 1) % 5 == 0:
                nchan = nchan * 2
            if iconv < nconv - 1:
                conv = [
                    nn.Conv1d(chin, nchan, 3, stride=2, padding=1),
                    nn.BatchNorm1d(nchan),
                    nn.LeakyReLU(),
                ]
                if dp != 0:
                    conv.append(nn.Dropout(p=dp))
            else:
                conv = [
                    nn.Conv1d(chin, nchan, 3, stride=1, padding=1),
                    nn.BatchNorm1d(nchan),
                    nn.LeakyReLU(),
                ]

            self.convs.append(nn.Sequential(*conv))
            self.chan_w.append(nn.Parameter(torch.randn(nchan), requires_grad=True))

        if dist_act == "sig":
            self.act = nn.Sigmoid()
        elif dist_act == "tanh":
            self.act = nn.Tanh()
        elif dist_act == "tshrink":
            self.act = nn.Tanhshrink()
        elif dist_act == "exp":
            self.act = None
        elif dist_act == "no":
            self.act = nn.Identity()
        else:
            self.act = None

    def forward(self, xref, xper):
        device = next(self.parameters()).device
        xref = xref.unsqueeze(1).to(device)
        xper = xper.unsqueeze(1).to(device)
        dist = 0
        for iconv in range(self.nconv):
            xref = self.convs[iconv](xref)
            xper = self.convs[iconv](xper)
            diff = (xref - xper).permute(0, 2, 1)
            wdiff = diff * self.chan_w[iconv]
            wdiff = (
                torch.sum(torch.abs(wdiff), dim=(1, 2)) / diff.shape[1] / diff.shape[2]
            )
            dist = dist + wdiff
        if self.dist_act == "exp":
            dist = torch.exp(torch.clamp(dist, max=20.0)) / (10**5)  # exp(20) ~ 4*10**8
        else:
            dist = self.act(dist)
        return dist


def dpam_model_setup(cache_dir="versa_cache", use_gpu=False):
    device = "cpu" if not use_gpu else "cuda"
    model_path = Path(cache_dir) / "dpam" / "dataset_combined_linear.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with filelock.FileLock(model_path.with_suffix(".lock")):
        if not model_path.exists():
            logging.info(f"Downloading model to {model_path}...")
            urllib.request.urlretrieve(MODEL_URL, model_path)
            logging.info("Download complete.")
    state = torch.load(model_path, map_location="cpu", weights_only=False)["state"]
    prefix = "model_dist."
    state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    model = lossnet(nconv=14, nchan=16, dp=0, dist_act="tshrink")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def dpam_metric(model, pred_x, gt_x, fs):
    if fs != TARGET_FS:
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=TARGET_FS)
        gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=TARGET_FS)
    pred_x = torch.from_numpy(pred_x).unsqueeze(0).float()
    gt_x = torch.from_numpy(gt_x).unsqueeze(0).float()
    dist = model(gt_x, pred_x)
    return {"dpam_distance": dist.detach().cpu().numpy().item()}


if __name__ == "__main__":
    a = np.random.random(22050)
    b = np.random.random(22050)
    model = dpam_model_setup()
    print("metrics: {}".format(dpam_metric(model, a, b, 22050)))

#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np

try:
    from pystoi import stoi
except ImportError:
    raise ImportError("Please install pystoi and retry: pip install stoi")


def stoi_metric(pred_x, gt_x, fs):
    if pred_x.shape[0] != gt_x.shape[0]:
        min_length = min(pred_x.shape[0], gt_x.shape[0])
        pred_x = pred_x[:min_length]
        gt_x = gt_x[:min_length]
    score = stoi(gt_x, pred_x, fs, extended=False)
    return {"stoi": score}


def estoi_metric(pred_x, gt_x, fs):                                                 
    if pred_x.shape[0] != gt_x.shape[0]:                                           
        min_length = min(pred_x.shape[0], gt_x.shape[0])                           
        pred_x = pred_x[:min_length]                                               
        gt_x = gt_x[:min_length]                                                   
    score = stoi(gt_x, pred_x, fs, extended=True)                                 
    return {"estoi": score}


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    scores = stoi_metric(a, b, 16000)
    print(scores)

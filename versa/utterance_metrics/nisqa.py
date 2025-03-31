#!/usr/bin/env python3

# Copyright 2025 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import torch

import versa.utterance_metrics.nisqa_utils.nisqa_lib as NL

def nisqa_model_setup():


    pass


def nisqa_metric(nisqa_model_path, pred_x, fs, use_gpu=False):
    """
    Placeholder for NISQA metric calculation.
    Currently, NISQA is not implemented.
    """

    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    # Check if the model path is provided
    if nisqa_model_path is None:
        raise ValueError("NISQA model path must be provided.")

    checkpoint = torch.load(nisqa_model_path, map_location="cpu")
    args = checkpoint.get("args", None)
    if args is None:
        raise ValueError("Model checkpoint does not contain the required arguments. Might due to a wrong checkpoint.")
    

    if args["model"] == "NISQA_DIM":
        args["dim"] = True
        args["csv_mos_train"] = None  # column names hardcoded for dim models
        args["csv_mos_val"] = None
    else:
        args["dim"] = False

    if args["model"] == "NISQA_DE":
        args["double_ended"] = True
    else:
        args["double_ended"] = False
        args["csv_ref"] = None

    # Load Model
    model_args = {
        "ms_seg_length": args["ms_seg_length"],
        "ms_n_mels": args["ms_n_mels"],
        "cnn_model": args["cnn_model"],
        "cnn_c_out_1": args["cnn_c_out_1"],
        "cnn_c_out_2": args["cnn_c_out_2"],
        "cnn_c_out_3": args["cnn_c_out_3"],
        "cnn_kernel_size": args["cnn_kernel_size"],
        "cnn_dropout": args["cnn_dropout"],
        "cnn_pool_1": args["cnn_pool_1"],
        "cnn_pool_2": args["cnn_pool_2"],
        "cnn_pool_3": args["cnn_pool_3"],
        "cnn_fc_out_h": args["cnn_fc_out_h"],
        "td": args["td"],
        "td_sa_d_model": args["td_sa_d_model"],
        "td_sa_nhead": args["td_sa_nhead"],
        "td_sa_pos_enc": args["td_sa_pos_enc"],
        "td_sa_num_layers": args["td_sa_num_layers"],
        "td_sa_h": args["td_sa_h"],
        "td_sa_dropout": args["td_sa_dropout"],
        "td_lstm_h": args["td_lstm_h"],
        "td_lstm_num_layers": args["td_lstm_num_layers"],
        "td_lstm_dropout": args["td_lstm_dropout"],
        "td_lstm_bidirectional": args["td_lstm_bidirectional"],
        "td_2": args["td_2"],
        "td_2_sa_d_model": args["td_2_sa_d_model"],
        "td_2_sa_nhead": args["td_2_sa_nhead"],
        "td_2_sa_pos_enc": args["td_2_sa_pos_enc"],
        "td_2_sa_num_layers": args["td_2_sa_num_layers"],
        "td_2_sa_h": args["td_2_sa_h"],
        "td_2_sa_dropout": args["td_2_sa_dropout"],
        "td_2_lstm_h": args["td_2_lstm_h"],
        "td_2_lstm_num_layers": args["td_2_lstm_num_layers"],
        "td_2_lstm_dropout": args["td_2_lstm_dropout"],
        "td_2_lstm_bidirectional": args["td_2_lstm_bidirectional"],
        "pool": args["pool"],
        "pool_att_h": args["pool_att_h"],
        "pool_att_dropout": args["pool_att_dropout"],
    }

    if args["double_ended"]:
        model_args.update(
            {
                "de_align": args["de_align"],
                "de_align_apply": args["de_align_apply"],
                "de_fuse_dim": args["de_fuse_dim"],
                "de_fuse": args["de_fuse"],
            }
        )

    if args["model"] == "NISQA":
        model = NL.NISQA(**model_args)
    elif args["model"] == "NISQA_DIM":
        model = NL.NISQA_DIM(**model_args)
    elif args["model"] == "NISQA_DE":
        model = NL.NISQA_DE(**model_args)
    else:
        raise NotImplementedError("Model not available")

    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    if missing_keys:
        print("[NISQA] missing_keys:")
        print(missing_keys)
    if unexpected_keys:
        print("[NISQA] unexpected_keys:")
        print(unexpected_keys)
    model.args = args
    model.device = device
    return model


if __name__ == "__main__":
    a = np.random.random(16000)
    fs = 16000
    try:
        nisqa_model = nisqa_model_setup()
        score = nisqa_metric(nisqa_model, a, fs)
        print("NISQA Score: {}".format(score))
    except NotImplementedError as e:
        print(e)
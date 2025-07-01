#!/usr/bin/env python3

# Copyright 2025 BoHao Su
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def metric_loader_setup(score_output_file):
    """
    Reads an scp-like file where each line is:
      utt_id <JSON-metrics>
    Returns a dict mapping utt_id → metrics dict.
    """
    data = {}
    with open(score_output_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, json_str = line.split(maxsplit=1)
            data[utt_id] = json.loads(json_str)
    return data


def load_interpreter_modules(interpreter_config, use_gpu):
    assert interpreter_config, "no interpreter function is provided"
    interpreter_modules = {}
    for config in interpreter_config:
        print(config, flush=True)
        if config["model_name"] == "Qwen/Qwen2.5-7B-Instruct":
            model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                torch_dtype="auto",
                device_map="auto" if use_gpu else None,
            )
            tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
            interpreter_modules[config["model_name"]] = {
                "args": {
                    "model": model,
                    "tokenizer": tokenizer,
                },
            }
        elif config["model_name"] == "mistralai/Mistral-7B-Instruct-v0.3":
            login(token=config["HF_TOKEN"])
            model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                torch_dtype="auto",
                device_map="auto" if use_gpu else None,
            )
            tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
            interpreter_modules[config["model_name"]] = {
                "args": {
                    "model": model,
                    "tokenizer": tokenizer,
                },
            }
        elif config["model_name"] == "meta-llama/Llama-3.1-8B-Instruct":
            login(token=config["HF_TOKEN"])
            pipe = pipeline(
                "text-generation",
                model=config["model_name"],
                torch_dtype=torch.bfloat16,
                device_map="auto" if use_gpu else None,
            )
            interpreter_modules[config["model_name"]] = {
                "args": {
                    "pipe": pipe,
                },
            }
        else:
            raise ValueError(f"Unsupported model_name: {config['model_name']}")
    return interpreter_modules

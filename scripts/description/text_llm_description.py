#!/usr/bin/env python3

# Copyright 2025 BoHao Su
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Module for text LLM description."""

import json

from tqdm.auto import tqdm


def create_template(metrics: dict) -> str:
    prompt = f"""
    ## Background
    You are a professional audio descriptor.\n\n
    ## Task instruction
    Describe the audio according to following predicted metrics and choose top-10 most important metrics according to your description.\n\n
    ## Output Format
    Provide the audio description, top-10 metric selection, reasoning for your selection, 
    You should respond in JSON format. First provide a one-sentence concise description for the audio in field ‘description‘, then your top-10 metrics selection in field ‘top-10 metrics‘ followed by a reason in the field 'reasoning'. For example, 
    ```
    {{"description": "<a one-sentence concise description for the audio>", "top-10 metrics": < your top-10 metrics selection>, "reasoning": <your reason of top-10 metrics selection>}} 
    ```
    ## Metrics:\n{metrics}\n\n
    ## Answer:\n
    """
    return prompt


def describe_all(metrics_dict: dict, model_name: str, modules: dict) -> list:
    """
    For each utt_id in metrics_dict, run the prompt through the appropriate
    module (either tokenizer+model or pipeline) and return a list of results.
    """
    results = []
    mod = modules[model_name]["args"]
    for utt_id, metrics in tqdm(
        metrics_dict.items(), desc=f"Describing with {model_name}"
    ):
        prompt = create_template(metrics)
        if "model" in mod:
            # Qwen or Mistral-style
            messages = [
                {
                    "role": "system",
                    "content": "You are a professional audio descriptor.",
                },
                {"role": "user", "content": prompt},
            ]
            text = mod["tokenizer"].apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = mod["tokenizer"]([text], return_tensors="pt").to(
                mod["model"].device
            )
            gen_ids = mod["model"].generate(**inputs, max_new_tokens=1024)
            # strip off prompt
            gen_ids = [out[len(inp) :] for inp, out in zip(inputs.input_ids, gen_ids)]
            response = mod["tokenizer"].batch_decode(gen_ids, skip_special_tokens=True)[
                0
            ]
        else:
            # llama pipeline
            messages = [
                {
                    "role": "system",
                    "content": "You are a professional audio descriptor.",
                },
                {"role": "user", "content": prompt},
            ]
            response = mod["pipe"](messages, max_new_tokens=1024)[0]["generated_text"][
                -1
            ]["content"]

        # clean & parse
        clean = response.strip().strip("```json").strip("```").replace("\n", " ")
        try:
            obj = json.loads(clean)
            obj["utt_id"] = utt_id
            results.append(obj)
        except json.JSONDecodeError:
            # fallback: store raw text
            results.append({"utt_id": utt_id, "error": "parse_failed", "raw": response})
    return results

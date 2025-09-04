# Speech Evaluation Interpreter

This tool loads utterance-level metrics and uses LLM interpreters to generate natural-language descriptions.


## Files
- `interpreter.py`: CLI entry point, loads config, metrics, runs interpreters, saves JSON.
- `interpreter_shared.py`: utilities for loading metrics and models.
- `text_llm_description.py`: **you implement** `describe_all(...)` to describe each utterance.


## Example Input

### scores.scp

```
utt_0001 {"SNR": 23.1, "WER": 0.08, "MOS": 4.2}
utt_0002 {"SNR": 12.7, "WER": 0.30, "MOS": 3.0}
```

### egs/interpreter.yaml
```yaml
interpreter_config:
  - model_name: "Qwen/Qwen2.5-7B-Instruct"
```

## Run

```bash
python interpreter.py \
  --config egs/interpreter.yaml \
  --score_output_file scores.scp \
  --output_file descriptions.json \
  --use_gpu False \
  --verbose 1
```
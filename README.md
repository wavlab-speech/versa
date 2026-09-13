<div align="left"><img src="assets/images/versa-light-char.png" width="550"/></div>

# VERSA: Versatile Evaluation of Speech and Audio

[![GitHub stars](https://img.shields.io/github/stars/wavlab-speech/versa?style=social)](https://github.com/wavlab-speech/versa/stargazers)
![CI](https://github.com/wavlab-speech/versa/actions/workflows/ci.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2412.17667-b31b1b.svg)](https://arxiv.org/abs/2412.17667)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

VERSA (Versatile Evaluation of Speech and Audio) is a comprehensive toolkit for evaluating speech and audio quality. It provides seamless access to over 90 evaluation/profiling metrics with 10x variants, enabling researchers and developers to assess audio quality through multiple dimensions.

## 🚨 Exciting News
- May 2026 - Completed the metric object-oriented migration cleanup, including
  registry-backed corpus FAD, individual FAD, and KID metrics.
- Sep 2025 - Add visualization and text LLM summarization supports for VERSA-v2.
- Jun 2025 - Update launch scripts for local machine to support multi-process/multi-gpu (automatic rank assignment) for VERSA.
- May 2025 – VERSA presented at NAACL 2025, showcasing its unified multi-metric evaluation framework for speech and audio ([🎥 Presentation Video](https://www.youtube.com/watch?v=e7TdOlzyJcE))
- Feb 2025 – Integrated support for Qwen2-Audio-based perceptual metrics, extending VERSA's capacity for LLM-informed audio quality profiling
- Dec 2024 – Official release of VERSA v1.0, featuring 90+ evaluation metrics and full integration with ESPnet and Slurm-based distributed evaluation

## 🚀 Features

- **Comprehensive**: 90+ metrics covering perceptual quality, intelligibility, and technical measurements (check [full metrics documentation](https://github.com/wavlab-speech/versa/blob/main/docs/supported_metrics.md) for a complete list)
- **Integrated**: Widely used in speech toolkits and challenges (check the [incomplete list of toolkits/challenges](https://github.com/wavlab-speech/versa/blob/main/docs/users.md) using versa)
- **Flexible**: Support for various input formats (file paths, SCP files, Kaldi-style ARKs)
- **Scalable**: Built-in support for distributed evaluation using Slurm, with resume support for interrupted scoring runs
- **Visualizable**: Interactive visualization with VERSA results (check [our visualization guideline](https://github.com/wavlab-speech/versa/blob/main/docs/visualization.md))

## 🔍 Interactive Demo

Try our interactive demo from the Interspeech 2024 Tutorial:
[Colab Demonstration](https://colab.research.google.com/drive/11c0vZxbSa8invMSfqM999tI3MnyAVsOp?usp=sharing)

## 📦 Installation

### Basic Installation

```bash
git clone https://github.com/wavlab-speech/versa.git
cd versa
pip install .
```

The base install keeps dependency resolution light and includes the shared scorer
runtime. Install optional groups for metrics that need larger model stacks or
external toolkits:

```bash
pip install ".[audio,text,ml]"
pip install ".[songeval]" # SongEval Python dependencies only
pip install ".[external]"  # Git/toolkit-backed metrics
pip install ".[dev]"       # tests, linting, and formatting
```

or alternatively, without cloning:

```bash
python -m pip install git+https://github.com/wavlab-speech/versa.git#egg=versa-speech-audio-toolkit --no-build-isolation
```
### Metric-Specific Dependencies

VERSA aligns with original APIs provided by algorithm developers rather than redistributing models. The base package does not install every optional metric backend by default.

For metrics marked without "x" in the "Auto-Install" column of our metrics tables, please use the installers provided in the `tools` directory.

Some real model-backed tests and metrics also need checkpoint assets that are
too large to keep in the package. Prepare those assets in a repo-visible cache
before running the full real-model checks:

```bash
PYTHON=python tools/setup_huggingface_cache.sh
```

This populates `versa_cache/huggingface` and
`versa_cache/discrete_speech_metrics`. To reuse an existing local Hugging Face
cache without network access, run:

```bash
SOURCE_HF_CACHE="$HOME/.cache/huggingface/hub" \
VERSA_HF_LOCAL_ONLY=1 \
PYTHON=python \
tools/setup_huggingface_cache.sh
```

### Installation Notes

Some optional metric backends emit warnings during setup or first use. ESPnet may
print a `flash_attn` warning when Flash Attention is not available; VERSA can
still run metrics that do not require that backend. FADTK is only needed for
FAD/KID-style metrics and can be installed with `tools/install_fadtk.sh` when
those metrics are selected.

SongEval is an optional, reference-free metric for full songs. Its upstream
toolkit predicts coherence, musicality, memorability, structural clarity, and
vocal naturalness on a 1--5 scale. Install its Python dependencies and pinned
upstream assets explicitly before scoring:

```bash
PYTHON=python tools/install_songeval.sh
python versa/bin/scorer.py \
    --score_config egs/separate_metrics/songeval.yaml \
    --pred path/to/generated_wav.scp \
    --output_file songeval.jsonl \
    --io soundfile \
    --use_gpu
```

On first use, VERSA downloads a pinned SongEval checkout into
`versa_cache/SongEval` and MuQ into `versa_cache/huggingface`. For a fully local
run, set `model_dir`, `muq_model`, and `offline: true` in the YAML.

MAPSS is an optional multi-source metric for source-separation systems. Its
backend supports Python 3.10--3.12 and constrains `transformers<4.53`, so install
it in a compatible environment instead of adding it to the VERSA base install:

```bash
PYTHON=python tools/install_mapss.sh
versa-score \
    --score_config egs/separate_metrics/mapss.yaml \
    --pred_sources estimates/source1.scp estimates/source2.scp \
    --gt_sources references/source1.scp references/source2.scp \
    --output_file mapss.jsonl \
    --io soundfile \
    --use_gpu
```

The SCP lists are positional: predicted source *i* must estimate reference
source *i*, and every list must contain the same mixture keys. VERSA writes the
diagnostic per-source means to JSONL and retains MAPSS's frame-level PS, PM, and
confidence tables under `versa_cache/mapss/results`. The frame-level outputs
should be retained for scientific reporting; the PS convenience mean is not the
paper's formal utterance-level pooling protocol.

If NLTK downloads fail with a certificate verification error, point Python at
the certificate bundle used by `certifi` before running the tests:

```bash
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```


## 🧪 Quick Testing

```bash
# Test dependency-light core functionality
python -m pytest test/test_metrics/test_definition.py

# Test specific metrics that require additional installation
python -m pytest test/test_metrics/test_{metric}.py

# Run real model-backed checks after preparing the visible model cache
VERSA_RUN_REAL_MODEL_TESTS=1 \
VERSA_HF_CACHE_DIR="$PWD/versa_cache/huggingface" \
VERSA_DISCRETE_SPEECH_CACHE_DIR="$PWD/versa_cache/discrete_speech_metrics" \
python -m pytest --import-mode=importlib test
```


## 🔧 Usage Examples

### Basic Usage

Use `--cache_folder` to place downloads for all metrics below one shareable
root. VERSA shares subdirectories for metrics using the same model backend
(such as Hugging Face, Whisper, ESPnet, and Torch Hub) and isolates other
metric-specific files. An explicit `cache_dir` in the score YAML takes
precedence:

```bash
python versa/bin/scorer.py \
    --score_config egs/speech_cpu.yaml \
    --pred test/test_samples/test2 \
    --gt test/test_samples/test1 \
    --output_file test_result \
    --io dir \
    --cache_folder /shared/versa_cache
```

```bash
# Direct usage with file paths
python versa/bin/scorer.py \
    --score_config egs/speech_cpu.yaml \
    --gt test/test_samples/test1 \
    --pred test/test_samples/test2 \
    --output_file test_result \
    --io dir

# With SCP-style input
python versa/bin/scorer.py \
    --score_config egs/speech_cpu.yaml \
    --gt test/test_samples/test1.scp \
    --pred test/test_samples/test2.scp \
    --output_file test_result \
    --io soundfile \
    --num_workers 4

# With Kaldi-ARK style input (compatible with ESPnet)
python versa/bin/scorer.py \
    --score_config egs/speech_cpu.yaml \
    --gt test/test_samples/test1.scp \
    --pred test/test_samples/test2.scp \
    --output_file test_result \
    --io kaldi
  
# Including text transcription information
python versa/bin/scorer.py \
    --score_config egs/separate_metrics/wer_tiny.yaml \
    --gt test/test_samples/test1.scp \
    --pred test/test_samples/test2.scp \
    --output_file test_result \
    --text test/test_samples/text \
    --io soundfile

# Resume an interrupted utterance-level scoring run
python versa/bin/scorer.py \
    --score_config egs/speech_cpu.yaml \
    --gt test/test_samples/test1.scp \
    --pred test/test_samples/test2.scp \
    --output_file test_result \
    --io soundfile \
    --resume

# Load and score one metric at a time to reduce peak GPU memory
python versa/bin/scorer.py \
    --score_config egs/speech_gpu.yaml \
    --gt test/test_samples/test1.scp \
    --pred test/test_samples/test2.scp \
    --output_file test_result \
    --io soundfile \
    --use_gpu True \
    --scoring_mode metric
```

`--resume` reads existing JSONL rows from `--output_file`, skips utterance keys
that have already been scored, and preserves their results. This is useful for
long-running evaluations that are interrupted or restarted. With
`--num_workers > 1`, newly computed rows are appended in input key order, while
existing rows keep their original positions in the file; the returned scores are
ordered by input key, but a resumed JSONL file may not be globally sorted by
input key.

`--num_workers` runs utterance-level CPU scoring in local worker processes while
preserving input key order in newly written JSONL output for non-resume runs.
GPU scoring, metric-oriented scoring (`--scoring_mode metric`), and
corpus/distributional metrics remain serial in this first implementation;
`--num_workers > 1` cannot be combined with `--use_gpu`, `--scoring_mode
metric`, or corpus-only configurations.

`--scoring_mode metric` loads and runs one metric at a time, then releases
metric resources before moving to the next metric. This can reduce peak GPU
memory when many model-backed metrics are configured together.

### Metric Discovery

Use the `versa-score` CLI to inspect available metrics and generate starter
configs without reading the full metrics table manually:

```bash
# List metrics registered in the current environment
versa-score --list-metrics

# Show metadata, dependencies, references, and aliases for one metric
versa-score --describe-metric pesq

# Print a recommended YAML score config for a task/device pair
versa-score --recommend-config --task tts --device gpu
```

`--list-metrics` also supports `--metric-category` and `--metric-type` filters,
for example:

```bash
versa-score --list-metrics --metric-category dependent --metric-type float
```

### Distributed Evaluation with Slurm

The launcher defaults to 4 CPUs and a 12-hour limit per job. It checks the
actual chunk count against `MAX_JOBS` (default 50) and the combined requested
CPU time against `MAX_TOTAL_CPU_HOURS` (default 5000) before submitting.
These limits apply to one invocation, not all jobs already running on the cluster.
Use a fresh score directory; existing split files are rejected. Submission requires
confirmation; pass `--yes` for unattended runs (resource checks still apply).

Override resources with `CPUS`, `MEM` (MB per CPU), `CPU_TIME`, and `GPU_TIME`.
Numeric settings must be integers from 1 to 999999999. Time limits accept Slurm's
finite positive time formats, including minutes, minutes:seconds, hours:minutes:seconds,
and days-hours[:minutes[:seconds]]. Seconds round up to a minute for the estimate.
Jobs request one task on one node; cluster allocation and billing policies may differ
from the requested CPU estimate. `CPU_OTHER_OPTS` and `GPU_OTHER_OPTS` accept
space-separated `--name=value` options for account, qos, constraint, reservation,
dependency, exclude, nodelist, mail-type, mail-user, and comment. Resource overrides,
arrays, and other implicit `SBATCH_*` settings are rejected to keep the estimate valid.

```bash
# Option 1: With ground truth speech
./launch_slurm.sh \
  <pred_speech_scp> \
  <gt_speech_scp> \
  <score_dir> \
  <split_job_num> 

# Option 2: Without ground truth speech
./launch_slurm.sh \
  <pred_speech_scp> \
  None \
  <score_dir> \
  <split_job_num>

# Aggregate results
cat <score_dir>/result/*.result.cpu.txt > <score_dir>/utt_result.cpu.txt
cat <score_dir>/result/*.result.gpu.txt > <score_dir>/utt_result.gpu.txt

# Visualize results
python scripts/show_result.py <score_dir>/utt_result.cpu.txt
python scripts/show_result.py <score_dir>/utt_result.gpu.txt 
```

Explore `egs/*.yaml` for configuration examples for different evaluation scenarios.

## 📊 Supported Metrics

VERSA organizes metrics into four categories:

1. **Independent Metrics** - Standalone metrics that don't require reference audio
2. **Dependent Metrics** - Metrics that compare predicted audio against reference audio
3. **Non-match Metrics** - Metrics that work with non-matching references or information from other modalities
4. **Distributional Metrics** - Metrics that evaluate statistical properties of audio collections

*See the [full metrics documentation](https://github.com/wavlab-speech/versa/blob/main/docs/supported_metrics.md) for a complete list with references.*

## 📝 Citation

If you use VERSA in your research, please cite our papers:

```bibtex
@inproceedings{shi2025versa,
title={{VERSA}: A Versatile Evaluation Toolkit for Speech, Audio, and Music},
author={Jiatong Shi and Hye-jin Shim and Jinchuan Tian and Siddhant Arora and Haibin Wu and Darius Petermann and Jia Qi Yip and You Zhang and Yuxun Tang and Wangyou Zhang and Dareen Safar Alharthi and Yichen Huang and Koichi Saito and Jionghao Han and Yiwen Zhao and Chris Donahue and Shinji Watanabe},
booktitle={2025 Annual Conference of the North American Chapter of the Association for Computational Linguistics -- System Demonstration Track},
year={2025},
url={https://openreview.net/forum?id=zU0hmbnyQm}
}

@inproceedings{shi2024versaversatileevaluationtoolkit,
  author={Shi, Jiatong and Tian, Jinchuan and Wu, Yihan and Jung, Jee-Weon and Yip, Jia Qi and Masuyama, Yoshiki and Chen, William and Wu, Yuning and Tang, Yuxun and Baali, Massa and Alharthi, Dareen and Zhang, Dong and Deng, Ruifan and Srivastava, Tejes and Wu, Haibin and Liu, Alexander and Raj, Bhiksha and Jin, Qin and Song, Ruihua and Watanabe, Shinji},
  booktitle={2024 IEEE Spoken Language Technology Workshop (SLT)}, 
  title={ESPnet-Codec: Comprehensive Training and Evaluation of Neural Codecs For Audio, Music, and Speech}, 
  year={2024},
  pages={562-569},
  keywords={Training;Measurement;Codecs;Speech coding;Conferences;Focusing;Neural codecs;codec evaluation},
  doi={10.1109/SLT61566.2024.10832289}
}
```

## 🙏 Acknowledgement

We sincerely thank all the authors of the open-source implementations listed in our [metrics documentation](https://github.com/wavlab-speech/versa/blob/main/docs/supported_metrics.md).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Please check the [contributing guideline](https://github.com/wavlab-speech/versa/blob/main/docs/contributing.md) first.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

# VERSA: Versatile Evaluation of Speech and Audio

[![GitHub stars](https://img.shields.io/github/stars/wavlab-speech/versa?style=social)](https://github.com/wavlab-speech/versa/stargazers)
[![PyPI version](https://badge.fury.io/py/versa.svg)](https://badge.fury.io/py/versa)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

VERSA (Versatile Evaluation of Speech and Audio) is a comprehensive toolkit for evaluating speech and audio quality. It provides seamless access to over 80 evaluation metrics with 10x variants, enabling researchers and developers to assess audio quality through multiple dimensions.

## 🚀 Features

- **Comprehensive**: 80+ metrics covering perceptual quality, intelligibility, and technical measurements
- **Integrated**: Tightly integrated with [ESPnet](https://github.com/espnet/espnet.git)
- **Flexible**: Support for various input formats (file paths, SCP files, Kaldi-style ARKs)
- **Scalable**: Built-in support for distributed evaluation using Slurm

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

### Metric-Specific Dependencies

VERSA aligns with original APIs provided by algorithm developers rather than redistributing models. The core package includes many metrics by default, but some require additional installation.

For metrics marked without "x" in the "Auto-Install" column of our metrics tables, please check the [installation guide](https://github.com/wavlab-speech/versa?tab=readme-ov-file#list-of-metrics) or use the installers provided in the `tools` directory.


## 🧪 Quick Testing

```bash
# Test core functionality
python versa/test/test_general.py

# Test specific metrics that require additional installation
python versa/test/test_{metric}.py
```


## 🔧 Usage Examples

### Basic Usage

```bash
# Direct usage with file paths
python versa/bin/scorer.py \
    --score_config egs/speech.yaml \
    --gt test/test_samples/test1 \
    --pred test/test_samples/test2 \
    --output_file test_result

# With SCP-style input
python versa/bin/scorer.py \
    --score_config egs/speech.yaml \
    --gt test/test_samples/test1.scp \
    --pred test/test_samples/test2.scp \
    --output_file test_result

# With Kaldi-ARK style input
python versa/bin/scorer.py \
    --score_config egs/speech.yaml \
    --gt test/test_samples/test1.scp \
    --pred test/test_samples/test2.scp \
    --output_file test_result \
    --io kaldi
  
# Including text transcription information
python versa/bin/scorer.py \
    --score_config egs/separate_metrics/wer.yaml \
    --gt test/test_samples/test1.scp \
    --pred test/test_samples/test2.scp \
    --output_file test_result \
    --text test/test_samples/text
```

### Distributed Evaluation with Slurm

```bash
# Option 1: With ground truth speech
./launch.sh \
  <pred_speech_scp> \
  <gt_speech_scp> \
  <score_dir> \
  <split_job_num> 

# Option 2: Without ground truth speech
./launch.sh \
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

*See the [full metrics documentation](https://github.com/wavlab-speech/versa/blob/main/supported_metrics.md) for a complete list with references.*

## 📝 Citation

If you use VERSA in your research, please cite our papers:

```bibtex
@misc{shi2024versaversatileevaluationtoolkit,
      title={VERSA: A Versatile Evaluation Toolkit for Speech, Audio, and Music}, 
      author={Jiatong Shi and Hye-jin Shim and Jinchuan Tian and Siddhant Arora and Haibin Wu and Darius Petermann and Jia Qi Yip and You Zhang and Yuxun Tang and Wangyou Zhang and Dareen Safar Alharthi and Yichen Huang and Koichi Saito and Jionghao Han and Yiwen Zhao and Chris Donahue and Shinji Watanabe},
      year={2024},
      eprint={2412.17667},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2412.17667}, 
}

@misc{shi2024espnetcodeccomprehensivetrainingevaluation,
      title={ESPnet-Codec: Comprehensive Training and Evaluation of Neural Codecs for Audio, Music, and Speech}, 
      author={Jiatong Shi and Jinchuan Tian and Yihan Wu and Jee-weon Jung and Jia Qi Yip and Yoshiki Masuyama and William Chen and Yuning Wu and Yuxun Tang and Massa Baali and Dareen Alharhi and Dong Zhang and Ruifan Deng and Tejes Srivastava and Haibin Wu and Alexander H. Liu and Bhiksha Raj and Qin Jin and Ruihua Song and Shinji Watanabe},
      year={2024},
      eprint={2409.15897},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2409.15897}, 
}
```

## 🙏 Acknowledgement

We sincerely thank all the authors of the open-source implementations listed in our [metrics documentation](https://github.com/wavlab-speech/versa/blob/main/supported_metrics.md).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Please check the [contributing guideline](https://github.com/wavlab-speech/versa/blob/main/contributing.md) first.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

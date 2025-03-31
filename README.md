# VERSA
VERSA (Versatile Evaluation of Speech and Audio) is a toolkit dedicated to collecting evaluation metrics in speech and audio quality. Our goal is to provide a comprehensive connection to the cutting-edge techniques developed for evaluation. The toolkit is also tightly integrated into [ESPnet](https://github.com/espnet/espnet.git).

## Colab Demonstration

[Colab Demonstration at Interspeech2024 Tutorial](https://colab.research.google.com/drive/11c0vZxbSa8invMSfqM999tI3MnyAVsOp?usp=sharing)


## Install

The base-installation is as easy as follows:
```
git clone https://github.com/wavlab-speech/versa.git
cd versa
pip install .
```

As for collection purposes, VERSA instead of re-distributing the model, we try to align as much to the original API provided by the algorithm developer. Therefore, we have many dependencies. We try to include as many as default, but there are cases where the toolkit needs specific installation requirements. Please refer to our [list-of-metric section](https://github.com/shinjiwlab/versa?tab=readme-ov-file#list-of-metrics) for more details on whether the metrics are automatically included or not.  If not, we provide an installation guide or installers in `tools`.


## Quick test
```
python versa/test/test_general.py

# test metrics with additional installation
python versa/test/test_{metric}.py
```

## Usage

Simple usage case for a few samples.
```
# direct usage
python versa/bin/scorer.py \
    --score_config egs/speech.yaml \
    --gt test/test_samples/test1 \
    --pred test/test_samples/test2 \
    --output_file test_result

# with scp-style input
python versa/bin/scorer.py \
    --score_config egs/speech.yaml \
    --gt test/test_samples/test1.scp \
    --pred test/test_samples/test2.scp \
    --output_file test_result

# with kaldi-ark style
python versa/bin/scorer.py \
    --score_config egs/speech.yaml \
    --gt test/test_samples/test1.scp \
    --pred test/test_samples/test2.scp \
    --output_file test_result \
    --io kaldi
  
# For text information
python versa/bin/scorer.py \
    --score_config egs/separate_metrics/wer.yaml \
    --gt test/test_samples/test1.scp \
    --pred test/test_samples/test2.scp \
    --output_file test_result \
    --text test/test_samples/text
```

Use launcher with slurm job submissions
```
# use the launcher
# Option1: with gt speech
./launch.sh \
  <pred_speech_scp> \
  <gt_speech_scp> \
  <score_dir> \
  <split_job_num> 

# Option2: without gt speech
./launch.sh \
  <pred_speech_scp> \
  None \
  <score_dir> \
  <split_job_num>

# aggregate the results
cat <score_dir>/result/*.result.cpu.txt > <score_dir>/utt_result.cpu.txt
cat <score_dir>/result/*.result.gpu.txt > <score_dir>/utt_result.gpu.txt

# show result
python scripts/show_result.py <score_dir>/utt_result.cpu.txt
python scripts/show_result.py <score_dir>/utt_result.gpu.txt 

```

Access `egs/*.yaml` for different configs for different setups.

## List of Metrics

### Independent Metrics

We include x mark if the metric is auto-installed in versa. 

|Number| Auto-Install | Metric Name  (Auto-Install)  | Key in config | Key in report |  Code Source                                                                                                     | References                                                                                       |
|---|---|------------------|---------------|---------------|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| 1 | x | Deep Noise Suppression MOS Score of P.835 (DNSMOS)  | pseudo_mos | dnsmos_overall | [speechmos (MS)](https://pypi.org/project/speechmos/) | [paper](https://arxiv.org/abs/2110.01763) |
| 2 | x | Deep Noise Suppression MOS Score of P.808 (DNSMOS)  | pseudo_mos | dnsmos_p808 | [speechmos (MS)](https://pypi.org/project/speechmos/) | [paper](https://arxiv.org/abs/2005.08138) |
| 3 | x | Non-intrusive Speech Quality and Naturalness Assessment (NISQA) | nisqa | {nisqa_mos_pred, nisqa_noi_pred, nisqa_dis_pred, nisqa_col_pred, nisqa_loud_pred} | [NISQA](https://github.com/gabrielmittag/NISQA) | [paper](https://www.isca-archive.org/interspeech_2021/mittag21_interspeech.pdf) |
| 4 | x | UTokyo-SaruLab System for VoiceMOS Challenge 2022 (UTMOS)  | pseudo_mos | utmos | [speechmos](https://github.com/tarepan/SpeechMOS) | [paper](https://arxiv.org/abs/2204.02152) |
| 5 | x | Packet Loss Concealment-related MOS Score (PLCMOS)  | pseudo_mos | plcmos | [speechmos (MS)](https://pypi.org/project/speechmos/) | [paper](https://arxiv.org/abs/2305.15127)|
| 6 | x | PESQ in TorchAudio-Squim  | squim_no_ref | torch_squim_pesq | [torch_squim](https://pytorch.org/audio/main/tutorials/squim_tutorial.html) | [paper](https://arxiv.org/abs/2304.01448) |
| 7 | x | STOI in TorchAudio-Squim  | squim_no_ref | torch_squim_stoi | [torch_squim](https://pytorch.org/audio/main/tutorials/squim_tutorial.html) | [paper](https://arxiv.org/abs/2304.01448) |
| 8 | x | SI-SDR in TorchAudio-Squim  | squim_no_ref | torch_squim_si_sdr | [torch_squim](https://pytorch.org/audio/main/tutorials/squim_tutorial.html) | [paper](https://arxiv.org/abs/2304.01448) |
| 9 | x | Singing voice MOS  | singmos | singmos |[singmos](https://github.com/South-Twilight/SingMOS/tree/main) | [paper](https://arxiv.org/abs/2406.10911) |
| 10 | x | Sheet SSQA MOS Models | sheet_ssqa | sheet_ssqa |[Sheet](https://github.com/unilight/sheet/tree/main) | [paper](https://arxiv.org/abs/2411.03715) |
| 11 |   | UTMOSv2: UTokyo-SaruLab MOS Prediction System | utmosv2 | utmosv2 |[UTMOSv2](https://github.com/sarulab-speech/UTMOSv2) | [paper](https://arxiv.org/abs/2409.09305) |
| 12 |   | Speech Contrastive Regression for Quality Assessment without reference (ScoreQ) | scoreq_nr | scoreq_nr |[ScoreQ](https://github.com/ftshijt/scoreq/tree/main) | [paper](https://arxiv.org/pdf/2410.06675) |
| 13 | x | Speech enhancement-based SI-SNR | se_snr | se_si_snr | [ESPnet](https://github.com/espnet/espnet.git) | |
| 14 | x | Speech enhancement-based CI-SDR | se_snr | se_ci_sdr | [ESPnet](https://github.com/espnet/espnet.git) | |
| 15 | x | Speech enhancement-based SAR | se_snr | se_sar | [ESPnet](https://github.com/espnet/espnet.git) | |
| 16 | x | Speech enhancement-based SDR | se_snr | se_sdr | [ESPnet](https://github.com/espnet/espnet.git) | |
| 17 | x | PAM: Prompting Audio-Language Models for Audio Quality Assessment | pam | pam | [PAM](https://github.com/soham97/PAM/tree/main) | [Paper](https://arxiv.org/pdf/2402.00282)|
| 18 |  | Speech-to-Reverberation Modulation energy Ratio (SRMR) | srmr | srmr | [SRMRpy](https://github.com/shimhz/SRMRpy.git) | [Paper](http://www.individual.utoronto.ca/falkt/falk/pdf/FalkChan_TASLP2010.pdf)|
| 19 | x | Voice Activity Detection (VAD) | vad | vad_info | [SileroVAD](https://github.com/snakers4/silero-vad) | |
| 20 |  | Speaker Turn Taking (SPK-TT) |  |  |  |  |
| 21 | x | Speaking Word/Character Rate (SWR) | speaking_rate  | speaking_rate | - | - |
| 22 | x | Auti-spoofing Score (SpoofS) with AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks | asvspoof_score | asvspoof_score | [AASIST](https://github.com/clovaai/aasist/tree/main) | [Paper](https://ieeexplore.ieee.org/document/9747766)|
| 23 | x | Language Identification | lid  | language | [ESPnet](https://github.com/espnet/espnet.git) | [Paper](https://arxiv.org/pdf/2401.16658) |
| 24 |   | Audiobox Aesthetics | audiobox_aesthetics  | {audiobox_aesthetics_CE, audiobox_aesthetics_CU, audiobox_aesthetics_PC, audiobox_aesthetics_PQ} | [Audiobox-Aesthetics](https://github.com/facebookresearch/audiobox-aesthetics) | [Paper](https://arxiv.org/abs/2502.05139) |
| 25 |   | Qwen2 Speaker Characteristics - Count | qwen2_speaker_count_metric | qwen2_speaker_count_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 26 |   | Qwen2 Speaker Characteristics - Gender | qwen2_speaker_gender_metric | qwen2_speaker_gender_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 27 |   | Qwen2 Speaker Characteristics - Age | qwen2_speaker_age_metric | qwen2_speaker_age_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 28 |   | Qwen2 Speaker Characteristics - Speech Impairment | qwen2_speech_impairment_metric | qwen2_speech_impairment_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 29 |   | Qwen2 Voice Properties - Pitch | qwen2_voice_pitch_metric | qwen2_voice_pitch_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 30 |   | Qwen2 Voice Properties - Pitch Range | qwen2_pitch_range_metric | qwen2_pitch_range_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 31 |   | Qwen2 Voice Properties - Voice Type | qwen2_voice_type_metric | qwen2_voice_type_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 32 |   | Qwen2 Voice Properties - Volume Level | qwen2_speech_volume_level_metric | qwen2_speech_volume_level_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 33 |   | Qwen2 Speech Content - Language | qwen2_language_metric | qwen2_language_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 34 |   | Qwen2 Speech Content - Register | qwen2_speech_register_metric | qwen2_speech_register_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 35 |   | Qwen2 Speech Content - Vocabulary Complexity | qwen2_vocabulary_complexity_metric | qwen2_vocabulary_complexity_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 36 |   | Qwen2 Speech Content - Purpose | qwen2_speech_purpose_metric | qwen2_speech_purpose_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 37 |   | Qwen2 Speech Delivery - Emotion | qwen2_speech_emotion_metric | qwen2_speech_emotion_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 38 |   | Qwen2 Speech Delivery - Clarity | qwen2_speech_clarity_metric | qwen2_speech_clarity_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 39 |   | Qwen2 Speech Delivery - Rate | qwen2_speech_rate_metric | qwen2_speech_rate_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 40 |   | Qwen2 Speech Delivery - Style | qwen2_speaking_style_metric | qwen2_speaking_style_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 41 |   | Qwen2 Speech Delivery - Emotional Vocalizations | qwen2_laughter_crying_metric | qwen2_laughter_crying_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 42 |   | Qwen2 Interaction Patterns - Overlapping Speech | qwen2_overlapping_speech_metric | qwen2_overlapping_speech_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 43 |   | Qwen2 Recording Environment - Background | qwen2_speech_background_environment_metric | qwen2_speech_background_environment_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 44 |   | Qwen2 Recording Environment - Quality | qwen2_recording_quality_metric | qwen2_recording_quality_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 45 |   | Qwen2 Recording Environment - Channel Type | qwen2_channel_type_metric | qwen2_channel_type_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |


### Dependent Metrics
|Number| Auto-Install | Metric Name  (Auto-Install)  | Key in config | Key in report |  Code Source                                                                                                     | References                                                                                       |
|---|---|------------------|---------------|---------------|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| 1 | x | Mel Cepstral Distortion (MCD)  | mcd_f0 | mcd | [espnet](https://github.com/espnet/espnet) and [s3prl-vc](https://github.com/unilight/s3prl-vc) | [paper](https://ieeexplore.ieee.org/iel2/3220/9154/00407206.pdf) |
| 2 | x | F0 Correlation | mcd_f0 | f0_corr | [espnet](https://github.com/espnet/espnet) and [s3prl-vc](https://github.com/unilight/s3prl-vc) | [paper](https://ieeexplore.ieee.org/iel7/9040208/9052899/09053512.pdf) |
| 3 | x | F0 Root Mean Square Error  | mcd_f0 | f0_rmse | [espnet](https://github.com/espnet/espnet) and [s3prl-vc](https://github.com/unilight/s3prl-vc) | [paper](https://ieeexplore.ieee.org/iel7/9040208/9052899/09053512.pdf) |
| 4 | x | Signal-to-interference  Ratio (SIR)  | signal_metric | sir | [espnet](https://github.com/espnet/espnet) | - |
| 5 | x | Signal-to-artifact Ratio (SAR)  | signal_metric | sar | [espnet](https://github.com/espnet/espnet) | - |
| 6 | x | Signal-to-distortion Ratio (SDR)  | signal_metric | sdr | [espnet](https://github.com/espnet/espnet) | - |
| 7 | x | Convolutional scale-invariant signal-to-distortion ratio (CI-SDR)  | signal_metric | ci-sdr | [ci_sdr](https://github.com/fgnt/ci_sdr) | [paper](https://arxiv.org/abs/2011.15003) |
| 8 | x | Scale-invariant signal-to-noise ratio (SI-SNR)  | signal_metric | si-snr | [espnet](https://github.com/espnet/espnet) | [paper](https://arxiv.org/abs/1711.00541) |
| 9 | x | Perceptual Evaluation of Speech Quality (PESQ)  | pesq | pesq | [pesq](https://pypi.org/project/pesq/) | [paper](https://ieeexplore.ieee.org/document/941023) |
| 10 | x | Short-Time Objective Intelligibility (STOI)  | stoi | stoi | [pystoi](https://github.com/mpariente/pystoi) | [paper](https://ieeexplore.ieee.org/document/5495701) |
| 11 | x | Speech BERT Score  | discrete_speech | speech_bert | [discrete speech metric](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics) | [paper](https://arxiv.org/abs/2401.16812) |
| 12 | x | Discrete Speech BLEU Score  | discrete_speech | speech_belu | [discrete speech metric](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics) | [paper](https://arxiv.org/abs/2401.16812) |
| 13 | x | Discrete Speech Token Edit Distance  | discrete_speech | speech_token_distance | [discrete speech metric](https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics) | [paper](https://arxiv.org/abs/2401.16812) |
| 14 |   | Dynamic Time Warping Cost Metric | warpq | warpq |[WARP-Q](https://github.com/wjassim/WARP-Q) | [paper](https://arxiv.org/abs/2102.10449) |
| 15 |   | Speech Contrastive Regression for Quality Assessment with reference (ScoreQ) |  scoreq_ref | scoreq_ref |[ScoreQ](https://github.com/ftshijt/scoreq/tree/main) | [paper](https://arxiv.org/pdf/2410.06675) |
| 16 |  | 2f-Model |   |  |  |  |
| 17 | x | Log-Weighted Mean Square Error | log_wmse | log_wmse |[log_wmse](https://github.com/nomonosound/log-wmse-audio-quality) |
| 18 | x | ASR-oriented Mismatch Error Rate (ASR-Mismatch) | asr_match | asr_match_error_rate | - | - |
| 19 |   | Virtual Speech Quality Objective Listener (VISQOL)  | visqol | visqol | [google-visqol](https://github.com/google/visqol) | [paper](https://arxiv.org/abs/2004.09584) |
| 20 |  | Frequency-Weighted SEGmental SNR (FWSEGSNR) | pysepm | pysepm_fwsegsnr | [pysepm](https://github.com/shimhz/pysepm.git) | [Paper](https://ecs.utdallas.edu/loizou/speech/obj_paper_jan08.pdf)|
| 21 |  | Weighted Spectral Slope (WSS) | pysepm | pysepm_wss | [pysepm](https://github.com/shimhz/pysepm.git) | [Paper](https://ecs.utdallas.edu/loizou/speech/obj_paper_jan08.pdf)|
| 22 |  | Cepstrum Distance Objective Speech Quality Measure (CD) | pysepm | pysepm_cd | [pysepm](https://github.com/shimhz/pysepm.git) | [Paper](https://ieeexplore.ieee.org/document/407206)|
| 23 |  | Composite Objective Speech Quality (composite) | pysepm | pysepm_Csig, pysepm_Cbak, pysepm_Covl | [pysepm](https://github.com/shimhz/pysepm.git) | [Paper](https://ecs.utdallas.edu/loizou/speech/obj_paper_jan08.pdf)|
| 24 |  | Coherence and speech intelligibility index (CSII) | pysepm | pysepm_csii_high, pysepm_csii_mid, pysepm_csii_low | [pysepm](https://github.com/shimhz/pysepm.git) | [Paper](https://www.researchgate.net/profile/James-Kates-2/publication/7842209_Coherence_and_the_speech_intelligibility_index/links/546f5dab0cf2d67fc0310f88/Coherence-and-the-speech-intelligibility-index.pdf)|
| 25 |  | Normalized-covariance measure (NCM) | pysepm | pysepm_ncm | [pysepm](https://github.com/shimhz/pysepm.git) | [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3037773/pdf/JASMAN-000128-003715_1.pdf)|


### Non-match Metrics

|Number| Auto-Install | Metric Name  (Auto-Install)  | Key in config | Key in report |  Code Source                                                                                                     | References                                                                                       |
|---|---|------------------|---------------|---------------|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| 1 |  | NORESQA : A Framework for Speech Quality Assessment using Non-Matching References | noresqa | noresqa | [Noresqa](https://github.com/shimhz/Noresqa.git) | [Paper](https://proceedings.neurips.cc/paper/2021/file/bc6d753857fe3dd4275dff707dedf329-Paper.pdf)|
| 2 | x | MOS in TorchAudio-Squim  | squim_ref | torch_squim_mos |[torch_squim](https://pytorch.org/audio/main/tutorials/squim_tutorial.html) | [paper](https://arxiv.org/abs/2304.01448) |
| 3 | x | ESPnet Speech Recognition-based Error Rate | espnet_wer | espnet_wer |[ESPnet](https://github.com/espnet/espnet) | [paper](https://arxiv.org/pdf/1804.00015) |
| 4 | x | ESPnet-OWSM Speech Recognition-based Error Rate | owsm_wer | owsm_wer |[ESPnet](https://github.com/espnet/espnet) | [paper](https://arxiv.org/abs/2309.13876) |
| 5 | x | OpenAI-Whisper Speech Recognition-based Error Rate | whisper_wer | whisper_wer |[Whisper](https://github.com/openai/whisper) | [paper](https://arxiv.org/abs/2212.04356) |
| 6 |   | Emotion2vec similarity (emo2vec) | emo2vec_similarity | emotion_similarity | [emo2vec](https://github.com/ftshijt/emotion2vec/tree/main) | [paper](https://arxiv.org/abs/2312.15185) | 
| 7 | x | Speaker Embedding Similarity  | speaker | spk_similarity | [espnet](https://github.com/espnet/espnet) | [paper](https://arxiv.org/abs/2401.17230) |
| 8 |   | NOMAD: Unsupervised Learning of Perceptual Embeddings For Speech Enhancement and Non-Matching Reference Audio Quality Assessment |  nomad | nomad |[Nomad](https://github.com/shimhz/nomad/tree/main) | [paper](https://arxiv.org/abs/2309.16284) |
| 9 |   | Contrastive Language-Audio Pretraining Score (CLAP Score) | clap_score | clap_score | [fadtk](https://github.com/gudgud96/frechet-audio-distance) | [paper](https://arxiv.org/abs/2301.12661) |
| 10 |   | Accompaniment Prompt Adherence (APA) | apa | apa | [Sony-audio-metrics](https://github.com/SonyCSLParis/audio-metrics) | [paper](https://arxiv.org/abs/2404.00775) |
| 11 |  | Log Likelihood Ratio (LLR) | pysepm | pysepm_llr | [pysepm](https://github.com/shimhz/pysepm.git) | [Paper](https://ecs.utdallas.edu/loizou/speech/obj_paper_jan08.pdf)|


### Distributional Metrics (in verifying)

|Number| Auto-Install | Metric Name  (Auto-Install)  | Key in config | Key in report |  Code Source                                                                                                     | References                                                                                       |
|---|---|------------------|---------------|---------------|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| 1 |   | Frechet Audio Distance (FAD) | fad | fad | [fadtk](https://github.com/microsoft/fadtk) | [paper](https://arxiv.org/abs/1812.08466) |
| 2 |   | Kullback-Leibler Divergence on Embedding Distribution | kl_embedding | kl_embedding | [Stability-AI](https://github.com/Stability-AI/stable-audio-metrics) |  |
| 3 |   | Audio Density Score | audio_density_coverage | audio_density | [Sony-audio-metrics](https://github.com/SonyCSLParis/audio-metrics) | [paper](https://arxiv.org/abs/2002.09797) |
| 4 |   | Audio Coverage Score | audio_density_coverage | audio_coverage | [Sony-audio-metrics](https://github.com/SonyCSLParis/audio-metrics) | [paper](https://arxiv.org/abs/2002.09797) |
| 5 |  | KID : Kernel Distance Metric for Audio/Music Quality | [KID](https://github.com/SonyCSLParis/audio-metrics/tree/main) | [Paper](https://arxiv.org/abs/1812.08466)|


## Citation

If you find this repo useful, please cite the following papers:
```
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


## Acknowledgement
We sincerely thank all the open-source implementations listed in https://github.com/shinjiwlab/versa/tree/main#list-of-metrics 

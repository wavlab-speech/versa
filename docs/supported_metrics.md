
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
| 25 | x | Qwen2 Speaker Characteristics - Count | qwen2_speaker_count_metric | qwen2_speaker_count_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 26 | x | Qwen2 Speaker Characteristics - Gender | qwen2_speaker_gender_metric | qwen2_speaker_gender_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 27 | x | Qwen2 Speaker Characteristics - Age | qwen2_speaker_age_metric | qwen2_speaker_age_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 28 | x | Qwen2 Speaker Characteristics - Speech Impairment | qwen2_speech_impairment_metric | qwen2_speech_impairment_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 29 | x | Qwen2 Voice Properties - Pitch | qwen2_voice_pitch_metric | qwen2_voice_pitch_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 30 | x | Qwen2 Voice Properties - Pitch Range | qwen2_pitch_range_metric | qwen2_pitch_range_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 31 | x | Qwen2 Voice Properties - Voice Type | qwen2_voice_type_metric | qwen2_voice_type_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 32 | x | Qwen2 Voice Properties - Volume Level | qwen2_speech_volume_level_metric | qwen2_speech_volume_level_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 33 | x | Qwen2 Speech Content - Language | qwen2_language_metric | qwen2_language_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 34 | x | Qwen2 Speech Content - Register | qwen2_speech_register_metric | qwen2_speech_register_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 35 | x | Qwen2 Speech Content - Vocabulary Complexity | qwen2_vocabulary_complexity_metric | qwen2_vocabulary_complexity_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 36 | x | Qwen2 Speech Content - Purpose | qwen2_speech_purpose_metric | qwen2_speech_purpose_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 37 | x | Qwen2 Speech Delivery - Emotion | qwen2_speech_emotion_metric | qwen2_speech_emotion_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 38 | x | Qwen2 Speech Delivery - Clarity | qwen2_speech_clarity_metric | qwen2_speech_clarity_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 39 | x | Qwen2 Speech Delivery - Rate | qwen2_speech_rate_metric | qwen2_speech_rate_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 40 | x | Qwen2 Speech Delivery - Style | qwen2_speaking_style_metric | qwen2_speaking_style_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 41 | x | Qwen2 Speech Delivery - Emotional Vocalizations | qwen2_laughter_crying_metric | qwen2_laughter_crying_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 42 | x | Qwen2 Interaction Patterns - Overlapping Speech | qwen2_overlapping_speech_metric | qwen2_overlapping_speech_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 43 | x | Qwen2 Recording Environment - Background | qwen2_speech_background_environment_metric | qwen2_speech_background_environment_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 44 | x | Qwen2 Recording Environment - Quality | qwen2_recording_quality_metric | qwen2_recording_quality_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 45 | x | Qwen2 Recording Environment - Channel Type | qwen2_channel_type_metric | qwen2_channel_type_metric | [Qwen2 Audio](https://github.com/QwenLM/Qwen2-Audio) | [paper](https://arxiv.org/abs/2407.10759) |
| 46 | x | Dimensional Emotion | w2v2_dimensional_emotion | w2v2_dimensional_emotion | [w2v2-how-to](https://github.com/audeering/w2v2-how-to) | [paper](https://arxiv.org/pdf/2203.07378) |
| 47 | x | Uni-VERSA (Versatile Speech Assessment with a Unified Framework) | universa | universa_{sub_metrics} | [Uni-VERSA](https://huggingface.co/collections/espnet/universa-6834e7c0a28225bffb6e2526) | [paper](https://arxiv.org/abs/2505.20741) |
| 48 | x | DNSMOS Pro: A Reduced-Size DNN for Probabilistic MOS of Speech  | pseudo_mos | dnsmos_pro_bvcc | [DNSMOSPro](https://github.com/fcumlin/DNSMOSPro/tree/main) | [paper](https://www.isca-archive.org/interspeech_2024/cumlin24_interspeech.html) |
| 49 | x | DNSMOS Pro: A Reduced-Size DNN for Probabilistic MOS of Speech  | pseudo_mos | dnsmos_pro_nisqa | [DNSMOSPro](https://github.com/fcumlin/DNSMOSPro/tree/main) | [paper](https://www.isca-archive.org/interspeech_2024/cumlin24_interspeech.html) |
| 50 | x | DNSMOS Pro: A Reduced-Size DNN for Probabilistic MOS of Speech  | pseudo_mos | dnsmos_pro_vcc2018 | [DNSMOSPro](https://github.com/fcumlin/DNSMOSPro/tree/main) | [paper](https://www.isca-archive.org/interspeech_2024/cumlin24_interspeech.html) |



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
| 26 | x | Uni-VERSA (Versatile Speech Assessment with a Unified Framework) with Paired Reference | universa | universa_{sub_metrics} | [Uni-VERSA](https://huggingface.co/collections/espnet/universa-6834e7c0a28225bffb6e2526) | [paper](https://arxiv.org/abs/2505.20741) |
| 27 | x | Chroma-related Alignment | chroma_alignment | chroma_{stft,cqt,cens}_{cosine, euclidean}_dtw{"", _log, _raw} | - | - |
| 28 | x | Deep Perceptual Audio Metric (DPAM) | dpam | dpam_distance | [PerceptualAudio_Pytorch](https://github.com/adrienchaton/PerceptualAudio_pytorch)  | [paper](https://arxiv.org/abs/2001.04460) |
| 29 | x | Contrastive learning-based Deep Perceptual Audio Metric (CDPAM) | cdpam | cdpam_distance | [PerceptualAudio](https://github.com/pranaymanocha/PerceptualAudio/cdpam) | [paper](https://arxiv.org/abs/2102.05109) |


### Non-match Metrics

|Number| Auto-Install | Metric Name  (Auto-Install)  | Key in config | Key in report |  Code Source                                                                                                     | References                                                                                       |
|---|---|------------------|---------------|---------------|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| 1 |  | NORESQA : A Framework for Speech Quality Assessment using Non-Matching References | noresqa | noresqa | [Noresqa](https://github.com/shimhz/Noresqa.git) | [Paper](https://proceedings.neurips.cc/paper/2021/file/bc6d753857fe3dd4275dff707dedf329-Paper.pdf)|
| 2 | x | MOS in TorchAudio-Squim  | squim_ref | torch_squim_mos |[torch_squim](https://pytorch.org/audio/main/tutorials/squim_tutorial.html) | [paper](https://arxiv.org/abs/2304.01448) |
| 3 | x | ESPnet Speech Recognition-based Error Rate | espnet_wer | espnet_wer |[ESPnet](https://github.com/espnet/espnet) | [paper](https://arxiv.org/pdf/1804.00015) |
| 4 | x | ESPnet-OWSM Speech Recognition-based Error Rate | owsm_wer | owsm_wer |[ESPnet](https://github.com/espnet/espnet) | [paper](https://arxiv.org/abs/2309.13876) |
| 5 | x | OpenAI-Whisper Speech Recognition-based Error Rate | whisper_wer | whisper_wer |[Whisper](https://github.com/openai/whisper) | [paper](https://arxiv.org/abs/2212.04356) |
| 6 |   | Faster-Whisper Speech Recognition-based Error Rate | faster_whisper_wer | faster_whisper_wer |[Faster-Whisper](https://github.com/systran/faster-whisper) | - |
| 7 | x | NVIDIA Conformer-Transducer X-Large Speech Recognition-based Error Rate | nemo_wer | nemo_wer |[NeMo](https://github.com/NVIDIA/NeMo) | [paper](https://arxiv.org/abs/2005.08100) |
| 8 | x | Facebook Hubert-Large-Finetuned Speech Recognition-based Error Rate | hubert_wer | hubert_wer |[HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert) | [paper](https://arxiv.org/abs/2106.07447) |
| 9 |   | Emotion2vec similarity (emo2vec) | emo2vec_similarity | emotion_similarity | [emo2vec](https://github.com/ftshijt/emotion2vec/tree/main) | [paper](https://arxiv.org/abs/2312.15185) | 
| 10 | x | Speaker Embedding Similarity  | speaker | spk_similarity | [espnet](https://github.com/espnet/espnet) | [paper](https://arxiv.org/abs/2401.17230) |
| 11 |   | NOMAD: Unsupervised Learning of Perceptual Embeddings For Speech Enhancement and Non-Matching Reference Audio Quality Assessment |  nomad | nomad |[Nomad](https://github.com/shimhz/nomad/tree/main) | [paper](https://arxiv.org/abs/2309.16284) |
| 12 |   | Contrastive Language-Audio Pretraining Score (CLAP Score) | clap_score | clap_score | [fadtk](https://github.com/gudgud96/frechet-audio-distance) | [paper](https://arxiv.org/abs/2301.12661) |
| 13 |   | Accompaniment Prompt Adherence (APA) | apa | apa | [Sony-audio-metrics](https://github.com/SonyCSLParis/audio-metrics) | [paper](https://arxiv.org/abs/2404.00775) |
| 14 |  | Log Likelihood Ratio (LLR) | pysepm | pysepm_llr | [pysepm](https://github.com/shimhz/pysepm.git) | [Paper](https://ecs.utdallas.edu/loizou/speech/obj_paper_jan08.pdf)|
| 15 | x | Uni-VERSA (Versatile Speech Assessment with a Unified Framework) with Paired Text | universa | universa_{sub_metrics} | [Uni-VERSA](https://huggingface.co/collections/espnet/universa-6834e7c0a28225bffb6e2526) | [paper](https://arxiv.org/abs/2505.20741) |
| 16 |  | Singer Embedding Similarity  | singer | singer_similarity | [SSL-Singer-Identity](https://github.com/SonyCSLParis/ssl-singer-identity) | [paper](https://hal.science/hal-04186048v1) |

### Distributional Metrics (in verifying)

|Number| Auto-Install | Metric Name  (Auto-Install)  | Key in config | Key in report |  Code Source                                                                                                     | References                                                                                       |
|---|---|------------------|---------------|---------------|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| 1 |   | Frechet Audio Distance (FAD) | fad | fad | [fadtk](https://github.com/microsoft/fadtk) | [paper](https://arxiv.org/abs/1812.08466) |
| 2 |   | Kullback-Leibler Divergence on Embedding Distribution | kl_embedding | kl_embedding | [Stability-AI](https://github.com/Stability-AI/stable-audio-metrics) |  |
| 3 |   | Audio Density Score | audio_density_coverage | audio_density | [Sony-audio-metrics](https://github.com/SonyCSLParis/audio-metrics) | [paper](https://arxiv.org/abs/2002.09797) |
| 4 |   | Audio Coverage Score | audio_density_coverage | audio_coverage | [Sony-audio-metrics](https://github.com/SonyCSLParis/audio-metrics) | [paper](https://arxiv.org/abs/2002.09797) |
| 5 |  | KID : Kernel Distance Metric for Audio/Music Quality | [KID](https://github.com/SonyCSLParis/audio-metrics/tree/main) | [Paper](https://arxiv.org/abs/1812.08466)|

## Acknowledgement
We sincerely thank all the open-source implementations listed in https://github.com/shinjiwlab/versa/tree/main#list-of-metrics 

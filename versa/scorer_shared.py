#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import logging

import kaldiio
import librosa
import soundfile as sf
import yaml
from tqdm import tqdm

from versa.utils_shared import (check_all_same, check_minimum_length,
                                find_files, load_audio, wav_normalize)


def audio_loader_setup(audio, io):
    # get ready compute embeddings
    if io == "kaldi":
        audio_files = kaldiio.load_scp(audio)
    elif io == "dir":
        audio_files = find_files(audio)
    elif io == "soundfile":
        audio_files = {}
        with open(audio) as f:
            for line in f.readlines():
                key, value = line.strip().split(maxsplit=1)
                if value.endswith("|"):
                    raise ValueError(
                        "Not supported wav.scp format. Set IO interface to kaldi"
                    )
                audio_files[key] = value
    return audio_files


def load_score_modules(score_config, use_gt=True, use_gt_text=False, use_gpu=False):
    score_modules = {}
    for config in score_config:
        print(config, flush=True)
        if config["name"] == "mcd_f0":
            logging.info("Loading MCD & F0 evaluation...")
            from versa import mcd_f0

            score_modules["mcd_f0"] = {
                "module": mcd_f0,
                "args": {
                    "f0min": config.get("f0min", 0),
                    "f0max": config.get("f0max", 24000),
                    "mcep_shift": config.get("mcep_shift", 5),
                    "mcep_fftl": config.get("mcep_fftl", 1024),
                    "mcep_dim": config.get("mcep_dim", 39),
                    "mcep_alpha": config.get("mcep_alpha", 0.466),
                    "seq_mismatch_tolerance": config.get("seq_mismatch_tolerance", 0.1),
                    "power_threshold": config.get("power_threshold", -20),
                    "dtw": config.get("dtw", False),
                },
            }
            logging.info("Initiate MCD & F0 evaluation successfully.")

        elif config["name"] == "signal_metric":
            if not use_gt:
                logging.warning(
                    "Cannot use signal metric because no gt audio is provided"
                )
                continue

            logging.info("Loading signal metric evaluation...")
            from versa import signal_metric

            score_modules["signal_metric"] = {"module": signal_metric}
            logging.info("Initiate signal metric evaluation successfully.")

        elif config["name"] == "warpq":
            if not use_gt:
                logging.warning("Cannot use warpq because no gt audio is provided")
                continue

            logging.info("Loading WARPQ metric evaluation...")
            from versa.sequence_metrics.warpq import warpq, warpq_setup

            score_modules["warpq"] = {"model": warpq_setup(), "module": warpq}
            logging.info("Initiate WARP-Q metric...")

        elif config["name"] == "nisqa":

            logging.info("Loading NISQA evaluation...")
            from versa.utterance_metrics.nisqa import (nisqa_metric,
                                                       nisqa_model_setup)

            # Load the NISQA model
            nisqa_model = nisqa_model_setup(
                nisqa_model_path=config.get(
                    "model_path", "./tools/NISQA/weights/nisqa.tar"
                ),
                use_gpu=use_gpu,
            )
            score_modules["nisqa"] = {
                "module": nisqa_metric,
                "model": nisqa_model,
            }
            logging.info("Initiate NISQA evaluation successfully.")

        elif config["name"] == "discrete_speech":
            if not use_gt:
                logging.warning(
                    "Cannot use discrete speech metric because no gt audio is provided"
                )
                continue

            logging.info("Loading discrete speech evaluation...")
            from versa import discrete_speech_metric, discrete_speech_setup

            score_modules["discrete_speech"] = {
                "module": discrete_speech_metric,
                "model": discrete_speech_setup(use_gpu=use_gpu),
            }
            logging.info("Initiate discrete speech evaluation successfully.")

        elif config["name"] == "pseudo_mos":
            logging.info("Loading pseudo MOS evaluation...")
            from versa import pseudo_mos_metric, pseudo_mos_setup

            predictor_dict, predictor_fs = pseudo_mos_setup(
                use_gpu=use_gpu,
                predictor_types=config.get("predictor_types", ["utmos"]),
                predictor_args=config.get("predictor_args", {}),
            )
            score_modules["pseudo_mos"] = {
                "module": pseudo_mos_metric,
                "args": {
                    "predictor_dict": predictor_dict,
                    "predictor_fs": predictor_fs,
                    "use_gpu": use_gpu,
                },
            }
            logging.info("Initiate pseudo MOS evaluation successfully.")

        elif config["name"] == "pesq":
            if not use_gt:
                logging.warning(
                    "Cannot use pesq metric because no gt audio is provided"
                )
                continue

            logging.info("Loading pesq evaluation...")
            from versa import pesq_metric

            score_modules["pesq"] = {"module": pesq_metric}
            logging.info("Initiate pesq evaluation successfully.")

        elif config["name"] == "stoi":
            if not use_gt:
                logging.warning(
                    "Cannot use stoi metric because no gt audio is provided"
                )
                continue

            logging.info("Loading stoi evaluation...")
            from versa import stoi_metric

            score_modules["stoi"] = {"module": stoi_metric}
            logging.info("Initiate stoi evaluation successfully.")

        elif config["name"] == "estoi":                                             
            if not use_gt:                                                         
                logging.warning(                                                   
                    "Cannot use estoi metric because no gt audio is provided"       
                )                                                                  
                continue                                                           
                                                                                   
            logging.info("Loading stoi evaluation...")                             
            from versa import estoi_metric                                          
                                                                                   
            score_modules["estoi"] = {"module": estoi_metric}                        
            logging.info("Initiate stoi evaluation successfully.") 

        elif config["name"] == "visqol":
            if not use_gt:
                logging.warning(
                    "Cannot use visqol metric because no gt audio is provided"
                )
                continue

            logging.info("Loading visqol evaluation...")
            try:
                from versa import visqol_metric, visqol_setup
            except ImportError:
                logging.warning(
                    "VISQOL not installed, please check `tools` for installation guideline"
                )
                continue

            api, fs = visqol_setup(model=config.get("model", "default"))
            score_modules["visqol"] = {
                "module": visqol_metric,
                "args": {"api": api, "api_fs": fs},
            }
            logging.info("Initiate visqol evaluation successfully.")

        elif config["name"] == "speaker":
            if not use_gt:
                logging.warning(
                    "Cannot use speaker metric because no gt audio is provided"
                )
                continue

            logging.info("Loading speaker evaluation...")
            from versa import speaker_metric, speaker_model_setup

            spk_model = speaker_model_setup(
                model_tag=config.get("model_tag", "default"),
                model_path=config.get("model_path", None),
                model_config=config.get("model_config", None),
                use_gpu=use_gpu,
            )
            score_modules["speaker"] = {
                "module": speaker_metric,
                "args": {"model": spk_model},
            }
            logging.info("Initiate speaker evaluation successfully.")

        elif config["name"] == "sheet_ssqa":

            logging.info("Loading Sheet SSQA models for evaluation...")
            from versa import sheet_ssqa, sheet_ssqa_setup

            sheet_model = sheet_ssqa_setup(
                model_tag=config.get("model_tag", "default"),
                model_path=config.get("model_path", None),
                model_config=config.get("model_config", None),
                use_gpu=use_gpu,
            )
            score_modules["sheet_ssqa"] = {
                "module": sheet_ssqa,
                "args": {"model": sheet_model, "use_gpu": use_gpu},
            }
            logging.info("Initiate Sheet SSQA evaluation successfully.")

        elif config["name"] == "squim_ref":
            if not use_gt:
                logging.warning("Cannot use squim_ref because no gt audio is provided")
                continue

            logging.info("Loading squim metrics with reference")
            from versa import squim_metric

            score_modules["squim_ref"] = {
                "module": squim_metric,
            }
            logging.info("Initiate torch squim (with reference) successfully")

        elif config["name"] == "squim_no_ref":

            logging.info("Loading squim metrics with reference")
            from versa import squim_metric_no_ref

            score_modules["squim_no_ref"] = {
                "module": squim_metric_no_ref,
            }
            logging.info("Initiate torch squim (without reference) successfully")

        elif config["name"] == "espnet_wer":
            if not use_gt_text:
                logging.warning("Cannot use espnet_wer because no gt text is provided")
                continue

            logging.info("Loading espnet_wer metric with reference text")
            from versa import espnet_levenshtein_metric, espnet_wer_setup

            score_modules["espnet_wer"] = {
                "module": espnet_levenshtein_metric,
                "args": espnet_wer_setup(
                    model_tag=config.get("model_tag", "default"),
                    beam_size=config.get("beam_size", 1),
                    text_cleaner=config.get("text_cleaner", "whisper_basic"),
                    use_gpu=use_gpu,
                ),
            }
            logging.info("Initiate ESPnet WER calculation successfully")

        elif config["name"] == "owsm_wer":
            if not use_gt_text:
                logging.warning("Cannot use owsm_wer because no gt text is provided")
                continue

            logging.info("Loading owsm_wer metric with reference text")
            from versa import owsm_levenshtein_metric, owsm_wer_setup

            score_modules["owsm_wer"] = {
                "module": owsm_levenshtein_metric,
                "args": owsm_wer_setup(
                    model_tag=config.get("model_tag", "default"),
                    beam_size=config.get("beam_size", 1),
                    text_cleaner=config.get("text_cleaner", "whisper_basic"),
                    use_gpu=use_gpu,
                ),
            }
            logging.info("Initiate ESPnet-OWSM WER calculation successfully")

        elif config["name"] == "whisper_wer":
            if not use_gt_text:
                logging.warning("Cannot use whisper_wer because no gt text is provided")
                continue

            logging.info("Loading whisper_wer metric with reference text")
            from versa import whisper_levenshtein_metric, whisper_wer_setup

            # Load whisper model if it is already loaded
            if (
                "speaking_rate" in score_modules.keys()
                or "asr_matching" in score_modules.keys()
            ):
                args_cache = score_modules["speaking_rate"]["args"]
            else:
                args_cache = whisper_wer_setup(
                    model_tag=config.get("model_tag", "default"),
                    beam_size=config.get("beam_size", 1),
                    text_cleaner=config.get("text_cleaner", "whisper_basic"),
                    use_gpu=use_gpu,
                )

            score_modules["whisper_wer"] = {
                "module": whisper_levenshtein_metric,
                "args": args_cache,
            }
            logging.info("Initiate Whisper WER calculation successfully")

        elif config["name"] == "scoreq_ref":
            if not use_gt:
                logging.warning("Cannot use scoreq_ref because no gt audio is provided")
                continue

            logging.info("Loading scoreq metrics with reference")
            from versa import scoreq_ref, scoreq_ref_setup

            model = scoreq_ref_setup(
                data_domain=config.get("data_domain", "synthetic"),
                cache_dir=config.get("model_cache", "./scoreq_pt-models"),
                use_gpu=use_gpu,
            )

            score_modules["scoreq_ref"] = {
                "module": scoreq_ref,
                "model": model,
            }
            logging.info("Initiate scoreq (with reference) successfully")

        elif config["name"] == "scoreq_nr":
            logging.info("Loading scoreq metrics without reference")
            from versa import scoreq_nr, scoreq_nr_setup

            model = scoreq_nr_setup(
                data_domain=config.get("data_domain", "synthetic"),
                cache_dir=config.get("model_cache", "./scoreq_pt-models"),
                use_gpu=use_gpu,
            )

            score_modules["scoreq_nr"] = {
                "module": scoreq_nr,
                "model": model,
            }
            logging.info("Initiate scoreq (with reference) successfully")

        elif config["name"] == "nomad":
            logging.info("Loading nomad metrics with reference")
            from versa import nomad, nomad_setup

            model = nomad_setup(
                cache_dir=config.get("model_cache", "./scoreq_pt-models"),
                use_gpu=use_gpu,
            )

            score_modules["nomad"] = {
                "module": nomad,
                "model": model,
            }
            logging.info("Initiate nomad successfully")

        elif config["name"] == "emo2vec_similarity":
            if not use_gt:
                logging.warning(
                    "Cannot use emo2vec_similarity metric because no gt audio is provided"
                )
                continue

            logging.info("Loading emo2vec metrics with reference")
            from versa import emo2vec_setup, emo_sim

            model = emo2vec_setup(
                model_tag=config.get("model_tag", "default"),
                model_path=config.get("model_path", None),
                use_gpu=use_gpu,
            )

            score_modules["emotion"] = {
                "module": emo_sim,
                "model": model,
            }
            logging.info("Initiate emo2vec successfully")

        elif config["name"] == "se_snr":
            logging.info("Loading se_snr metrics with reference")
            from versa import se_snr, se_snr_setup

            model = se_snr_setup(
                model_tag=config.get("model_tag", "default"),
                model_path=config.get("model_path", None),
                use_gpu=use_gpu,
            )

            score_modules["se_snr"] = {
                "module": se_snr,
                "model": model,
            }
            logging.info("Initiate se_snr successfully")

        elif config["name"] == "pam":

            logging.info("Loading pam metric without reference...")
            from versa.utterance_metrics.pam import pam_metric, pam_model_setup

            pam_model = pam_model_setup(model_config=config, use_gpu=use_gpu)
            score_modules["pam"] = {
                "module": pam_metric,
                "model": pam_model,
            }
            logging.info("Initiate pam metric successfully.")
        elif config["name"] == "vad":
            logging.info("Loading vad metric without reference...")
            from versa.utterance_metrics.vad import vad_metric, vad_model_setup

            vad_model = vad_model_setup(
                threshold=config.get("threshold", 0.5),
                min_speech_duration_ms=config.get("min_speech_duration_ms", 250),
                max_speech_duration_s=config.get("max_speech_duration_s", float("inf")),
                min_silence_duration_ms=config.get("min_silence_duration_ms", 100),
                speech_pad_ms=config.get("speech_pad_ms", 30),
            )
            score_modules["vad"] = {
                "module": vad_metric,
                "args": vad_model,
            }
            logging.info("Initiate vad metric successfully.")

        elif config["name"] == "asvspoof_score":

            logging.info("Loading asvspoof score metric without reference...")
            from versa.utterance_metrics.asvspoof_score import (
                asvspoof_metric, deepfake_detection_model_setup)

            deepfake_detection_model = deepfake_detection_model_setup(use_gpu=use_gpu)
            score_modules["asvspoof_score"] = {
                "module": asvspoof_metric,
                "model": deepfake_detection_model,
            }
            logging.info("Initiate asvspoof score metric successfully.")

        elif config["name"] == "pysepm":
            if not use_gt:
                logging.warning("Cannot use pysepm because no gt audio is provided")
                continue

            logging.info("Loading pysepm metrics with reference")
            from versa import pysepm_metric

            score_modules["pysepm"] = {
                "module": pysepm_metric,
                "args": {
                    "frame_len": config.get("frame_len", 0.03),
                    "overlap": config.get("overlap", 0.75),
                },
            }
            logging.info("Initiate pysepm successfully")

        elif config["name"] == "srmr":
            logging.info("Loading srmr metrics with reference")
            from versa import srmr_metric

            score_modules["srmr"] = {
                "module": srmr_metric,
                "args": {
                    "n_cochlear_filters": config.get("n_cochlear_filters", 23),
                    "low_freq": config.get("low_freq", 125),
                    "min_cf": config.get("min_cf", 128),
                    "max_cf": config.get("max_cf", 128),
                    "fast": config.get("fast", True),
                    "norm": config.get("norm", False),
                },
            }
            logging.info("Initiate srmr successfully")

        elif config["name"] == "noresqa":
            if not use_gt:
                logging.warning("Cannot use noresqa because no gt audio is provided")
                continue

            logging.info("Loading noresqa metrics with reference")

            from versa.utterance_metrics.noresqa import (noresqa_metric,
                                                         noresqa_model_setup)

            noresqa_model = noresqa_model_setup(
                metric_type=config.get("metric_type", 0), use_gpu=use_gpu
            )
            score_modules["noresqa"] = {
                "module": noresqa_metric,
                "args": {
                    "metric_type": config.get("metric_type", 0),
                    "model": noresqa_model,
                },
            }
            logging.info("Initiate noresqa score metric successfully.")

        elif config["name"] == "speaking_rate":
            logging.info("Loading speaking rate metrics without reference")
            from versa import speaking_rate_metric, speaking_rate_model_setup

            # Load whisper model if it is already loaded
            if "whisper_wer" in score_modules.keys():
                speaking_rate_model = score_modules["whisper_wer"]["args"]
            else:
                speaking_rate_model = speaking_rate_model_setup(
                    model_tag=config.get("model_tag", "default"),
                    beam_size=config.get("beam_size", 1),
                    text_cleaner=config.get("text_cleaner", "whisper_basic"),
                    use_gpu=use_gpu,
                )

            score_modules["speaking_rate"] = {
                "module": speaking_rate_metric,
                "args": speaking_rate_model,
            }
            logging.info("Initiate speaking rate metric successfully.")

        elif config["name"] == "asr_match":
            if not use_gt:
                logging.warning("Cannot use asr_match because no gt audio is provided")
                continue

            logging.info("Loading asr_match metric with reference text")
            from versa import asr_match_metric, asr_match_setup

            # Load whisper model if it is already loaded
            if "whisper_wer" in score_modules.keys():
                asr_model = score_modules["whisper_wer"]["args"]
            elif "speaking_rate" in score_modules.keys():
                asr_model = score_modules["speaking_rate"]["args"]
            else:
                asr_model = asr_match_setup(
                    model_tag=config.get("model_tag", "default"),
                    beam_size=config.get("beam_size", 1),
                    text_cleaner=config.get("text_cleaner", "whisper_basic"),
                    use_gpu=use_gpu,
                )

            score_modules["asr_match"] = {
                "module": asr_match_metric,
                "args": asr_model,
            }
            logging.info("Initiate asr_match metric successfully")

        elif config["name"] == "lid":
            logging.info("Loading language identification metric")
            from versa import language_id, owsm_lid_model_setup

            owsm_model = owsm_lid_model_setup(
                model_tag=config.get("model_tag", "default"),
                nbest=config.get("nbest", 3),
                use_gpu=use_gpu,
            )

            score_modules["lid"] = {
                "module": language_id,
                "args": owsm_model,
            }

        elif config["name"] == "audiobox_aesthetics":
            logging.info("Loading audiobox aesthetics metric")
            from versa import (audiobox_aesthetics_score,
                               audiobox_aesthetics_setup)

            audiobox_model = audiobox_aesthetics_setup(
                model_path=config.get("model_path", None),
                batch_size=config.get("batch_size", 1),
                precision=config.get("precision", "bf16"),
                cache_dir=config.get("cache_dir", "versa_cache/audiobox"),
                use_huggingface=config.get("use_huggingface", True),
                use_gpu=use_gpu,
            )

            score_modules["audiobox_aesthetics"] = {
                "module": audiobox_aesthetics_score,
                "args": {"model": audiobox_model},
            }
            logging.info("Initiate audiobox aesthetics metric successfully")

        elif "qwen2_audio" in config["name"]:
            logging.info("Loading qwen2-audio model")
            from versa import qwen2_model_setup

            if "qwen2_audio" not in score_modules.keys():
                qwen_model = qwen2_model_setup(
                    model_tag=config.get("model_tag", "default"),
                )
                score_modules["qwen2_audio"] = {
                    "module": qwen_model,
                    "start_prompt": config.get("start_prompt", None),
                }

            # 1. Speaker Characteristics
            if config["name"] == "qwen2_audio_speaker_count":
                from versa import qwen2_speaker_count_metric

                score_modules["qwen2_audio_speaker_count"] = {
                    "module": qwen2_speaker_count_metric,
                    "prompt": config.get("prompt", None),
                }
            elif config["name"] == "qwen2_audio_speaker_gender":
                from versa import qwen2_speaker_gender_metric

                score_modules["qwen2_audio_speaker_gender"] = {
                    "module": qwen2_speaker_gender_metric,
                    "prompt": config.get("prompt", None),
                }
            elif config["name"] == "qwen2_audio_speaker_age":
                from versa import qwen2_speaker_age_metric

                score_modules["qwen2_audio_speaker_age"] = {
                    "module": qwen2_speaker_age_metric,
                    "prompt": config.get("prompt", None),
                }
            elif config["name"] == "qwen2_audio_speech_impairment":
                from versa import qwen2_speech_impairment_metric

                score_modules["qwen2_audio_speech_impairment"] = {
                    "module": qwen2_speech_impairment_metric,
                    "prompt": config.get("prompt", None),
                }

            # 2. Voice Properties
            elif config["name"] == "qwen2_audio_voice_pitch":
                from versa import qwen2_voice_pitch_metric

                score_modules["qwen2_audio_voice_pitch"] = {
                    "module": qwen2_voice_pitch_metric,
                    "prompt": config.get("prompt", None),
                }
            elif config["name"] == "qwen2_audio_pitch_range":
                from versa import qwen2_pitch_range_metric

                score_modules["qwen2_audio_pitch_range"] = {
                    "module": qwen2_pitch_range_metric,
                    "prompt": config.get("prompt", None),
                }
            elif config["name"] == "qwen2_audio_voice_type":
                from versa import qwen2_voice_type_metric

                score_modules["qwen2_audio_voice_type"] = {
                    "module": qwen2_voice_type_metric,
                    "prompt": config.get("prompt", None),
                }
            elif config["name"] == "qwen2_audio_speech_volume_level":
                from versa import qwen2_speech_volume_level_metric

                score_modules["qwen2_audio_speech_volume_level"] = {
                    "module": qwen2_speech_volume_level_metric,
                    "prompt": config.get("prompt", None),
                }

            # 3. Speech Content
            elif config["name"] == "qwen2_audio_language":
                from versa import qwen2_language_metric

                score_modules["qwen2_audio_language"] = {
                    "module": qwen2_language_metric,
                    "prompt": config.get("prompt", None),
                }
            elif config["name"] == "qwen2_audio_speech_register":
                from versa import qwen2_speech_register_metric

                score_modules["qwen2_audio_speech_register"] = {
                    "module": qwen2_speech_register_metric,
                    "prompt": config.get("prompt", None),
                }
            elif config["name"] == "qwen2_audio_vocabulary_complexity":
                from versa import qwen2_vocabulary_complexity_metric

                score_modules["qwen2_audio_vocabulary_complexity"] = {
                    "module": qwen2_vocabulary_complexity_metric,
                    "prompt": config.get("prompt", None),
                }
            elif config["name"] == "qwen2_audio_speech_purpose":
                from versa import qwen2_speech_purpose_metric

                score_modules["qwen2_audio_speech_purpose"] = {
                    "module": qwen2_speech_purpose_metric,
                    "prompt": config.get("prompt", None),
                }

            # 4. Speech Delivery
            elif config["name"] == "qwen2_audio_speech_emotion":
                from versa import qwen2_speech_emotion_metric

                score_modules["qwen2_audio_speech_emotion"] = {
                    "module": qwen2_speech_emotion_metric,
                    "prompt": config.get("prompt", None),
                }
            elif config["name"] == "qwen2_audio_speech_clarity":
                from versa import qwen2_speech_clarity_metric

                score_modules["qwen2_audio_speech_clarity"] = {
                    "module": qwen2_speech_clarity_metric,
                    "prompt": config.get("prompt", None),
                }
            elif config["name"] == "qwen2_audio_speech_rate":
                from versa import qwen2_speech_rate_metric

                score_modules["qwen2_audio_speech_rate"] = {
                    "module": qwen2_speech_rate_metric,
                    "prompt": config.get("prompt", None),
                }
            elif config["name"] == "qwen2_audio_speaking_style":
                from versa import qwen2_speaking_style_metric

                score_modules["qwen2_audio_speaking_style"] = {
                    "module": qwen2_speaking_style_metric,
                    "prompt": config.get("prompt", None),
                }
            elif config["name"] == "qwen2_audio_laughter_crying":
                from versa import qwen2_laughter_crying_metric

                score_modules["qwen2_audio_laughter_crying"] = {
                    "module": qwen2_laughter_crying_metric,
                    "prompt": config.get("prompt", None),
                }

            # 5. Interaction Patterns
            elif config["name"] == "qwen2_audio_overlapping_speech":
                from versa import qwen2_overlapping_speech_metric

                score_modules["qwen2_audio_overlapping_speech"] = {
                    "module": qwen2_overlapping_speech_metric,
                    "prompt": config.get("prompt", None),
                }

            # 6. Recording Environment
            elif config["name"] == "qwen2_audio_speech_background_environment":
                from versa import qwen2_speech_background_environment_metric

                score_modules["qwen2_audio_speech_background_environment"] = {
                    "module": qwen2_speech_background_environment_metric,
                    "prompt": config.get("prompt", None),
                }
            elif config["name"] == "qwen2_audio_recording_quality":
                from versa import qwen2_recording_quality_metric

                score_modules["qwen2_audio_recording_quality"] = {
                    "module": qwen2_recording_quality_metric,
                    "prompt": config.get("prompt", None),
                }
            elif config["name"] == "qwen2_audio_channel_type":
                from versa import qwen2_channel_type_metric

                score_modules["qwen2_audio_channel_type"] = {
                    "module": qwen2_channel_type_metric,
                    "prompt": config.get("prompt", None),
                }

            logging.info(
                "Initiate qwen2 audio metric: {} successfully".format(config["name"])
            )
    return score_modules


def process_cache_info(cache_info, score_modules, output_file):
    batch_score_info = []
    for utt_info in cache_info:
        key, gen_wav, gt_wav, gen_sr, text = utt_info
        utt_score = {"key": key}
        utt_score.update(
            use_score_modules(score_modules, gen_wav, gt_wav, gen_sr, text)
        )
        batch_score_info.append(utt_score)
        if output_file is not None:
            output_file.write(f"{utt_score}\n")
    return batch_score_info


def use_score_modules(score_modules, gen_wav, gt_wav, gen_sr, text=None):
    utt_score = {}

    # general cache information to reduce recaculation
    general_cache = {
        "whisper_hyp_text": None,
    }
    for key in score_modules.keys():
        if key == "mcd_f0":
            score = score_modules[key]["module"](
                gen_wav, gt_wav, gen_sr, **score_modules[key]["args"]
            )
        elif key == "signal_metric":
            try:
                score = score_modules[key]["module"](gen_wav, gt_wav)
            except ValueError as e:
                logging.warning(
                    "Value error in signal metric. Usually due to silence audio: {}".format(
                        e
                    )
                )
                continue
        elif key == "warpq":
            score = score_modules[key]["module"](
                score_modules[key]["model"], gen_wav, gt_wav, gen_sr
            )
        elif key == "nisqa":
            try:
                score = score_modules[key]["module"](
                    score_modules[key]["model"],
                    gen_wav,
                    gen_sr,
                )
            except ValueError as e:
                logging.warning(
                    "Value error in NISQA metric. Usually due to silence audio: {}".format(
                        e
                    )
                )
                continue
        elif key == "discrete_speech":
            score = score_modules[key]["module"](
                score_modules[key]["model"],
                gen_wav,
                gt_wav,
                gen_sr,
            )
        elif key == "pseudo_mos":
            score = score_modules[key]["module"](
                gen_wav, gen_sr, **score_modules[key]["args"]
            )
        elif key in ["pesq", "stoi", "estoi"]:
            score = score_modules[key]["module"](gen_wav, gt_wav, gen_sr)
        elif key == "visqol":
            score = score_modules[key]["module"](
                score_modules[key]["args"]["api"],
                score_modules[key]["args"]["api_fs"],
                gen_wav,
                gt_wav,
                gen_sr,
            )
        elif key == "speaker":
            score = score_modules[key]["module"](
                score_modules[key]["args"]["model"], gen_wav, gt_wav, gen_sr
            )
        elif key == "sheet_ssqa":
            score = score_modules[key]["module"](
                score_modules[key]["args"]["model"],
                gen_wav,
                gen_sr,
                use_gpu=score_modules[key]["args"]["use_gpu"],
            )
        elif key == "squim_ref":
            score = score_modules[key]["module"](gen_wav, gt_wav, gen_sr)
        elif key == "squim_no_ref":
            score = score_modules[key]["module"](gen_wav, gen_sr)
        elif key == "nomad":
            score = score_modules[key]["module"](
                score_modules[key]["model"], gen_wav, gt_wav, gen_sr
            )
        elif key == "espnet_wer" or key == "owsm_wer" or key == "whisper_wer":
            score = score_modules[key]["module"](
                score_modules[key]["args"],
                gen_wav,
                text,
                gen_sr,
            )
            if key == "whisper_wer":
                general_cache["whisper_hyp_text"] = score["whisper_hyp_text"]
        elif key in ["scoreq_ref", "emotion"]:
            score = score_modules[key]["module"](
                score_modules[key]["model"], gen_wav, gt_wav, gen_sr
            )
        elif key in ["scoreq_nr", "se_snr"]:
            score = score_modules[key]["module"](
                score_modules[key]["model"], gen_wav, gen_sr
            )
        elif key in ["pam", "asvspoof_score"]:
            score = score_modules[key]["module"](
                score_modules[key]["model"], gen_wav, fs=gen_sr
            )
        elif key in ["vad", "lid"]:
            score = score_modules[key]["module"](
                score_modules[key]["args"],
                gen_wav,
                gen_sr,
            )
        elif key == "pysepm":
            score = score_modules[key]["module"](gen_wav, gt_wav, fs=gen_sr)
        elif key == "srmr":
            score = score_modules[key]["module"](gen_wav, fs=gen_sr)
        elif key == "noresqa":
            score = score_modules[key]["module"](
                score_modules[key]["args"]["model"],
                gen_wav,
                gt_wav,
                fs=gen_sr,
                metric_type=score_modules[key]["args"]["metric_type"],
            )
        elif key == "speaking_rate":
            cache_text = None
            if general_cache.get("whisper_hyp_text", None) is not None:
                cache_text = utt_score["whisper_hyp_text"]
            score = score_modules[key]["module"](
                score_modules[key]["args"],
                gen_wav,
                cache_text,
                gen_sr,
            )
            if cache_text is None:
                general_cache["whisper_hyp_text"] = score["whisper_hyp_text"]
        elif key == "asr_match":
            cache_text = None
            if general_cache.get("whisper_hyp_text", None) is not None:
                cache_text = utt_score["whisper_hyp_text"]
            score = score_modules[key]["module"](
                score_modules[key]["args"],
                gen_wav,
                gt_wav,
                cache_text,
                gen_sr,
            )
            if cache_text is None:
                general_cache["whisper_hyp_text"] = score["whisper_hyp_text"]
        elif key == "audiobox_aesthetics":
            score = score_modules[key]["module"](
                score_modules[key]["args"]["model"],
                gen_wav,
                gen_sr,
            )
        elif "qwen2_audio" in key:
            if key == "qwen2_audio":
                continue  # skip the base model, only use the specific metrics
            # Support qwen2_audio metrics
            score = score_modules[key]["module"](
                score_modules["qwen2_audio"]["module"],
                gen_wav,
                gen_sr,
                custom_prompt=score_modules[key]["prompt"],
            )
            print("score: {}".format(score), flush=True)
        else:
            raise NotImplementedError(f"Not supported {key}")

        logging.info(f"Score for {key} is {score}")
        utt_score.update(score)
    return utt_score


def list_scoring(
    gen_files,
    score_modules,
    gt_files=None,
    text_info=None,
    output_file=None,
    io="kaldi",
    batch_size=1,
):
    if output_file is not None:
        f = open(output_file, "w", encoding="utf-8")
    else:
        f = None

    score_info = []
    cache_info = []  # for batch processing
    for key in tqdm(gen_files.keys()):
        # Step1: load source speech and conduct basic checks
        gen_sr, gen_wav = load_audio(gen_files[key], io)
        gen_wav = wav_normalize(gen_wav)

        # length check
        if not check_minimum_length(gen_wav.shape[0] / gen_sr, score_modules.keys()):
            logging.warning(
                "audio {} (generated, length {}) is too short to be evaluated with some metric metrics, skipping".format(
                    key, gen_wav.shape[0] / gen_sr
                )
            )
            continue

        # Step2: load reference (gt) speech and conduct basic checks
        if gt_files is not None:
            if key not in gen_files.keys():
                logging.warning(
                    "key {} not found in ground truth files though provided, skipping".format(
                        key
                    )
                )
                continue

            gt_sr, gt_wav = load_audio(gt_files[key], io)
            gt_wav = wav_normalize(gt_wav)

            # check ground truth audio files
            if check_all_same(gt_wav):
                logging.warning(
                    "gt audio of key {} has only the same value, skipping".format(key)
                )
                continue

            # length check
            if not check_minimum_length(gt_wav.shape[0] / gt_sr, score_modules.keys()):
                logging.warning(
                    "audio {} (ground truth, length {}) is too short to be evaluated with many metrics, skipping".format(
                        key, gt_wav.shape[0] / gt_sr
                    )
                )
                continue
        else:
            gt_wav = None
            gt_sr = None

        # Step3: load text information if provided
        text = None
        if text_info is not None:
            if key not in text_info.keys():
                logging.warning(
                    "key {} not found in ground truth transcription though provided, skipping".format(
                        key
                    )
                )
                continue
            else:
                text = text_info[key]

        # Step4: check if the sampling rate of generated and gt audio are the same
        if gt_sr is not None and gen_sr > gt_sr:
            logging.warning(
                "Resampling the generated audio to match the ground truth audio"
            )
            gen_wav = librosa.resample(gen_wav, orig_sr=gen_sr, target_sr=gt_sr)
            gen_sr = gt_sr
        elif gt_sr is not None and gen_sr < gt_sr:
            logging.warning(
                "Resampling the ground truth audio to match the generated audio"
            )
            gt_wav = librosa.resample(gt_wav, orig_sr=gt_sr, target_sr=gen_sr)

        # Step5: cache for batch processing
        utterance_info = (key, gen_wav, gt_wav, gen_sr, text)

        cache_info.append(utterance_info)
        if len(cache_info) == batch_size:
            # Process after a batch is collected
            score_info.extend(process_cache_info(cache_info, score_modules, f))
            cache_info = []
        else:
            # continue collect the batch
            continue

    # Process left-over batch
    score_info.extend(process_cache_info(cache_info, score_modules, f))

    logging.info("Scoring completed and save score at {}".format(output_file))
    return score_info


def load_summary(score_info):
    summary = {}
    for key in score_info[0].keys():
        if "text" in key or "vad" in key or "language" in key or key == "key":
            # NOTE(jiatong): skip text cases
            continue
        summary[key] = sum([score[key] for score in score_info])
        if "_wer" not in key and "_cer" not in key:
            # Average for non-WER/CER metrics
            summary[key] /= len(score_info)
    return summary


def load_corpus_modules(score_config, cache_folder=".cache", use_gpu=False, io="kaldi"):
    score_modules = {}
    for config in score_config:
        if config["name"] == "fad":
            logging.info("Loading FAD evaluation with specific models...")
            # TODO(jiatong): fad will automatically use cuda if detected
            # need to sync to the same space
            from versa import fad_scoring, fad_setup

            fad_info = fad_setup(
                fad_embedding=config.get("fad_embedding", "default"),
                baseline=config.get("baseline_audio", "missing"),
                cache_dir=config.get("cache_dir", cache_folder),
                use_inf=config.get("use_inf", False),
                io=io,
            )

            fad_key = "fad_{}".format(config.get("model", "default"))

            score_modules[fad_key] = {
                "module": fad_scoring,
                "args": fad_info,
            }
            logging.info(
                "Initiate {} calculation evaluation successfully.".format(fad_key)
            )
        elif config["name"] == "kid":
            logging.info("Loading KID evaluation with specific models...")
            from versa import kid_scoring, kid_setup

            kid_info = kid_setup(
                model_tag=config.get("model_tag", "default"),
                model_path=config.get("model_path", None),
                model_config=config.get("model_config", None),
                use_gpu=use_gpu,
            )
            kid_key = "kid_{}".format(config.get("model", "default"))
            score_modules[kid_key] = {
                "module": kid_scoring,
                "args": kid_info,
            }
            logging.info(
                "Initiate {} calculation evaluation successfully.".format(kid_key)
            )

    return score_modules


def corpus_scoring(
    gen_files,
    score_modules,
    base_files=None,
    text_info=None,
    output_file=None,
):
    score_info = {}
    for key in score_modules.keys():
        if key.startswith("fad"):
            fad_info = score_modules[key]["args"]
            if base_files is not None:
                fad_info["baseline"] = base_files
            elif fad_info["baseline"] == "missing":
                raise ValueError("Baseline audio not provided for FAD")
            score_result = score_modules[key]["module"](
                gen_files, fad_info, key_info=key
            )
        elif key.startswith("kld"):
            kid_info = score_modules[key]["args"]
            if base_files is not None:
                kid_info["baseline"] = base_files
            elif kid_info["baseline"] == "missing":
                raise ValueError("Baseline audio not provided for FAD")
            score_result = score_modules[key]["module"](
                gen_files, kid_info, key_info=key
            )
        else:
            raise NotImplementedError("Not supported {}".format(key))
        score_info.update(score_result)

    if output_file is not None:
        with open(output_file, "w") as f:
            yaml.dump(score_info, f)
    return score_info

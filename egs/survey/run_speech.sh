#!/bin/bash

# Originally prepared by Haibin Wu (2024)
# Adapted by Jiatong Shi (2025)

. ../activate_python.sh

stage=2
# pred_path=data/LibriSpeech/test-clean/prepared/ori.scp
pred_path=librispeech_discrete_hifigan.scp
gt_path=data/LibriSpeech/test-clean/prepared/ori.scp

# download data
if [ $stage -eq 0 ]; then
    echo stage $stage: Prepare data

    if [ ! -d data/LibriSpeech/test-clean ]; then
        mkdir -p data
        wget http://www.openslr.org/resources/12/test-clean.tar.gz -P ./data
        (cd ./data && tar -xvzf test-clean.tar.gz)
        rm data/test-clean.tar.gz
    fi

    if [ ! -d data/LibriSpeech/test-clean/prepared ]; then
        python scripts/survey/prepare_librispeech-test-clean.py --root_dir data/LibriSpeech/test-clean
    fi

fi

# Evaluation
if [ $stage -eq 1 ]; then
    result_path="librispeech_discrete_hifigan"

    echo stage $stage: Evaluation
    if test -f ${result_path}; then
        echo ${result_path} exists
    else
        python versa/bin/scorer.py \
            --score_config egs/survey/configs/speech.yaml \
            --use_gpu True \
            --gt ${gt_path} \
            --pred ${pred_path} \
            --output_file ${result_path}
    fi

    python scripts/survey/average_result.py --file_path ${result_path} >> ${result_path}
    tail -n 20 ${result_path}
fi

# Word error rate evaluation
# Currently use whisper_wer
if [ $stage -eq 2 ]; then
    wer_result_path="librispeech_discrete_hifigan_asr"

    echo stage $stage: Word error rate evaluation

    if test -f ${wer_result_path}; then
        echo ${wer_result_path} exists
    else
        python versa/bin/scorer.py \
            --score_config egs/separate_metrics/wer_english.yaml \
            --use_gpu True \
            --gt ${gt_path} \
            --pred ${pred_path} \
            --text data/LibriSpeech/test-clean/prepared/transcriptions.txt \
            --output_file ${wer_result_path}
    fi

    python scripts/survey/get_wer.py --file_path ${wer_result_path} >> ${wer_result_path}
    tail -n 1 ${wer_result_path}
fi

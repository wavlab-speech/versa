# Originally prepared by Haibin Wu (2024)
# Adapted by Jiatong Shi (2025)

stage=0
pred_path=data/LibriSpeech/test-clean/prepared/ori.scp
gt_path=data/LibriSpeech/test-clean/prepared/ori.scp
tag=audio

# download data
if [ $stage -eq 0 ]; then
    echo stage $stage: Prepare data

    # musdb
    if [ ! -d data/musdb/test ]; then
        wget https://zenodo.org/records/3338373/files/musdb18hq.zip -P data
        (cd data && unzip musdb18hq.zip -d ./musdb)
        rm data/musdb18hq.zip
    fi

    if [ ! -d data/musdb/prepared ]; then
        python scripts/survey/prepare_musdb.py --main_directory data/musdb/ --output_dir data/musdb/prepared --chunk_length 5.0
    fi

    # audioset
    if [ ! -d data/audioset ]; then
        python scripts/survey/prepare_audioset-test.py --output_dir data/audioset
    fi

fi

# Evaluation
if [ $stage -eq 1 ]; then
    result_path="test_result_${tag}"

    echo stage $stage: Evaluation
    if test -f ${result_path}; then
        echo ${result_path} exists
    else
        python versa/bin/scorer.py \
            --score_config egs/survey/general.yaml \
            --use_gpu True \
            --gt ${gt_path} \
            --pred ${pred_path} \
            --output_file ${result_path}
    fi

    python scripts/survey/average_result.py --file_path ${result_path} >> ${result_path}
    tail -n 20 ${result_path}
fi


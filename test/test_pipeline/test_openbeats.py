import logging
import os

import yaml
import numpy as np

from versa.scorer_shared import (
    find_files,
    list_scoring,
    load_score_modules,
)

TEST_INFO = {
    "openbeats_embedding_extraction": np.array([-0.42187455, -0.6287595, 0.1792216]),
    "openbeats_embedding_similarity": 1.0,
}


def test_openbeats_embedding_extraction(embedding_result):
    """Test OpenBEATs embedding extraction."""
    # Read embedding
    assert (
        "embedding_file" in embedding_result
    ), "Embedding result does not contain 'embedding_file'"
    with open(embedding_result["embedding_file"], "rb") as f:
        embedding_result["embedding"] = np.load(f)

    assert embedding_result["embedding"].shape[:-1] == (
        1,
        48,
    ), f'The frame size is off. Expected (1,48) but got {embedding_result["embedding"].shape[:-1]}'
    summary_value = embedding_result["embedding"][0, :3, 0]
    if np.any(
        np.abs(TEST_INFO["openbeats_embedding_extraction"] - summary_value) > 1e-3
    ):
        raise ValueError(
            "Value issue in the test case, might be some issue in scorer {}".format(
                "openbeats_embedding_extraction"
            )
        )


def test_openbeats_embedding_similarity(embedding_result):
    """Test OpenBEATs embedding similarity."""
    assert (
        "similarity_score" in embedding_result
    ), "Embedding result does not contain 'similarity_score'"
    similarity_score = embedding_result["similarity_score"]
    assert (
        np.abs(TEST_INFO["openbeats_embedding_similarity"] - similarity_score) < 1e-3
    ), "Similarity score should be 1.0, got {}".format(similarity_score)


def test_openbeats_class_prediction(class_prediction_result):
    """Test OpenBEATs class prediction."""
    assert (
        "class_probabilities" in class_prediction_result
    ), "Class prediction result does not contain 'class_probabilities'"
    class_probabilities = class_prediction_result["class_probabilities"]
    assert isinstance(
        class_probabilities, np.ndarray
    ), "Class probabilities should be a numpy array"
    assert class_probabilities.shape == (
        1,
        10,
    ), "Expected shape (1, 10) for class probabilities"
    print("Class probabilities: {}".format(class_probabilities), flush=True)


def info_update():

    # find files
    if os.path.isdir("test/test_samples/test2"):
        gen_files = find_files("test/test_samples/test2")

    logging.info("The number of utterances = %d" % len(gen_files))

    with open("egs/separate_metrics/openbeats.yaml", "r", encoding="utf-8") as f:
        score_config = yaml.full_load(f)

    score_modules = load_score_modules(
        score_config,
        use_gt=True,
        use_gpu=False,
    )

    assert len(score_config) > 0, "no scoring function is provided"

    score_info = list_scoring(
        gen_files, score_modules, gt_files=gen_files, output_file=None, io="soundfile"
    )

    test_openbeats_embedding_extraction(score_info[0])
    test_openbeats_embedding_similarity(score_info[0])
    # test_openbeats_class_prediction(score_info[0])

    print("check successful", flush=True)


if __name__ == "__main__":
    info_update()

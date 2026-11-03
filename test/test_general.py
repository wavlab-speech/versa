import os
import math
import pytest
import yaml
import logging

from versa.scorer_shared import (
    configure_metric_cache_dirs,
    configure_shared_cache_environment,
    find_files,
    list_scoring,
    load_score_modules,
    load_summary,
)


def test_configure_metric_cache_dirs_uses_shared_root_and_preserves_overrides(
    tmp_path,
):
    configs = [
        {"name": "SongEval"},
        {"name": "qwen2_audio_speaker_age"},
        {"name": "qwen2_audio_speaker_gender"},
        {"name": "speaking_rate"},
        {"name": "whisper_wer", "cache_dir": "/existing/shared-whisper"},
    ]

    configured = configure_metric_cache_dirs(configs, tmp_path)

    assert configured == [
        {"name": "SongEval", "cache_dir": str(tmp_path / "songeval")},
        {
            "name": "qwen2_audio_speaker_age",
            "cache_dir": str(tmp_path / "huggingface"),
        },
        {
            "name": "qwen2_audio_speaker_gender",
            "cache_dir": str(tmp_path / "huggingface"),
        },
        {"name": "speaking_rate", "cache_dir": str(tmp_path / "whisper")},
        {"name": "whisper_wer", "cache_dir": "/existing/shared-whisper"},
    ]
    assert configs == [
        {"name": "SongEval"},
        {"name": "qwen2_audio_speaker_age"},
        {"name": "qwen2_audio_speaker_gender"},
        {"name": "speaking_rate"},
        {"name": "whisper_wer", "cache_dir": "/existing/shared-whisper"},
    ]


def test_configure_metric_cache_dirs_without_root_returns_original_config():
    configs = [{"name": "songeval"}]

    assert configure_metric_cache_dirs(configs) is configs


def test_configure_shared_cache_environment(monkeypatch, tmp_path):
    for variable in (
        "VERSA_CACHE_DIR",
        "VERSA_HF_CACHE_DIR",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
        "NEMO_CACHE_DIR",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.setenv(variable, "previous-cache-value")

    cache_root = configure_shared_cache_environment(tmp_path / "shared-cache")

    hf_cache = cache_root / "huggingface"
    assert os.environ["VERSA_CACHE_DIR"] == str(cache_root)
    assert os.environ["VERSA_HF_CACHE_DIR"] == str(hf_cache)
    assert os.environ["HF_HOME"] == str(hf_cache)
    assert os.environ["HF_HUB_CACHE"] == str(hf_cache)
    assert os.environ["TRANSFORMERS_CACHE"] == str(hf_cache)
    assert os.environ["HF_DATASETS_CACHE"] == str(hf_cache / "datasets")
    assert os.environ["TORCH_HOME"] == str(cache_root / "torch")
    assert os.environ["NEMO_CACHE_DIR"] == str(cache_root / "nemo")
    assert os.environ["XDG_CACHE_HOME"] == str(cache_root)


# The expected test values from the original code
TEST_INFO = {
    "mcd": 5.045226506332897,
    "f0rmse": 20.281004489942777,
    "f0corr": -0.07540903652440145,
    "sdr": 4.873952979593643,
    "sir": float("inf"),
    "sar": 4.873952979593643,
    "si_snr": 1.0702790021896362,
    "ci_sdr": 4.873951435089111,
    "pesq": 1.5722705125808716,
    "stoi": 0.007625108859647406,
    "utmos": 1.9074374437332153,
    "dns_overall": 1.4526055142443377,
    "dns_p808": 2.09430193901062,
    "plcmos": 3.16458740234375,
    "spk_similarity": 0.895357072353363,
    "singmos": 2.0403144359588623,
    "sheet_ssqa": 1.5056276321411133,
    "se_sdr": -10.220576129334987,
    "se_sar": -10.220576129334987,
    "se_si_snr": -16.837026596069336,
    "se_ci_sdr": -10.220579147338867,
    "torch_squim_mos": 3.948253870010376,
    "torch_squim_stoi": 0.6027805209159851,
    "torch_squim_pesq": 1.1683127880096436,
    "torch_squim_si_sdr": -11.109052658081055,
    "dpam_distance": 0.4179654121398926,
    "cdpam_distance": 0.039433546364307404,
    "dnsmos_pro_bvcc": 1.1717286109924316,
    "dnsmos_pro_nisqa": 1.4733699560165405,
    "dnsmos_pro_vcc2018": 1.930935263633728,
}

TEST_TOLERANCE = {
    "se_si_snr": 1e-3,
}


@pytest.fixture
def setup_paths():
    """Setup the paths for test files"""
    gen_path = "test/test_samples/test2"
    gt_path = "test/test_samples/test1"

    # Check if directories exist
    gen_exists = os.path.isdir(gen_path)
    gt_exists = os.path.isdir(gt_path)

    return {
        "gen_path": gen_path,
        "gt_path": gt_path,
        "gen_exists": gen_exists,
        "gt_exists": gt_exists,
    }


@pytest.fixture
def load_config():
    """Load the scoring configuration"""
    with open("egs/speech.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_scoring_pipeline(setup_paths, load_config, caplog):
    """Test the scoring pipeline against expected values"""
    caplog.set_level(logging.INFO)

    paths = setup_paths
    score_config = load_config

    # Skip test if required directories don't exist
    if not paths["gen_exists"] or not paths["gt_exists"]:
        pytest.skip("Required test directories not found")

    # Get files
    gen_files = find_files(paths["gen_path"])
    gt_files = find_files(paths["gt_path"])

    # Log number of utterances
    logging.info(f"The number of utterances = {len(gen_files)}")

    # Load score modules
    score_modules = load_score_modules(
        score_config,
        use_gt=(True if gt_files is not None else False),
        use_gpu=False,
    )

    # Ensure we have scoring functions
    assert len(score_config) > 0, "no scoring function is provided"

    # Run scoring
    score_info = list_scoring(
        gen_files, score_modules, gt_files, output_file=None, io="soundfile"
    )

    # Get summary
    summary = load_summary(score_info)
    print(f"Summary: {summary}", flush=True)

    # Validate results
    for key in summary:
        if key not in TEST_INFO:
            pytest.fail(f"Unexpected metric: {key}")

        # Handle infinite values
        if math.isinf(TEST_INFO[key]) and math.isinf(summary[key]):
            continue

        # Special case for plcmos which is non-deterministic
        if key == "plcmos":
            continue

        # Check if values match within tolerance
        tolerance = TEST_TOLERANCE.get(key, 1e-4)
        assert abs(TEST_INFO[key] - summary[key]) <= tolerance, (
            f"Value issue in scorer {key}: expected {TEST_INFO[key]}, "
            f"got {summary[key]}"
        )

    print("Check successful", flush=True)


def test_missing_gt_files(setup_paths, load_config):
    """Test behavior when ground truth files are missing"""
    paths = setup_paths
    score_config = load_config

    # Skip if generated files don't exist
    if not paths["gen_exists"]:
        pytest.skip("Required test directories not found")

    # Get files
    gen_files = find_files(paths["gen_path"])

    # Load score modules with use_gt=False
    score_modules = load_score_modules(
        score_config,
        use_gt=False,
        use_gpu=False,
    )

    # This test is more of a smoke test to ensure no crashes
    score_info = list_scoring(
        gen_files, score_modules, None, output_file=None, io="soundfile"
    )

    summary = load_summary(score_info)
    assert summary is not None, "Summary should not be None"


def test_empty_score_config():
    """Test behavior with empty score config"""
    empty_config = {}

    with pytest.raises(AssertionError, match="no scoring function is provided"):
        load_score_modules(empty_config, use_gt=False, use_gpu=False)

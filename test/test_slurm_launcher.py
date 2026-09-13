"""Exercise resource safeguards without a Slurm installation."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "launch_slurm.sh"


@pytest.fixture
def launch(tmp_path):
    """Run the real launcher with a scheduler that only records arguments."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    scheduler = bin_dir / "sbatch"
    scheduler.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['JOB_LOG'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "print(123)\n"
    )
    scheduler.chmod(0o755)
    pred = tmp_path / "pred.scp"
    score = tmp_path / "scores"
    job_log = tmp_path / "jobs.jsonl"

    def run(*args, settings=None, chunks="1", text="a a.wav\nb b.wav\n", reply=""):
        pred.write_text(text)
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("SBATCH_")
            and key
            not in {
                "MAX_JOBS",
                "MAX_TOTAL_CPU_HOURS",
                "CPUS",
                "MEM",
                "CPU_TIME",
                "GPU_TIME",
                "CPU_OTHER_OPTS",
                "GPU_OTHER_OPTS",
            }
        }
        env.update(PATH=f"{bin_dir}:{env['PATH']}", JOB_LOG=str(job_log))
        env.update(settings or {})
        result = subprocess.run(
            ["bash", str(LAUNCHER), str(pred), "None", str(score), chunks, *args],
            env=env,
            input=reply,
            text=True,
            capture_output=True,
            timeout=10,
        )
        jobs = (
            [json.loads(line) for line in job_log.read_text().splitlines()]
            if job_log.exists()
            else []
        )
        return result, jobs, score

    return run


@pytest.mark.parametrize("mode,count", [("--cpu-only", 1), ("--gpu-only", 1), ("", 2)])
def test_submission_modes(launch, mode, count):
    result, jobs, _ = launch(*([mode] if mode else []), "--yes")
    assert result.returncode == 0, result.stderr
    assert len(jobs) == count
    assert all(job[job.index("--cpus-per-task") + 1] == "4" for job in jobs)


@pytest.mark.parametrize(
    "time", ["60", "60:00", "01:00:00", "0-1", "0-1:00", "0-1:00:00"]
)
def test_slurm_time_formats(launch, time):
    result, jobs, _ = launch(
        "--cpu-only",
        "--yes",
        settings={"CPU_TIME": time, "CPUS": "1", "MAX_TOTAL_CPU_HOURS": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert len(jobs) == 1


@pytest.mark.parametrize("time", ["00:30:00", "00:00:01", "0-0:30:00"])
def test_subhour_limits_never_round_down(launch, time):
    result, jobs, score = launch(
        "--cpu-only",
        "--yes",
        settings={"CPU_TIME": time, "CPUS": "100", "MAX_TOTAL_CPU_HOURS": "1"},
    )
    assert result.returncode != 0
    assert "MAX_TOTAL_CPU_HOURS" in result.stderr
    assert not jobs
    assert not score.exists()


@pytest.mark.parametrize(
    "time", ["bad", "0", "0-0:00:00", "1:", "1:2:3:4", "9999999999"]
)
def test_invalid_or_unlimited_time_fails_closed(launch, time):
    result, jobs, score = launch("--cpu-only", "--yes", settings={"CPU_TIME": time})
    assert result.returncode != 0
    assert not jobs
    assert not score.exists()


@pytest.mark.parametrize("name", ["MAX_JOBS", "MAX_TOTAL_CPU_HOURS", "CPUS", "MEM"])
@pytest.mark.parametrize("value", ["0", "-1", "oops", "1+1", "99999999999999999999"])
def test_invalid_resource_values(launch, name, value):
    result, jobs, _ = launch("--yes", settings={name: value})
    assert result.returncode != 0
    assert name in result.stderr
    assert not jobs


@pytest.mark.parametrize("chunks", ["0", "-1", "1+1", "99999999999999999999"])
def test_invalid_split_size(launch, chunks):
    result, jobs, _ = launch("--yes", chunks=chunks)
    assert result.returncode != 0
    assert "split_size" in result.stderr
    assert not jobs


def test_large_estimate_cannot_overflow(launch):
    result, jobs, _ = launch(
        "--yes", settings={"CPUS": "999999999", "GPU_TIME": "999999999-00:00:00"}
    )
    assert result.returncode != 0
    assert not jobs


def test_job_cap_uses_actual_chunks(launch):
    result, jobs, _ = launch(
        "--cpu-only", "--yes", chunks="008", settings={"MAX_JOBS": "02"}
    )
    assert result.returncode == 0, result.stderr
    assert len(jobs) == 2


def test_combined_job_cap(launch):
    result, jobs, _ = launch("--yes", chunks="2", settings={"MAX_JOBS": "3"})
    assert result.returncode != 0
    assert "MAX_JOBS" in result.stderr
    assert not jobs


def test_combined_cpu_budget(launch):
    result, jobs, _ = launch("--yes", settings={"MAX_TOTAL_CPU_HOURS": "50"})
    assert result.returncode != 0
    assert not jobs


def test_stale_chunks_cannot_bypass_budget(launch):
    result, jobs, _ = launch("--cpu-only", "--yes", chunks="2")
    assert result.returncode == 0
    result, later_jobs, _ = launch("--cpu-only", "--yes", settings={"MAX_JOBS": "1"})
    assert result.returncode != 0
    assert "fresh score directory" in result.stderr
    assert later_jobs == jobs


@pytest.mark.parametrize("reply,success", [("n\n", True), ("", False), ("y\n", True)])
def test_confirmation(launch, reply, success):
    result, jobs, _ = launch("--cpu-only", reply=reply)
    assert (result.returncode == 0) == success
    assert len(jobs) == (1 if reply == "y\n" else 0)


def test_empty_input(launch):
    result, jobs, _ = launch("--yes", text="")
    assert result.returncode != 0
    assert not jobs


def test_missing_final_newline(launch):
    result, jobs, _ = launch("--cpu-only", "--yes", text="a a.wav")
    assert result.returncode == 0, result.stderr
    assert len(jobs) == 1


@pytest.mark.parametrize(
    "options",
    [
        "--time=2-0",
        "--cpus-per-task=100",
        "--array=1-100",
        "--exclusive",
        "-n100",
        "--nodes=10",
    ],
)
def test_extra_options_cannot_override_estimate(launch, options):
    result, jobs, _ = launch(
        "--cpu-only", "--yes", settings={"CPU_OTHER_OPTS": options}
    )
    assert result.returncode != 0
    assert "extra sbatch option" in result.stderr
    assert not jobs


def test_scheduling_options_are_forwarded(launch):
    result, jobs, _ = launch(
        "--cpu-only",
        "--yes",
        settings={"CPU_OTHER_OPTS": "--account=lab --constraint=gpu*"},
    )
    assert result.returncode == 0, result.stderr
    assert "--account=lab" in jobs[0]
    assert "--constraint=gpu*" in jobs[0]
    assert jobs[0][jobs[0].index("--ntasks") + 1] == "1"


def test_implicit_array_cannot_bypass_job_limit(launch):
    result, jobs, _ = launch("--yes", settings={"SBATCH_ARRAY_INX": "1-100"})
    assert result.returncode != 0
    assert "SBATCH_ARRAY_INX" in result.stderr
    assert not jobs

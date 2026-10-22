# Additional pretrained MOS adapters

WhiSQA and DNSMOS-C implement M4 and M5 of the
October 2026 MOS survey. Both are optional, reference-free
utterance metrics. Discovery imports neither backend nor downloads weights.
These adapters have contract tests; numerical parity with real pretrained weights
and domain/hardware qualification remain pending.

## WhiSQA

Use a separate environment with the upstream requirements compatible with your
PyTorch installation. Obtain the pinned author code and Git LFS checkpoints:

```sh
git clone https://github.com/leto19/WhiSQA tools/WhiSQA
git -C tools/WhiSQA checkout 0ad007e97093aa54c0c8cdac1af3e6ad6305af3f
git -C tools/WhiSQA lfs pull
python -m pip install -r tools/WhiSQA/requirements.txt
```

Configure `repo_path` and `head: single` or `head: multi`, as in
[the example](../egs/separate_metrics/whisqa.yaml). The checkpoint must contain
actual LFS weights, not a pointer file. The author architecture also loads its
Whisper encoder through Transformers; pre-cache that encoder for offline runs.
Keep the checkout available at runtime. Upstream uses a generic `models` Python
package; a conflicting package produces an error and requires a fresh process.

Input must be nonempty finite mono audio; VERSA resamples it to 16 kHz and passes
the entire waveform to the author model, preserving its internal duration policy.
The upstream CLI multiplies raw predictions by five; this adapter does likewise
without clipping. `whisqa_mos` is always emitted. Multi-head mode additionally
emits `whisqa_noisiness`, `whisqa_coloration`, `whisqa_discontinuity`, and
`whisqa_loudness`, in the author's order. These quality ratings are distinct from
physical measurements. Code/checkpoint repository: MIT; encoder terms are separate.

## DNSMOS-C

Obtain the survey-pinned TorchScript assets:

```sh
git clone https://github.com/Hope-Liang/DNSMOS-C tools/DNSMOS-C
git -C tools/DNSMOS-C checkout 861583995c10a8bf7b18462081c99ae2c258b16e
git -C tools/DNSMOS-C lfs pull
python -m pip install torch librosa numpy scipy
```

Configure both `dataset` (`NISQA`, `BVCC`, or `Tencent`) and `checkpoint`, following
[the example](../egs/separate_metrics/dnsmos_c.yaml). Dataset selection records the
training identity; it cannot prove the provenance of an arbitrary local file.
Use the corresponding `runs/<dataset>/model_best.pt`. Only load trusted TorchScript
assets. The author repository declares MIT.

The adapter resamples mono audio to 16 kHz using librosa's default resampler (as in the upstream dataset loaders), repeats
short clips and crops long clips to the first ten seconds, then computes the
author default log-magnitude STFT (320-sample FFT/window, 160-sample hop, centered
librosa defaults, magnitude clipped to 1e-7…1e7 before log10). It handles both
plain and tuple TorchScript outputs. `dnsmos_c_mean` is the predicted MOS;
`dnsmos_c_variance` is uncertainty in score-squared units, not a MOS rating or a
validated confidence interval. Reports retain its descriptive statistics but omit
quality best/worst labels and group rankings. No additional calibration or clipping is applied.

## Reproducibility and qualification

Preserve the code commits above and record SHA-256 hashes of downloaded weight
payloads and encoder revisions with evaluation results. Local assets allow offline
DNSMOS-C runs; WhiSQA additionally requires a populated encoder cache. CPU is the
default; `use_gpu: true` selects CUDA explicitly. Setup never selects MPS silently.

Before claiming pretrained parity, compare these wrappers against the pinned
upstream inference on a fixed audio manifest spanning silence, short inputs,
resampling and duration boundaries, with stated numerical tolerances. Contract
tests use synthetic models and verify shape, preprocessing and output semantics;
they do not establish human-label accuracy or numerical checkpoint parity.

# Distill-MOS

Optional reference-free speech quality prediction from Microsoft's
[Distill-MOS](https://github.com/microsoft/Distill-MOS), described in
[Stahl and Gamper (ICASSP 2025)](https://arxiv.org/abs/2502.05356).
This is distinct from BaleYang/DistilMOS.

Install into the environment used for VERSA with
`bash tools/install_distill_mos.sh` (or set `PYTHON` to that environment's Python).
Use `egs/separate_metrics/distill_mos.yaml` with the existing scorer configuration
option. `use_gpu: true` selects CUDA; the default is CPU.
The optional backend is imported only when constructing the metric.

The output key is `distill_mos`: native MOS, nominally 1–5, higher is better.
No reference or text is needed. Supply a nonempty finite mono waveform and
`metadata.sample_rate` (integer Hz, default 16000). VERSA converts to float32 and
uses TorchAudio resampling to 16 kHz, matching the author example. Stereo must be
converted to mono before this adapter. No clipping or extra normalization is
applied by VERSA.

The upstream forward pass zero-pads to 122880 samples (7.68 seconds), uses
uniformly spaced overlapping windows with maximum 16000-sample hop for longer
clips, peak-normalizes each window, averages predictions, and converts to MOS.
VERSA passes the full waveform with `segmenting_in_forward=True` to retain this
behavior, including silence and duration boundaries.

## Asset identity and validation

The installer pins code and bundled `distillmos/weights/distill_mos_v7.pt` to
`f676ccec11ab6205073b2689448a32f8d76a8083`. Upstream declares MIT for code/model;
retain the upstream license and cite the paper when using the model.
Dependencies are supplied by the optional upstream package, including Torch,
TorchAudio and XLS-R-SQA. Backend requirements may exceed VERSA's base floor.
The weights live inside the installed package; after installing the package and
its dependencies, setup uses that local file without a separate model download.
For offline deployments, prepare the pinned package and dependency wheels in an
online environment, then install them from a local wheelhouse.

Adapter tests cover default setup, lazy dependency errors, CPU inference context,
resampling, full-waveform forwarding at duration boundaries, and invalid inputs.
Real-checkpoint numerical parity, downloaded weight SHA-256, fixed versioned audio
fixtures, CPU/GPU throughput and held-out domain validation remain pending.
The adapter is implemented; research results and domain suitability have not
been independently reproduced.

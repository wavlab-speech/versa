## Contributor Guidelines

To implement a new metric to versa includes the following steps:

### Step1: Prepare metric
You may add the metric implementation in the following sub-directories (`versa/corpus_metrics`, `versa/utterance_metrics`, `versa/sequence_metrics`). Specifically,
- corpus_metrics: works for metrics that need the whole corpora to compute the metric (e.g., FAD or WER).
- utterance_metrics: works for utterance-level metrics
- sequence_metrics (will be deprecated in later versions and merged to utterance_metrics): stands for metrics that need comparing two feature sequences.

The typical format of the metric setup includes two functions, one for model setup, and the other for inference. Please refer to `versa/utterance/metrics/speaker.py` for an example of the implementation.

For special cases where the model setup is simple or not needed, we can simplify only one inference function without the setup function as exemplified in `versa/utterance_metrics/stoi.py`

**Special note**: 
- Please consider adding a simple test function at the end of the implementation.
- For consistency, we will have some fixed naming conventions to follow:
    - For the setup function, we will have an argument of `use_gpu` which is default set to `False`.
    - For the inference function, the previous preprocessor can provide five arguments so far (If you need more, please contact Jiatong Shi for further discussion on the interface):
        - model: the inference model to use
        - pred_x: audio signal to be evaluated
        - fs: audio signal's sampling rate
        - gt_x: [optional] the reference audio signal (automatically handled in the previous parts, the reference signal should also have the same sampling rate as the target signal to be evaluated)
        - ref_text: [optional] additional text information. It can be either the transcription for WER or the text description for audio signals.
- Toolkit development: to link the toolkit modeling to other implementations, the 
    - We recommend using the original tool/interface as much as possible if they can already be formatted into the current interface. However, if it is not, we recommend the following hacking options to link their methods to VERSA. This option also works for those packages that need very specific versions of the dependency packages :
        - 1. fork their repo
        - 2. add customized interface
        - 3. add localized install options to `tools`

### Step2: Register the metrics to the scoring list
For the second step, please add your metrics to the scoring list in `versa/scorer_shared.py`. Notably, you are expected to add the new metrics in both `load_score_modules()` and `use_score_modules()`.

At this step, please define a unique key for your metric to differentiate it from others. By referring to the key, you can declare the setup function in `load_score_modules()` and the inference function in `use_score_modules()`. Please refer to the existing examples so that they are following the same setup.

### Step3: Docs, Tests, Examples, and Code-wrapping up
At this step, the major implementation has been done and we mainly focus on the docs, test functions, examples, and code-wrapping up.

For Docs, please add your metrics to the `README.md` (List of Metrics Section). If the metrics need external tools from installers at `tools`, please include that with the `[ ]` mark in the first field (column).

For Tests, please add the local test functions at the corresponding metrics scripts temporarily (we will enable CI test in later stages).
- For metrics included in the default setup, you can add the test metric value in `test/test_general.py`
- For metrics not included in the default installation, you can add the test function in `test/test_{metric_name}.py`

For Examples, please put a separate `yaml` style configuration file in `egs/separate_metrics` following other examples.

For Code-wrapping up, we highly recommend you use `black` and `isort` to format your added scripts.

# Supplementary Material for Submission #3457

## Setup Instructions

1. Run the following command to add the current directory to the python path:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export TF_ENABLE_ONEDNN_OPTS=0
```

2. Install miniconda or anaconda. Then, using the following command, create the environments required for the experiments:

```bash
conda env create -f torch-env.yml
conda env create -f ollm.yml
```

Due to conflicting dependencies, please try running RQ1 and RQ2 in the `torch-env` environment, the baseline results for RQ3 in the `ollm` environment, and the CA and CD experiments in RQ3 using the `torch-env` environment.

## Setup Local Inference for Producing Baseline Results in RQ3

Install miniconda by following the [instructions](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) based on your operating system.
Then, use the following command to set up a local inference environment:

```bash
conda create -n sglang
conda activate sglang
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]>=0.4.3.post2" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
python -m sglang.launch_server --model-path <model-name> --port 49999 --host 0.0.0.0
```

Look for a log message like `max_total_num_tokens=15919, max_prefill_tokens=16384, max_running_requests=2049, context_len=12000` in the logs to get the maximum number of tokens that can be processed by the server.
Then, rename the `.env.example` file to `.env` and set the `MAX_TOTAL_NUM_TOKENS` environment variable to the maximum number of tokens that can be processed by the server. This is required for running the inference tasks.


## RQ1

Please refer to [tcav/README.md](tcav/README.md) for instructions on replicating the results of RQ1 and generating the plots.

## RQ2

The median test accuracy for `Javadoc`, `Inline`, and `Multiline` across all models is shown below:

![](tcav/figures/rq2_all.png)

To replicate the results of RQ2, please refer to [tcav/README.md](tcav/README.md) for instructions on training the classifiers and generating the plots.

## RQ3

The tasks are located in the `tasks` directory. To replicate the results of RQ3 for each task, please refer to the respective `README.md` files in the task directory within the `tasks` directory. The `Sheets` directory contains the performance results of the experiments for each task, including the baseline results, Concept Activation (CA), and Concept Deactivation (CD) results across all LLMs, metrics, and tasks.

## RQ4

The task instructions for each of the 10 SE tasks are located in the `rq4/prompts.json` file. We list them below for reference:

```json
{
    "code summarization": "Summarize the following function in one sentence.\n{code}",
    "code translation": "Translate this code from Java to Python.\n{code}",
    "test generation": "Write test cases for the following class to check its correctness.\n{code}",
    "code completion": "Complete the last line of the following code snippet.\n{code}",
    "fault localization": "Identify the buggy line(s) in the following code that cause it to fail.\n{code}",
    "program repair": "Fix the bug in the following code snippet to make it work as intended.\n{code}",
    "vulnerability detection": "Does the following code contain any security vulnerabilities? If yes, describe them.\n{code}",
    "code review": "Review the following code change and provide suggestions for improvement.\n{code}",
    "code refactoring": "Optimize the following code to improve performance without changing its output.\n{code}",
    "code documentation": "Generate inline comments and a docstring for the following function.\n{code}"
}
```

The `{code}` placeholder will be replaced with the actual code snippet for each task.
To run RQ4 experiments, please refer run the following command:

```bash
cd rq4
python run_rq4.py
```

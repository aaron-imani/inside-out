# Code Refinement Task

Replication package retrieved from the paper [Exploring the Potential of ChatGPT in Automated Code Refinement: An Empirical Study](https://dl.acm.org/doi/10.1145/3597503.3623306) [ICSE 2024].

The content is modified to accommodate our experiments with the filtered Java dataset. See the original code [here](https://sites.google.com/view/chatgptcodereview).

## Dataset

To filter the dataset, run the following command:

```bash
python filter.py
python split_comment_types.py
```

## RQ3

To run the experiments for RQ3, use the following command:

```bash
chmod +x run_ca.sh
chmod +x run_cd.sh
./run_ca.sh -m <model_name>
./run_cd.sh -m <model_name>
```

where `<model-name>` can be one of the following:
- `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ`
- `Qwen/Qwen2.5-32B-Instruct-AWQ`
- `Qwen/QwQ-32B-AWQ`
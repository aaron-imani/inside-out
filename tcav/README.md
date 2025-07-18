# RQ1-2: Testing with Concept Activation Vectors (TCAV)

Adopted from the repository of the paper [Uncovering Safety Risks of Large Language Models through Concept Activation Vector](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d3a230d716e65afab578a8eb31a8d25f-Abstract-Conference.html).



## Training Concept Classifiers 

To train the classifiers from scratch, follow these steps:

1. Run the `fetch_top_apache_repos.sh` script to fetch the top Apache repositories. This will create a directory named `repos` containing the repositories.

```bash
chmod +x fetch_top_apache_repos.sh
./fetch_top_apache_repos.sh
```

3. The next step is to parse the repositories to extract positive and negative examples. This can be done by running the following command that will create a `dataset` directory containing the dataset files for each comment concept:

```bash
python make_dataset.py
```

4. To train the classifiers for the Comment concept across all LLM, run the following commands:

```bash
chmod +x run_rq1.sh
./run_rq1.sh qwen2.5
./run_rq1.sh qwen2.5-coder
./run_rq1.sh qwq
```

5. To train the classifiers for the comment subtypes, i.e., `Javadoc`, `Inline`, and `Multiline`, run the following commands: 

```bash
chmod +x run_rq2.sh
./run_rq2.sh qwen2.5
./run_rq2.sh qwen2.5-coder
./run_rq2.sh qwq
```

This creates a directory called `classifiers` with the trained classifiers for each model and concept. The directory structure should look like this:

```
classifiers
├── qwen2.5
│   ├── comment
│   ├── inline
│   ├── javadoc
│   └── multiline
├── qwen2.5-coder
│   ├── comment
│   ├── inline
│   ├── javadoc
│   └── multiline
└── qwq
    ├── comment
    ├── inline
    ├── javadoc
    └── multiline
```

### Classification Results

The classification results can be found in the `classification_results` directory. Within this directory, you will find subdirectories for each model. Each subdirectory contains the results for each sample size and train size with the `sample size_train size` format across all concepts. The train size varies between 0.05, 0.1, 0.25, and 0.5. Regarding sample sizes, you can use the following cheat sheet to find the results for the specific concept:

- Javadoc: 760_X where X is the train size (e.g., 738_0.05, 738_0.1, etc.)
- Comment: 762_0 where X is the train size (e.g., 762_0.05, 762_0.1, etc.)
- Inline: 756_X where X is the train size (e.g., 756_0.05, 756_0.1, etc.)
- Multiline: 738_X where X is the train size (e.g., 738_0.05, 738_0.1, etc.)

## Generating Plots

To generate the plots, you should have the classifiers trained. Then, you can use the `generate_plots.ipynb` notebook to generate the plots for RQ1 and RQ2.
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from classifier_manager import ClassifierManager
from embedding_manager import load_embds_manager
from instructions import *
from model_extraction import ModelExtraction
from tabulate import tabulate
from tqdm import tqdm

from make_dataset import get_sample_dataset, sample_sizes

cur_dir = Path(__file__).parent.resolve()

torch.cuda.empty_cache()


def get_embeddings(insts, split, label, round_no):
    if split == 'train':
        file_path = (
            train_embeddings_path /  f"{split}_{label}_{round_no}.pth"
        )
    else:
        file_path = (
            embedings_path / f"{split}_{label}_{args.test_size}.pth"
        )

    if file_path.exists():
        return load_embds_manager(file_path)
    # else:
    positivity = 0 if label == "pos" else 1
    vectors = llm.extract_embds(insts[split][positivity])
    torch.save(vectors, file_path)
    return vectors


parser = ArgumentParser()
parser.add_argument("model_nickname", type=str, help="Nickname of the model to analyze")
parser.add_argument("dataset_name", type=str, help="Name of the dataset to analyze")
parser.add_argument(
    "-t",
    "--train-size",
    type=float,
    default=0.5,
    help="Proportion of data to use for training",
)
parser.add_argument(
    "--test-size",
    type=int,
    default=None,
    help="Number of samples to use for testing",
)
parser.add_argument(
    "-r",
    "--rounds",
    type=int,
    default=1,
    help="Number of rounds to run the analysis",
)
parser.add_argument(
    "--reset",
    action="store_true",
    help="Reset the embeddings and classifiers before running the analysis",
)
parser.add_argument(
    "-s",
    "--sample-size",
    type=int,
    default=0,
    help="Number of samples to use for each round",
)

args = parser.parse_args()
model_nickname = args.model_nickname
dataset_name = args.dataset_name
train_size = args.train_size

if args.sample_size == 0:
    sample_size = sample_sizes[dataset_name]
    args.test_size = sample_size // 2

    print('Train Size:', int(train_size * sample_size))
    print('Test Size:', args.test_size)
else:
    sample_size = args.sample_size

embedings_path = cur_dir / "dataset_embeddings" / model_nickname / dataset_name
embedings_path.mkdir(parents=True, exist_ok=True)

train_embeddings_path = embedings_path / f"{sample_size}_{train_size}"
train_embeddings_path.mkdir(parents=True, exist_ok=True)

figures_path = cur_dir / "figures" / model_nickname / dataset_name / f"{sample_size}_{train_size}"
figures_path.mkdir(parents=True, exist_ok=True)

csv_path = cur_dir / "classification_results" / model_nickname / f"{sample_size}_{train_size}" 
csv_path.mkdir(parents=True, exist_ok=True)

classifier_path = cur_dir / "classifiers" / model_nickname / dataset_name / f"{sample_size}_{train_size}"
classifier_path.mkdir(parents=True, exist_ok=True)

data_path = cur_dir / "dataset" / model_nickname / f"{sample_size}_{train_size}"
data_path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    if args.reset:
        print("Resetting embeddings and classifiers...")
        for path in embedings_path.glob("*.pth"):
            path.unlink()
        for path in classifier_path.glob("*.pth"):
            path.unlink()

    llm = ModelExtraction(model_nickname)

    layers_accuracies = []
    accuracy_reports = []

    for i in tqdm(
        range(args.rounds),
        desc="Processing rounds",
        unit="round",
        leave=False,
        colour="GREEN",
    ):
        # instruction_path = (
        #     cur_dir / "instructions" / f"{dataset_name}_instructions_{train_size}.json"
        # )
        # if instruction_path.exists():
        #     insts = load_instructions_from_file(instruction_path)
        # else:
        round_no = i + 1
        sample_dataset = get_sample_dataset(
            args.dataset_name, sample_size=sample_size, seed=round_no
        )
        dataset_path = (
            data_path
            / f"{dataset_name}_sample{round_no}.csv"
        )
        sample_dataset.to_csv(dataset_path, index=False)

        insts = load_instructions_by_size(
            dataset_name=dataset_name,
            label_list=[True, False],
            dataset_df=sample_dataset,
            train_size=train_size,
            test_size=args.test_size,
            seed=42,
        )
        # train_df = pd.DataFrame(insts["train"][0] + insts["train"][1])
        # train_df.to_csv(
        #     data_path / f"{dataset_name}_train_{round_no}.csv", index=False
        # )
        # test_df = pd.DataFrame(insts["test"][0] + insts["test"][1])
        # test_df.to_csv(
        #     data_path / f"{dataset_name}_test_{round_no}.csv", index=False
        # )
        # save_instructions_to_file(insts, instruction_path)

        pos_train_embds = get_embeddings(insts, "train", "pos", round_no)
        neg_train_embds = get_embeddings(insts, "train", "neg", round_no)
        pos_test_embds = get_embeddings(insts, "test", "pos", round_no)
        neg_test_embds = get_embeddings(insts, "test", "neg", round_no)

        clfr = ClassifierManager(dataset_name)

        clfr_path = classifier_path / f"{round_no}.pth"
        # if clfr_path.exists():
        #     clfr = load_classifier_manager(clfr_path)
        # else:
        clfr.fit(pos_train_embds, neg_train_embds, pos_test_embds, neg_test_embds)
        torch.save(clfr, clfr_path)

        # print(clfr.testacc)
        accuracies = {j + 1: clfr.testacc[j] for j in range(len(clfr.testacc))}
        accuracies["concept"] = dataset_name
        accuracies["train_size"] = train_size
        accuracies["mean"] = sum(clfr.testacc) / len(clfr.testacc)
        accuracies["max"] = max(clfr.testacc)
        accuracies["max_layer"] = np.argmax(clfr.testacc) + 1  # +1 for 1-based indexing
        layers_accuracies.append(accuracies)
        accuracy_reports.append(
            {
                "concept": dataset_name,
                "round": round_no,
                "max_accuracy": accuracies["max"] * 100,
                "max_layer": accuracies["max_layer"],
            }
        )
        # print(
        #     f'Max Accuracy Layer: {accuracies["max_layer"]} with {accuracies["max"]*100:.2f}% accuracy.'
        # )

        plt.plot(
            list(map(lambda x: x * 100, clfr.testacc)),
            marker=".",
            label=f"Round {round_no}",
            alpha=0.3,
        )
        del clfr, pos_train_embds, neg_train_embds, pos_test_embds, neg_test_embds

    table = tabulate(
        accuracy_reports,
        headers="keys",
        tablefmt="simple_outline",
        floatfmt=".2f",
    )
    print(f"Accuracy Report for {dataset_name} with {model_nickname}\n{table}\n")
    print("--" * 30)

    plt.title(f"Classifier Accuracy for {dataset_name} with {model_nickname}")
    plt.xlabel("Layer")
    plt.yticks(np.arange(0, 101, 10))
    plt.ylabel("Test Accuracy of $\\mathregular{{P_m}}$ (%)")
    plt.legend(loc="best")

    plt.savefig(figures_path / f"accuracy.png")
    plt.savefig(figures_path / f"accuracy.pdf")

    df = pd.DataFrame(layers_accuracies)
    # get summary statistics
    summary = df.describe().T
    summary.to_csv(csv_path / f"{dataset_name}_accuracy.csv")

    accuracies_path = Path(
        csv_path / f"layers_accuracies.csv"
    )
    if accuracies_path.exists():
        df.to_csv(accuracies_path, index=False, mode="a", header=False)
    else:
        df.to_csv(accuracies_path, index=False)
# plt.show()

import json
import random

import pandas as pd


def load_instructions_by_size(
    dataset_name: str,
    label_list: list[str],
    dataset_df: pd.DataFrame,
    train_size: float = 1.0,
    test_size: int = None,
    seed: int = None,
):
    assert 0 < train_size <= 1.0, "train_size should be in (0, 1]"
    ret = {
        "dataset_name": dataset_name,
        "label_list": label_list,
        "train": [],
        "test": [],
    }

    # To make sure that each label has the same number of samples
    if test_size is not None:
        test_size //= len(label_list)
        print('Test size per label:', test_size)
    train_size /= len(label_list)
    
    df = dataset_df[dataset_df["DatasetName"] == dataset_name]
    for label in label_list:
        label_df = df[df["Label"] == label]
        values = label_df["Instruction"].values.tolist()

        # if seed is not None:
        #     random.seed(seed)  # For reproducibility

        # random.shuffle(values)

        train_number = int(len(label_df) * train_size)
        print(f'Training size for {label}:', train_number)

        ret["train"].append(values[:train_number])

        if train_size < 1.0:
            if test_size is None or test_size >= len(values) - train_number:
                ret["test"].append(values[train_number:])
            else:
                ret["test"].append(
                    values[-test_size:]
                )
    return ret


def load_instructions_from_file(file_path):
    with open(file_path, "r") as file:
        instructions = json.load(file)
    return instructions


def save_instructions_to_file(instructions, file_path):
    with open(file_path, "w") as file:
        json.dump(instructions, file, indent=2)


def load_instructions_by_flag(
    dataset_name: str,
    label_list: list[str],
    instructions_path: str = "./instructions/instructions.csv",
):
    ret = {
        "dataset_name": dataset_name,
        "label_list": label_list,
        "train": [],
        "test": [],
    }
    df = pd.read_csv(instructions_path)
    df = df[df["DatasetName"] == dataset_name]

    for label in label_list:
        label_df = df[df["Label"] == label]
        label_df = label_df.sample(frac=1).reset_index(drop=True)

        train_df = label_df[label_df["TrainTestFlag"] == "Train"]
        test_df = label_df[label_df["TrainTestFlag"] == "Test"]
        ret["train"].append(train_df["Instruction"].values.tolist())
        ret["test"].append(test_df["Instruction"].values.tolist())

    return ret

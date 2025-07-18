import pandas as pd
from instructions import load_instructions_by_size
from plot import plot_reduction
from train_cav import (
    data_path,
    dataset_name,
    figures_path,
    get_embeddings,
    model_nickname,
    train_size,
)

for i in range(args.rounds):
    # round_no = 1
    round_no = i + 1
    try:
        sample_dataset = pd.read_csv(
            data_path / f"{dataset_name}_sample{round_no}_{train_size}.csv"
        )
    except FileNotFoundError:
        print(
            f"Sample dataset for {dataset_name} round {round_no} not found. Please run the train_cav script first."
        )
        exit(1)

    insts = load_instructions_by_size(
        dataset_name=dataset_name,
        label_list=[True, False],
        dataset_df=sample_dataset,
        train_size=train_size,
        seed=42,
    )
    # save_instructions_to_file(insts, instruction_path)

    pos_train_embds = get_embeddings(
        insts, model_nickname, dataset_name, "train", "pos", round_no
    )
    neg_train_embds = get_embeddings(
        insts, model_nickname, dataset_name, "train", "neg", round_no
    )
    pos_test_embds = get_embeddings(
        insts, model_nickname, dataset_name, "test", "pos", round_no
    )
    neg_test_embds = get_embeddings(
        insts, model_nickname, dataset_name, "test", "neg", round_no
    )

    fig = plot_reduction(
        insts,
        pos_train_embds,
        neg_train_embds,
        pos_test_embds,
        neg_test_embds,
        n_layer=64,
        char_range=500,
    )
    fig.write_image(
        figures_path / f"tSNE_{round_no}.pdf",
    )

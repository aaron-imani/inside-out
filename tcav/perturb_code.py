from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from classifier_manager import *
from model_generation import ModelGeneration
from perturbation import Perturbation
from tqdm import tqdm

cur_dir = Path(__file__).parent.resolve()
parser = ArgumentParser()
parser.add_argument("prompts_path", type=str, help="Path to the prompts csv file")
parser.add_argument("model_nickname", help="Nickname of the model to use")
parser.add_argument("concept", help="Which concept to pertub embeddings for")
parser.add_argument("classifier_id", type=int, help="ID of the classifier to use")
parser.add_argument(
    "-t",
    "--target-confidence",
    type=float,
    default=0.05,
    help="The minimum classification confidence of the concept classifier of each layer to activate perturbation",
)
parser.add_argument(
    "-a",
    "--accuracy-threshold",
    type=float,
    default=0.5,
    help="The minimum accuracy of the concept classifier of each layer to activate perturbation",
)
parser.add_argument(
    "-o",
    "--output-path",
    type=str,
    default="output.csv",
    help="Path to the output CSV file",
)
args = parser.parse_args()

try:
    prompts_df = pd.read_csv(args.prompts_path)
except Exception:
    print(
        f"Error opening prompts file: {args.prompts_path}. Please check the path and make sure the file exists and is a CSV file."
    )
    exit(1)

classifier_path = cur_dir / "classifiers" / args.model_nickname / args.concept

clfr = load_classifier_manager(classifier_path, args.classifier_id)
llm_gen = ModelGeneration(args.model_nickname)
pert = Perturbation(
    clfr,
    target_probability=args.target_confidence,
    accuracy_threshold=args.accuracy_threshold,
)

progress_bar = tqdm(
    prompts_df.iterrows(),
    total=prompts_df.shape[0],
    leave=False,
    desc="Generating outputs",
    unit="prompt",
    colour="CYAN",
)

for i, row in progress_bar:
    question = row["prompt"]
    if pd.isna(question):
        continue  # Skip rows with NaN in the 'prompt' column

    llm_gen.unset_perturbation()  # Reset perturbation before each generation
    progress_bar.set_description_str("Generating original output")
    output_original = llm_gen.generate(question)

    llm_gen.set_perturbation(pert)
    progress_bar.set_description_str("Generating perturbed output")
    output_perturbed = llm_gen.generate(question)

    prompts_df.at[i, "original_output"] = output_original["completion"]
    prompts_df.at[i, "perturbed_output"] = output_perturbed["completion"]

prompts_df.to_csv(args.output_path, index=False)
print(f"Outputs saved to {args.output_path}")

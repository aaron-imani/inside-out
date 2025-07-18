import json
from pathlib import Path
import pandas as pd
from argparse import ArgumentParser

cur_dir = Path(__file__).parent.resolve()

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Calculate the original performance per concept for the CodeReview dataset."
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        type=str,
        help="The base directory to use for the calculation. ",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
    )
    
    args = parser.parse_args()
    args.base_dir = Path(args.base_dir).resolve()

    model_results = pd.read_csv(args.base_dir / args.model_name / "direct_performance.csv", dtype={'id': str})


    with open(cur_dir / "comment_splits.json", "r") as f:
        comment_splits = json.load(f)

    comment_types = ['comment', 'inline', 'multiline']
    perturbations = ['uncomment', 'comment']
    performance_metrics = ['em', 'em_trim', 'bleu', 'bleu_trim']

    report = {}

    for comment_type in comment_types:
        for perturbation in perturbations:
            base_performance_str = "commented" if perturbation == "uncomment" else "comments_removed"
            concept_performance = model_results[model_results['id'].isin(comment_splits[comment_type])].copy()
            if base_performance_str not in report:
                report[base_performance_str] = {}

            report[base_performance_str][comment_type] = {}
            for metric in performance_metrics:
                metric_col = f"{base_performance_str}_{metric}"
                report[base_performance_str][comment_type][metric] = concept_performance[metric_col].mean()


    with open(args.base_dir / args.model_name / "per_concept_report.json", "w") as f:
        json.dump(report, f, indent=4)
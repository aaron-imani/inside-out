import json
import re
from pathlib import Path

import pandas as pd
from jsonlines import jsonlines
from openai import OpenAI
from tabulate import tabulate
from tqdm import tqdm

from common import env_loader
from common.cot import CoTAgent
from common.MAD.interactive import call_mad
from rq1.ICSE24_CodeReview_new.evaluation import myeval
from rq2.tcav.perturbator import Perturbator

base_url = env_loader.INFERENCE_URL
model_name = env_loader.MODEL_NAME
cur_dir = Path(__file__).resolve().parent

client = OpenAI(base_url=base_url, api_key="None")


def generate_direct_prompt(old_without_minus, review):
    """
    P1 + Scenario Description.

    Note that this prompt was reported to be the best performing prompt in the paper.
    See Table 4 in the [paper](https://dl.acm.org/doi/pdf/10.1145/3597503.3623306) for more details.
    """
    prompt = ""
    prompt += (
        "As a developer, imagine you've submitted a pull request and"
        " your team leader requests you to make a change to a piece of code."
        " The old code being referred to in the hunk of code changes is:\n"
    )
    prompt += "```\n{}\n```\n".format(old_without_minus)
    prompt += "There is the code review for this code:\n"
    prompt += review
    prompt += "\nPlease generate the revised code according to the review and send it wrapped between ``` and ```."
    return prompt


def get_chatgptapi_response(prompt, temperature=1.0):
    if model_name.find("QwQ") == -1:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an experienced developer."},
                {"role": "user", "content": prompt},
            ],
            temperature=float(temperature),
        )
    else:
        # Based on QwQ recommended settings
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an experienced developer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            top_p=0.95,
            max_tokens=25000,
        )

    # print(response)
    answer = response.choices[0].message.content
    # print("answer: ",answer)
    result = re.search(r"```(.*)```", answer, re.DOTALL)
    # print("result: ",result)
    if result:
        newcode = result.group(1)
    else:
        newcode = "no code"
        print("no code:")
        print(answer)
    return newcode, answer


def query_model(prompt, groundtruth):
    code = "no code"
    raw_answer = "no answer"

    try:
        if not args.concept:
            code, raw_answer = get_chatgptapi_response(prompt, temperature=0)
        else:
            code = pertubator.query(prompt)

    except Exception as e:
        print("Error in querying model:", e)
        code = ""

    # calc the em and bleu

    em, em_trim, _, _, bleu, bleu_trim = myeval(groundtruth, code)

    return code, em, em_trim, bleu, bleu_trim


def run_model():
    # The steps for codereview.jsonl and codereview_new.jsonl are essentially the same,
    # with the only difference being the method of storing data in the database.
    # read_path = "codereview.jsonl"
    read_path = "codereview_new_filtered.jsonl"
    # results_path = cur_dir / "rq1_results" / model_name
    # if args.perturbation:
    #     results_path = results_path / args.perturbation / args.concept

    result_records = []
    records_csv = []
    with jsonlines.open(read_path, "r") as f:
        for line in tqdm(list(f), desc="Refining code"):
            data = line
            if args.perturbation == 'uncomment' and str(data['_id']) not in comment_splits[args.concept]:
                continue
            
            old = data["commented"]
            new = data["new"]
            review = data["review"]
            report = []
            record = {"id": data["_id"]}

            # old_without_minus = []
            # for line in old.split("\n"):
            #     old_without_minus.append(line[1:])
            # old_without_minus = "\n".join(old_without_minus)
            if prompting_method != "mad":
                prompt = generate_direct_prompt(old, review)
            else:
                prompt = generate_mad_prompt(old, review)

            if not args.concept or (args.concept and args.perturbation == "uncomment"):
                llm_response, em, em_trim, bleu, bleu_trim = query_model(prompt, new)
                data["commented_performance"] = {
                    "em": em,
                    "em_trim": em_trim,
                    "bleu": bleu,
                    "bleu_trim": bleu_trim,
                }
                data['llm_response'] = llm_response
                for key in data["commented_performance"]:
                    record[f"commented_{key}"] = data["commented_performance"][key]

                report.append(data["commented_performance"])

            if not args.concept or (args.concept and args.perturbation == "comment"):
                prompt = generate_direct_prompt(data["comments_removed"], review)
                llm_response, em, em_trim, bleu, bleu_trim = query_model(prompt, new)

                data["comments_removed_performance"] = {
                    "em": em,
                    "em_trim": em_trim,
                    "bleu": bleu,
                    "bleu_trim": bleu_trim,
                }
                data['llm_response'] = llm_response
                for key in data["comments_removed_performance"]:
                    record[f"comments_removed_{key}"] = data[
                        "comments_removed_performance"
                    ][key]

                report.append(data["comments_removed_performance"])
                records_csv.append(record)
                print(tabulate(report, headers="keys", tablefmt="grid"))
            result_records.append(data)

    df = pd.DataFrame(records_csv)
    df.to_csv(results_path / "performance.csv", index=False)

    # Print the average performance for each setting, e.g., commented and comments_removed
    print("Average performance for each setting:")
    performance_report = {}
    settings = ["commented", "comments_removed"] if not args.concept else ["commented"]

    for setting in settings:
        if len(result_records) == 0:
            print(f"No records found for {setting}.")
            continue

        em_sum = 0
        em_trim_sum = 0
        bleu_sum = 0
        bleu_trim_sum = 0
        count = 0
        for record in result_records:
            if setting == "commented":
                performance = record["commented_performance"]
            else:
                performance = record["comments_removed_performance"]
            em_sum += performance["em"]
            em_trim_sum += performance["em_trim"]
            bleu_sum += performance["bleu"]
            bleu_trim_sum += performance["bleu_trim"]
            count += 1

        # print(
        #     f"{setting} - em: {em_sum / count:.4f}, em_trim: {em_trim_sum / count:.4f}, bleu: {bleu_sum / count:.4f}, bleu_trim: {bleu_trim_sum / count:.4f}"
        # )
        performance_report[setting] = {
            "em": em_sum / count,
            "em_trim": em_trim_sum / count,
            "bleu": bleu_sum / count,
            "bleu_trim": bleu_trim_sum / count,
        }

    print(tabulate(performance_report, headers="keys", tablefmt="grid"))
    with open(results_path / "report.json", "w") as f:
        json.dump(performance_report, f, indent=4)

    with jsonlines.open(
        results_path / "predictions.jsonl", "w"
    ) as f:
        f.write_all(result_records)
        # save to db


if __name__ == "__main__":
    from argparse import ArgumentParser
    import json

    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=model_name,
        help="The model to use for the generation. Default is the model specified in the environment variables.",
    )
    parser.add_argument(
        "--concept",
        type=str,
    )
    parser.add_argument(
        "--perturbation",
        type=str,
    )

    args = parser.parse_args()

    if not args.concept:
        results_path = cur_dir / "baseline_results" / args.model
    else:
        results_path = cur_dir / "rq3_results" / args.model / args.perturbation / args.concept
        pertubator = Perturbator(args.model, args.concept, args.perturbation)
        comment_splits_path = cur_dir / "comment_splits.json"
        with open(comment_splits_path, "r") as f:
            comment_splits = json.load(f)

    results_path.mkdir(parents=True, exist_ok=True)
    run_model()

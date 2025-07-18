import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from comment_removal.nirjas import get_comments_with_replacement_lines, remove_comments
from common.utils import calculate_sample_size

cur_dir = Path(__file__).parent.resolve()

repos_path = cur_dir / "repos"
assert (
    repos_path.exists()
), "Repos path does not exist. Please run the script to clone the repositories first."

dataset_path = cur_dir / "dataset"
dataset_path.mkdir(parents=True, exist_ok=True)


def make_dataset():
    total_files_created = 0
    total_commented_files = 0
    repos = [f for f in repos_path.iterdir() if f.is_dir()]
    extracted_comments = []
    code_records = []
    comment_type_records = []

    for repo in tqdm(repos, desc="Processing Repositories", unit="repo"):
        java_files = list(repo.rglob("*.java"))
        for file in tqdm(
            java_files,
            desc=f"Processing {repo.name}",
            leave=False,
            unit="file",
        ):
            with open(file, "r", encoding="utf-8") as f:
                code = f.read()
            comments = get_comments_with_replacement_lines(code)

            if comments and "Licensed" in "\n".join(comments[0][1]):
                code = code.replace("\n".join(comments[0][1]), "")
                comments = comments[1:]  # Skip the license comment if present

            if comments:
                total_commented_files += 1
                # base_path = file.parent.relative_to(repos_ path)
                # new_path = dataset_path / base_path / file.stem
                # new_path.mkdir(parents=True, exist_ok=True)
                uncommented_code = remove_comments(code, True)
                original_code = code

                uncommented_lines = uncommented_code.split("\n")
                assert len(uncommented_lines) == len(
                    code.split("\n")
                ), "Uncommented code lines do not match original code lines."

                variations = {
                    "without_inline": uncommented_lines.copy(),
                    "without_multiline": uncommented_lines.copy(),
                    "without_javadoc": uncommented_lines.copy(),
                }

                # with open(
                #     new_path / "uncommented.java", "w", encoding="utf-8"
                # ) as out_file:
                #     out_file.write(uncommented_code.strip())
                # with open(
                #     new_path / "original.java", "w", encoding="utf-8"
                # ) as original_file:
                #     original_file.write(code.strip())
                comments = get_comments_with_replacement_lines(code)

                has_inline_comments = False
                has_multiline_comments = False
                has_javadoc_comments = False

                for line_range, original_lines in comments:
                    line_start, line_end = line_range
                    try:
                        if 0 < len(original_lines) < 2:
                            comment_type = "inline"
                            comment = original_lines[0]
                            pos = comment.rfind("//")
                            if pos != -1:
                                comment = comment[pos:].strip()
                            else:
                                pos = comment.rfind("/*")
                                if pos != -1:
                                    comment = comment[pos + 2 :].strip()
                            has_inline_comments = True
                            extracted_comments.append(
                                {
                                    "file_path": str(file.relative_to(repos_path)),
                                    "line_start": line_start,
                                    "line_end": line_end,
                                    "comment": comment,
                                    "type": comment_type,
                                }
                            )
                        else:
                            comment_type = (
                                "javadoc"
                                if original_lines[0].lstrip().startswith("/*")
                                else "multiline"
                            )
                            extracted_comments.append(
                                {
                                    "file_path": str(file.relative_to(repos_path)),
                                    "line_start": line_start,
                                    "line_end": line_end,
                                    "comment": "\n".join(original_lines),
                                    "type": comment_type,
                                }
                            )
                            if comment_type == "multiline":
                                has_multiline_comments = True
                            else:
                                has_javadoc_comments = True
                        # Insert the comment into the uncommented code
                        # Insert the comment into the code
                        # resulting_lines = uncommented_lines.copy()

                        for (
                            c
                        ) in (
                            variations
                        ):  # putting other comment types back where they were
                            if c != f"without_{comment_type}":
                                variations[c][
                                    line_start : line_end + 1
                                ] = original_lines
                        # resulting_lines[line_start : line_end + 1] = original_lines
                        # code = "\n".join(resulting_lines)
                        # with open(
                        #     new_path / f"comment-{i+1}.java",
                        #     "w",
                        #     encoding="utf-8",
                        # ) as comment_file:
                        #     comment_file.write(code.strip())
                    except IndexError:
                        print(file)
                        raise

                if (
                    does_it_fit_the_model(code)
                    and original_code != uncommented_code.strip()
                ):
                    code_records.append(
                        {
                            "file_path": str(file.relative_to(repos_path)),
                            "uncommented_code": uncommented_code.strip(),
                            "original_code": original_code,
                            "has_inline_comments": has_inline_comments,
                            "has_multiline_comments": has_multiline_comments,
                            "has_javadoc_comments": has_javadoc_comments,
                        }
                    )
                    total_commented_files += 1

                for c in variations:
                    if does_it_fit_the_model(
                        "\n".join(variations[c])
                    ) and does_it_fit_the_model(original_code):
                        comment_type = c.split("_")[-1]

                        if (
                            (comment_type == "inline" and has_inline_comments)
                            or (comment_type == "multiline" and has_multiline_comments)
                            or (comment_type == "javadoc" and has_javadoc_comments)
                        ):
                            # print(f"Adding {comment_type} comment type")
                            comment_type_records.append(
                                {
                                    "file_path": str(file.relative_to(repos_path)),
                                    "original_code": original_code,
                                    c: "\n".join(variations[c]),
                                }
                            )
                        # with open(dataset_path / f"{c}.java", "w") as f:
                        #     f.write("\n".join(variations[c]))
                        # with open(
                        #     dataset_path / f"original.java", "w", encoding="utf-8"
                        # ) as f:
                        #     f.write(original_code)
                        total_files_created += 1
                # if total_files_created > 1:
                #     exit()

            else:
                # base_path = file.parent.relative_to(repos_path)
                # new_path = dataset_path / base_path / file.stem
                # new_path.mkdir(parents=True, exist_ok=True)
                uncommented_code = remove_comments(code, True)

                # with open(
                #     new_path / "uncommented.java", "w", encoding="utf-8"
                # ) as out_file:``
                #     out_file.write(uncommented_code.strip())

                # with open(
                #     new_path / "original.java", "w", encoding="utf-8"
                # ) as original_file:
                #     original_file.write(code.strip())
                if does_it_fit_the_model(uncommented_code):
                    code_records.append(
                        {
                            "file_path": str(file.relative_to(repos_path)),
                            "uncommented_code": uncommented_code.strip(),
                            "original_code": uncommented_code.strip(),
                            "has_inline_comments": False,
                            "has_multiline_comments": False,
                            "has_javadoc_comments": False,
                        }
                    )

        # break
    # print(f"Total files created: {total_files_created}")
    # print(f"Total commented files: {total_commented_files}")
    return extracted_comments, code_records, comment_type_records


def get_sample_dataset(comment_type, sample_size=0, seed=None):
    if comment_type == "comment":
        if sample_size == 0:
            sample_size = code_records_sample_size

        commented = [
            c
            for c in code_records
            if (
                c["has_inline_comments"]
                or c["has_multiline_comments"]
                or c["has_javadoc_comments"]
            )
        ]
        if seed != None:
            random.seed(seed)
        commented_sample = random.sample(
            commented, sample_size if len(commented) > sample_size else len(commented)
        )

        total = len(commented_sample)
        for i in range(total):
            commented_sample[i]["TrainTestFlag"] = "Train" if i < total // 2 else "Test"
            commented_sample[i]["Label"] = True

        not_commented_sample = []
        for c in commented_sample:
            new_record = c.copy()
            new_record["original_code"] = c["uncommented_code"]
            # Only add if it's actually different
            if new_record["original_code"] != c["original_code"]:
                new_record["Label"] = False
                not_commented_sample.append(new_record)

        assert not any(
            c1["original_code"] == c2["original_code"]
            for c1 in commented_sample
            for c2 in not_commented_sample
        ), "There are duplicate original_code entries between commented and not_commented samples."

        # not_commented_sample = random.sample(
        #     not_commented, 60 if len(not_commented) > 60 else len(not_commented)
        # )

        not_commented_df = pd.DataFrame(not_commented_sample + commented_sample)
        not_commented_df = not_commented_df[
            [
                "original_code",
                "Label",
                "TrainTestFlag",
            ]
        ].copy()
        not_commented_df.rename(
            columns={
                "original_code": "Instruction",
            },
            inplace=True,
        )
        not_commented_df["DatasetName"] = "comment"
        return not_commented_df

    if sample_size == 0:
        sample_size = comment_type_sample_size

    records_with_the_comment_type = comment_type_df[
        (comment_type_df[f"without_{comment_type}"].notna())
        # & (comment_type_df[f"has_{comment_type}_comments"] == True)
        & (
            comment_type_df["original_code"]
            != comment_type_df[f"without_{comment_type}"]
        )
    ].to_dict(orient="records")
    # not_inline_commented = [c for c in code_records if c[f"has_{type}"] == False]

    # sample_df = pd.DataFrame(inline_commented_sample + not_inline_commented_sample)
    # sample_df.to_csv(dataset_path / f"{type}_all_dataset.csv", index=False)
    if seed != None:
        random.seed(seed)

    records_with_the_comment_type = random.sample(
        records_with_the_comment_type,
        (
            sample_size
            if len(records_with_the_comment_type) > sample_size
            else len(records_with_the_comment_type)
        ),
    )

    total = len(records_with_the_comment_type)
    for i in range(total):
        records_with_the_comment_type[i]["TrainTestFlag"] = (
            "Train" if i < total // 2 else "Test"
        )
        records_with_the_comment_type[i]["Label"] = True

    records_wo_the_comment_type = []
    for i, c in enumerate(records_with_the_comment_type):
        new_record = c.copy()
        new_record["original_code"] = c[f"without_{comment_type}"]
        # Only add if it's actually different
        # if new_record["original_code"] != c["original_code"]:
        new_record["Label"] = False
        records_wo_the_comment_type.append(new_record)

    # not_inline_commented_sample = random.sample(
    #     not_inline_commented,
    #     60 if len(not_inline_commented) > 60 else len(not_inline_commented),
    # )
    positive_df = pd.DataFrame(records_with_the_comment_type)
    positive_df = positive_df[["original_code", "Label", "TrainTestFlag"]].copy()
    positive_df.rename(
        columns={
            "original_code": "Instruction",
        },
        inplace=True,
    )
    # positive_df.to_csv(dataset_path / f"{comment_type}_positive.csv", index=False)

    negative_df = pd.DataFrame(records_wo_the_comment_type)
    negative_df = negative_df[["original_code", "Label", "TrainTestFlag"]].copy()
    negative_df.rename(
        columns={
            "original_code": "Instruction",
        },
        inplace=True,
    )
    # negative_df.to_csv(dataset_path / f"{comment_type}_negative.csv", index=False)

    sample_df = pd.concat([positive_df, negative_df], ignore_index=True)
    sample_df["DatasetName"] = comment_type

    # sample_df.to_csv(dataset_path / f"{comment_type}_dataset.csv", index=False)
    return sample_df


def does_it_fit_the_model(text):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return (
        len(tokens) <= 12000
    )  # To ensure it fits in a GPU with 44GB VRAM after adding task instructions


if __name__ == "__main__":
    from argparse import ArgumentParser

    from transformers import AutoTokenizer

    parser = ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if not Path.exists(dataset_path / "comments.csv"):
        comment_records, code_records, comment_type_records = make_dataset()

        df = pd.DataFrame(comment_records)
        df.to_csv(dataset_path / "comments.csv", index=False)

        code_df = pd.DataFrame(code_records)
        code_df.to_csv(dataset_path / "code_records.csv", index=False)

        comment_type_df = pd.DataFrame(comment_type_records)
        comment_type_df.to_csv(
            dataset_path / "different_comment_types.csv", index=False
        )
    else:
        code_records = pd.read_csv(dataset_path / "code_records.csv").to_dict(
            orient="records"
        )
        comment_type_df = pd.read_csv(dataset_path / "different_comment_types.csv")

    inline_comments = get_sample_dataset("inline")
    multiline_comments = get_sample_dataset("multiline")
    javadoc_comments = get_sample_dataset("javadoc")
    not_commented_df = get_sample_dataset("comment")

    df = pd.concat(
        [inline_comments, multiline_comments, javadoc_comments, not_commented_df],
        ignore_index=True,
    )
    df.to_csv(dataset_path / "rq2_dataset.csv", index=False)
else:
    from tabulate import tabulate
    try:
        comment_type_df = pd.read_csv(dataset_path / "different_comment_types.csv")
        comment_type_sample_size = calculate_sample_size(len(comment_type_df)) * 2

        print(
            f"Comment type sample size: {comment_type_sample_size} (calculated from {len(comment_type_df)} records)"
        )
        sample_sizes = {}
        for comment_type in ["inline", "multiline", "javadoc"]:
            records_of_type = records_with_the_comment_type = comment_type_df[
                (comment_type_df[f"without_{comment_type}"].notna())
                # & (comment_type_df[f"has_{comment_type}_comments"] == True)
                & (
                    comment_type_df["original_code"]
                    != comment_type_df[f"without_{comment_type}"]
                )
            ]
            sample_sizes[comment_type] = calculate_sample_size(len(records_of_type)) * 2
            print(
                f"Sample size for {comment_type} comments: {sample_sizes[comment_type]} (calculated from {len(records_of_type)} records)"
            )

        code_records = pd.read_csv(dataset_path / "code_records.csv").to_dict(
            orient="records"
        )
        commented_records = [
            c
            for c in code_records
            if (
                c["has_inline_comments"]
                or c["has_multiline_comments"]
                or c["has_javadoc_comments"]
            )
        ]
        sample_sizes['comment'] = calculate_sample_size(
            len(commented_records)
        ) * 2
        print(
            tabulate(
                sample_sizes.items(),
                headers=["Comment Type", "Sample Size"],
                tablefmt="grid",
            )
        )
    except FileNotFoundError:
        print(
            "Please run the script to create the dataset first. The dataset is not available."
        )
        exit(1)

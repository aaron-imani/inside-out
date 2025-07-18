from pathlib import Path

from jsonlines import jsonlines

from comment_removal.nirjas import get_comments, remove_comments
from common.utils import get_sample

cur_dir = Path(__file__).resolve().parent
java_folder = cur_dir / "java_files"
java_folder.mkdir(parents=True, exist_ok=True)

total = 0
commented = 0
commented_records = []
with jsonlines.open("codereview_new.jsonl", mode="r") as reader:
    for obj in reader:
        if obj["language"] != "java":
            continue
        total += 1

        # Remove comments from the code
        old_without_minus = []
        for line in obj["old"].split("\n"):
            old_without_minus.append(line[1:])
        old_without_minus = "\n".join(old_without_minus)

        try:
            cleaned = remove_comments(old_without_minus)
            if cleaned == old_without_minus or len(cleaned.strip()) == 0:
                continue
        except Exception as e:
            continue

        new = obj["new"]
        new_code = []
        for line in new.split("\n"):
            if line.strip() != "":
                new_code.append(line[1:].strip())

        new_code = "\n".join(new_code)

        try:
            new_comments = get_comments(new_code)
            old_comments = get_comments(old_without_minus)
            # If comments are modified or the old comments don't have any comment block with at least 2 words, skip
            if old_comments != new_comments or not any(
                len(comment.strip().split()) > 1 for comment in old_comments
            ):
                continue
        except Exception as e:
            continue

        commented += 1
        cleaned_obj = obj.copy()
        cleaned_obj["commented"] = old_without_minus
        cleaned_obj["comments_removed"] = cleaned
        cleaned_obj["new"] = new_code

        # Write the modified object to the new file

        commented_records.append(cleaned_obj)

print(f"Total: {total}")
print(f"Commented: {commented}")
commented_records = get_sample(commented_records)
print(f"Sampled: {len(commented_records)}")

with jsonlines.open("codereview_new_filtered.jsonl", mode="w") as writer:
    for obj in commented_records:
        with open(java_folder / f"{obj['_id']}.java", "w") as f:
            f.write(obj["commented"])

        obj["old"] = old_without_minus
        writer.write(obj)

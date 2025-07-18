from pathlib import Path
import pandas as pd
import json
from jsonlines import jsonlines

cur_dir = Path(__file__).parent.resolve()
refinement_path = cur_dir.parent / "tasks/refinement"
translation_path = cur_dir.parent / "tasks/translation"
cceval_path = cur_dir.parent / "tasks/completion/data/java/sample"


def get_refinement_data() -> dict:
    with open(refinement_path / "comment_splits.json", 'r') as f:
        comment_splits = json.load(f)
    
    comment_type_splits = {
        'comment': [],
        'inline': [],
        'multiline': []
    }

    with jsonlines.open(refinement_path / 'codereview_new_filtered.jsonl', 'r') as reader:
        data = [obj for obj in reader]

    df = pd.DataFrame(data)
    df['_id'] = df['_id'].astype(str)  # Ensure _id is string

    for key in comment_splits:
        related_records = df[df['_id'].isin(comment_splits[key])]
        comment_type_splits[key] = related_records['commented'].tolist()

    return comment_type_splits


def get_translation_data() -> dict:
    dataset_path = translation_path / "filtered_dataset/codenet/Java/Code"
    with open(translation_path / "codenet_comment_splits.json", 'r') as f:
        comment_splits = json.load(f)
    
    comment_type_splits = {
        'comment': [],
        'javadoc': [],
        'inline': [],
        'multiline': []
    }
    for key in comment_splits:
        ids = [name.split("_")[0] for name in comment_splits[key]]
        for id in ids:
            with open(dataset_path/ (id+'_commented.java'), 'r') as f:
                commented = f.read()
            
            comment_type_splits[key].append(commented)

    return comment_type_splits

def get_cceval_data() -> dict:
    comment_type_splits = {
        'comment': [],
        'javadoc': [],
        'inline': [],
        'multiline': []
    }

    for key in comment_type_splits:
        with jsonlines.open(cceval_path / f'{key}_commented.jsonl', 'r') as reader:
            for obj in reader:
                code = obj['prompt'] + '\n\n' + obj['crossfile_context']['text']
                comment_type_splits[key].append(code)

    return comment_type_splits

def get_all_data() -> dict:
    comment_type_splits = {
        'comment': [],
        'javadoc': [],
        'inline': [],
        'multiline': []
    }
    cceval_data = get_cceval_data()
    translation_data = get_translation_data()
    refinement_data = get_refinement_data()

    for key in comment_type_splits:
        comment_type_splits[key].extend(cceval_data[key])
        comment_type_splits[key].extend(translation_data[key])
        if key != 'javadoc':  # Javadoc is not in refinement data
            comment_type_splits[key].extend(refinement_data[key])

    return comment_type_splits

def get_benchmarks_data() -> dict:
    """
    Returns a dictionary containing the benchmark data for code review, translation, and CCEval.
    """
    return {
        "refinement": get_refinement_data(),
        "translation": get_translation_data(),
        "completion": get_cceval_data()
    }
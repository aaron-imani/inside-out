from collections import defaultdict
from pathlib import Path
from comment_removal.nirjas import get_comment_types
import json

cur_dir = Path(__file__).parent.resolve()
java_files_dir = cur_dir / 'java_files'
java_files = list(java_files_dir.glob("*.java"))

comment_types = ['comment', 'javadoc', 'inline', 'multiline']

splits = defaultdict(list)

for file in java_files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Get the comment types for the current file
    comment_type_counts = get_comment_types(content)
    splits['comment'].append(file.stem)  # Always include the file in 'comment' split
    for comment_type, has_comments in comment_type_counts.items():
        if has_comments:
            splits[comment_type].append(file.stem)


for comment_type, files in splits.items():
    print(f"{comment_type}: {len(files)} files")

with open(cur_dir / 'comment_splits.json', 'w', encoding='utf-8') as f:
    json.dump(splits, f, indent=4)
    print(f"Saved comment splits to {cur_dir / 'comment_splits.json'}")
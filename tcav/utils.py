from pathlib import Path

cur_dir = Path(__file__).parent.resolve()
classifiers_dir = cur_dir / "classifiers"

def get_classifier_path(model_name: str, concept: str):
    if concept == 'javadoc':
        return classifiers_dir / model_name / 'javadoc' / '760_0.5' / '1.pth'
    elif concept == 'comment':
        return classifiers_dir / model_name / 'comment' / '762_0.5' / '1.pth'
    elif concept == 'inline':
        return classifiers_dir / model_name / 'inline' / '756_0.5' / '1.pth'
    else:
        return classifiers_dir / model_name / 'multiline' / '738_0.5' / '1.pth'

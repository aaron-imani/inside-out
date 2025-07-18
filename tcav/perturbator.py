from pathlib import Path

from .classifier_manager import *
from .model_generation import ModelGeneration
from .perturbation import Perturbation
from typing import Literal, Union

cur_dir = Path(__file__).parent.resolve()


class Perturbator:
    def __init__(self, model_name: str, concept:str, mode: Union[Literal['uncomment', 'comment']], target_probability=0.01, accuracy_threshold=0.82, max_length=16000):
        if 'Qwen2.5-Coder' in model_name:
            model_name = 'qwen2.5-coder'
        elif 'QwQ' in model_name:
            model_name = 'qwq'
        else:
            model_name = 'qwen2.5'
        
        self.model_name = model_name
        self.max_length = max_length
        classifier_path = self._get_classifier_path(model_name, concept)
        self.classifier = load_classifier_manager(classifier_path)
        direction = 'against' if mode == 'uncomment' else 'toward'
        self.perturbation = Perturbation(
            self.classifier,
            direction,
            target_probability=target_probability,
            accuracy_threshold=accuracy_threshold,
        )
   
        self.llm = ModelGeneration(model_name)
        self.llm.set_perturbation(self.perturbation)

    
    def _get_classifier_path(self, model_name: str, concept: str):
        if concept == 'javadoc':
            return cur_dir / 'classifiers' / model_name / 'javadoc' / '760_0.5' / '1.pth'
        elif concept == 'comment':
            return cur_dir / 'classifiers' / model_name / 'comment' / '762_0.5' / '1.pth'
        elif concept == 'inline':
            return cur_dir / 'classifiers' / model_name / 'inline' / '756_0.5' / '1.pth'
        else:
            return cur_dir / 'classifiers' / model_name / 'multiline' / '738_0.5' / '1.pth'

        
    def query(self, query: str):
        response = self.llm.generate(query, max_length=self.max_length, capture_original_outputs=False, capture_perturbed_outputs=False)
        return response['completion']
    
    
from tcav.classifier_manager import load_classifier_manager
from tcav.model_generation import ModelGeneration
from tcav.model_extraction import ModelExtraction
from tcav.perturbation import Perturbation
from tcav.utils import get_classifier_path
from typing import List, Tuple

class ConceptActivationAnalyzer:
    def __init__(self, model_name: str, concept: str):
        if 'Qwen2.5-Coder' in model_name:
            model_name = 'qwen2.5-coder'
        elif 'QwQ' in model_name:
            model_name = 'qwq'
        else:
            model_name = 'qwen2.5'
            
        self._model_name = model_name
        self._extractor = ModelExtraction(model_nickname=model_name)
        self._classifier = None

    def set_classifier(self, concept: str):
        classifier_path = get_classifier_path(self._model_name, concept)
        if self._classifier is not None:
            del self._classifier

        self._classifier = load_classifier_manager(classifier_path)

        # self._model.capture_layers_concept_activation = True
        # self.set_model_perturbation(concept)
    
    def set_model_perturbation(self, concept: str):
        if self._model.perturbation is not None:
            del self._model.perturbation
            
        classifier_path = get_classifier_path(self._model_name, concept)
        self._classifier = load_classifier_manager(classifier_path)
        self.perturbation = Perturbation(
            classifier_manager=self._classifier,
            direction='against',
            target_probability=0.01,
            accuracy_threshold=0.82
        )
        self._model.set_perturbation(self.perturbation, perturb=False)


    def get_concept_activations(self, query: str) -> Tuple[str, dict]:
        """
        Get concept activations for a given query.
        
        Args:
            query (str): The input query for which to get concept activations.

        Returns:
            Tuple[str, dict]: A tuple containing the response and a dictionary of concept activations.
        
        """
        response = self._model.generate(
            prompt=query,
            max_length=100,
            capture_perturbed_outputs=False,
            capture_original_outputs=False
        )
        return response['completion'], self._model.layers_concept_activation
    
    def analyze_concept_activation(self, prompts: List[str]) -> dict:
        """
        Analyze concept activation for a given query.
        
        Args:
            query (str): The input query for which to analyze concept activation.

        Returns:
            dict: A dictionary containing the concept activations.
        
        """
        assert self._classifier is not None, "Classifier is not set. Please set the classifier before analyzing concept activation."

        embedding_manager = self._extractor.extract_embds(prompts)
        probabilities = self._classifier.get_prediction_probability_across_layers(embedding_manager)
        return probabilities
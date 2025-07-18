from classifier_manager import ClassifierManager
import torch
from typing import Union, Literal

class Perturbation:
    def __init__(self, classifier_manager: ClassifierManager, direction: Literal['toward', 'against'], target_probability: float=0.001, accuracy_threshold: float=0.9, perturbed_layers: list[int]=None):
        self.classifier_manager = classifier_manager
        self.direction = direction
        self.target_probability = target_probability if direction == 'against' else 1 - target_probability
        self.accuracy_threshold = accuracy_threshold
        self.perturbed_layers = perturbed_layers

    def _should_prob_trigger(self, output_hook: torch.Tensor, layer: int) -> bool:
        if self.direction == 'against':
            return self.classifier_manager.classifiers[layer].predict_proba(output_hook[0][:, -1, :]) > self.target_probability
        else:
            return self.classifier_manager.classifiers[layer].predict_proba(output_hook[0][:, -1, :]) < self.target_probability
        
    def get_perturbation(self, output_hook: torch.Tensor, layer: int) -> torch.Tensor:
        if self.perturbed_layers is None or layer in self.perturbed_layers:
            if self.classifier_manager.testacc[layer] > self.accuracy_threshold and self._should_prob_trigger(output_hook, layer):
                perturbed_embds = self.classifier_manager.cal_perturbation(
                    embds_tensor=output_hook[0][:, -1, :],
                    layer=layer,
                    target_prob=self.target_probability,
                )
                output_hook[0][:, -1, :] = perturbed_embds
        return output_hook
    
    def get_concept_activation(self, output_hook: torch.Tensor, layer: int) -> float:
        if self.perturbed_layers is None or layer in self.perturbed_layers:
            # return torch.tensor([float(self.classifier_manager.classifiers[layer].predict(output_hook[0][:, -1, :]) > 0.5)]) if self.classifier_manager.testacc[layer] > self.accuracy_threshold else torch.tensor([0.0]) # Using binary classification
            return self.classifier_manager.classifiers[layer].predict_proba(output_hook[0][:, -1, :]) if self.classifier_manager.testacc[layer] > self.accuracy_threshold else torch.tensor([0.0]) # Using prediction probability

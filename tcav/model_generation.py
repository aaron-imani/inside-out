from functools import partial

import torch
from model_base import ModelBase
from perturbation import Perturbation


class ModelGeneration(ModelBase):
    def __init__(self, model_nickname: str):
        super().__init__(model_nickname)

        self.hooks = []
        self._register_hooks()
        self.perturbation: Perturbation = None

        self.original_outputs = []
        self.capture_original_outputs = False

        self.perturbed_outputs = []
        self.capture_perturbed_outputs = False
        self.perturb = False

        self.layers_concept_activation = [None] * self.llm_cfg.n_layer
        self.capture_layers_concept_activation = False

    def set_perturbation(self, perturbation, perturb=True):
        self.perturbation = perturbation
        self.perturb = perturb
    
    def dont_perturb(self):
        self.perturb = False

    def activate_perturbation(self):
        if self.perturbation is None:
            raise ValueError("No perturbation set. Please set a perturbation before activating it.")
        self.perturb = True

    def unset_perturbation(self):
        self.perturbation = None

    def _register_hooks(self):
        def _hook_fn(module, input, output, layer_idx):
            if self.capture_original_outputs:
                self.original_outputs.append(output[0].clone().detach())

            if self.perturbation is not None:
                if self.perturb:
                    output = self.perturbation.get_perturbation(output, layer_idx)
                if self.capture_layers_concept_activation:
                    self.layers_concept_activation[layer_idx] = float(self.perturbation.get_concept_activation(output, layer_idx)[0])

            if self.capture_perturbed_outputs:
                self.perturbed_outputs.append(output[0].clone().detach())

            return output

        for i in range(self.llm_cfg.n_layer):
            layer = self.model.model.layers[i]
            hook = layer.register_forward_hook(partial(_hook_fn, layer_idx=i))
            self.hooks.append(hook)

    def generate(
        self,
        prompt: str,
        max_length: int = 1000,
        capture_perturbed_outputs: bool = True,
        capture_original_outputs: bool = True,
    ) -> dict:

        self.capture_original_outputs = capture_original_outputs
        self.original_outputs = []

        self.capture_perturbed_outputs = capture_perturbed_outputs
        self.perturbed_outputs = []

        prompt = self.apply_inst_template(prompt)
        input_ids = self.tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        terminators = [
            self.tokenizer.eos_token_id,
            # self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        input_token_number = input_ids.size(1)
        max_length += input_token_number

        if "qwq" not in self.llm_cfg.model_nickname:
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                return_dict_in_generate=True,
                eos_token_id=terminators,
                do_sample=False,
            )
        else:
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                return_dict_in_generate=True,
                eos_token_id=terminators,
                do_sample=True,
                top_p=0.95,
                temperature=0.6,
            )

        result = {
            "completion_token_number": output.sequences[0].size(0) - input_token_number,
            "completion": self.tokenizer.decode(
                output.sequences[0][input_token_number:], skip_special_tokens=True
            ),
        }

        def __convert(hs):
            ret = []
            for i in range(len(hs)):
                embds = torch.zeros(self.llm_cfg.n_layer, self.llm_cfg.n_dimension).to(
                    self.device
                )
                for j in range(len(hs[i])):
                    embds[j, :] = hs[i][j][0, -1, :]
                ret.append(embds)
            return ret

        if self.capture_perturbed_outputs:
            n = len(self.perturbed_outputs) // self.llm_cfg.n_layer
            result["perturbed_outputs"] = __convert(
                [
                    self.perturbed_outputs[
                        i * self.llm_cfg.n_layer : (i + 1) * self.llm_cfg.n_layer
                    ]
                    for i in range(n)
                ]
            )

        if self.capture_original_outputs:
            n = len(self.original_outputs) // self.llm_cfg.n_layer
            result["original_outputs"] = __convert(
                [
                    self.original_outputs[
                        i * self.llm_cfg.n_layer : (i + 1) * self.llm_cfg.n_layer
                    ]
                    for i in range(n)
                ]
            )

        return result

    def __del__(self):
        for hook in self.hooks:
            hook.remove()

import torch
from llm_config import cfg, get_cfg
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging

logging.set_verbosity_error()

class ModelBase:
    def __init__(self, model_nickname: str):
        self.llm_cfg = get_cfg(model_nickname)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"[DEBUG] Using device: {self.device}")

        self.model = AutoModelForCausalLM.from_pretrained(self.llm_cfg.model_name).to(
            self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_cfg.model_name)

    def apply_sft_template(self, instruction, system_message=None):
        if system_message is not None:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": instruction},
            ]
        else:
            messages = [{"role": "user", "content": instruction}]

        return messages

    def apply_inst_template(self, text):
        messages = [{"role": "user", "content": text}]
        return messages

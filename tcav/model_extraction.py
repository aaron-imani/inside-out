import torch
from embedding_manager import EmbeddingManager
from model_base import ModelBase
from tqdm import tqdm


class ModelExtraction(ModelBase):
    def __init__(self, model_nickname: str):
        super().__init__(model_nickname)

    def extract_embds(
        self, inputs: list[str], system_message: str = None, message: str = None
    ) -> EmbeddingManager:
        embds_manager = EmbeddingManager(self.llm_cfg, message)
        embds_manager.layers = [
            torch.zeros(len(inputs), self.llm_cfg.n_dimension)
            for _ in range(self.llm_cfg.n_layer)
        ]

        for i, txt in tqdm(
            enumerate(inputs),
            desc="Extracting embeddings",
            leave=False,
            total=len(inputs),
        ):
            try:
                messages = self.apply_sft_template(
                    instruction=txt, system_message=system_message
                )
                torch.cuda.empty_cache()
                input_ids = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(input_ids, output_hidden_states=True)

                    hidden_states = outputs.hidden_states

                    for j in range(self.llm_cfg.n_layer):
                        embds_manager.layers[j][i, :] = (
                            hidden_states[j][:, -1, :].detach().cpu()
                        )

                    del outputs, hidden_states, input_ids
            except torch.cuda.OutOfMemoryError:
                print("Record tokens length:", len(self.tokenizer.encode(txt)))
                raise ValueError(
                    "Out of memory error. Reduce the batch size or sequence length."
                )

        return embds_manager

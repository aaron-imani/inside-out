__cfg = {
    "qwen2.5-coder": {
        "model_nickname": "qwen2.5-coder",
        "model_name": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
        "n_layer": 64,
        "n_dimension": 5120,
    },
    "qwen2.5": {
        "model_nickname": "qwen2.5",
        "model_name": "Qwen/Qwen2.5-32B-Instruct-AWQ",
        "n_layer": 64,
        "n_dimension": 5120,
    },
    "qwq": {
        "model_nickname": "qwq",
        "model_name": "Qwen/QwQ-32B-AWQ",
        "n_layer": 64,
        "n_dimension": 5120,
    },
}


class cfg:
    def __init__(self, cfg_dict: dict):
        self.__dict__.update(cfg_dict)


def get_cfg(model_nickname: str):
    assert model_nickname in __cfg, f"{model_nickname} not found in config"
    return cfg(__cfg[model_nickname])

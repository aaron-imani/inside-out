from transformers import AutoTokenizer

from . import env_loader

tokenizer = AutoTokenizer.from_pretrained(env_loader.MODEL_NAME)


def encode(text):
    return tokenizer.encode(text)


def decode(tokens):
    return tokenizer.decode(tokens)

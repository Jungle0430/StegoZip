import os
import json
import sys
import torch
from pathlib import Path
from typing import Tuple
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import Settings

def load_model(settings: Settings) -> PreTrainedModel:
    if settings.task == 'text':
        if settings.model_name in ['gpt2', 'distilgpt2']:  #TODO: add zip module
            model = GPT2LMHeadModel.from_pretrained(settings.model_name).to(settings.device)
            tokenizer = GPT2Tokenizer.from_pretrained(settings.model_name)
        elif settings.model_name == 'LLaMA-7B':
            print("Loading LLaMA-7B model")
            model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", torch_dtype=torch.bfloat16).to(settings.device)
            tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    model.eval()
    return model, tokenizer


def get_tokenizer(settings: Settings) -> PreTrainedTokenizer:
    assert settings.task == 'text'
    if settings.model_name in ['gpt2', 'distilgpt2']:
        tokenizer = GPT2Tokenizer.from_pretrained(settings.model_name)  # local_files_only=True
    elif settings.model_name == 'transfo-xl-wt103':
        tokenizer = TransfoXLTokenizer.from_pretrained(settings.model_name)  # local_files_only=True
    elif settings.model_name == 'llama-7b':
        # tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        tokenizer = AutoTokenizer.from_pretrained("yahma/llama-7b-hf")
    else:
        raise NotImplementedError
    return tokenizer

def get_model(settings: Settings) -> PreTrainedModel:
    if settings.task == 'text':
        if settings.model_name in ['gpt2', 'distilgpt2']:
            model = GPT2LMHeadModel.from_pretrained(settings.model_name).to(settings.device)
        elif settings.model_name == 'transfo-xl-wt103':
            model = TransfoXLLMHeadModel.from_pretrained(settings.model_name).to(settings.device)
        elif settings.model_name == 'llama-7b':
            model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", torch_dtype=torch.float32).to(settings.device)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    model.eval()
    return model
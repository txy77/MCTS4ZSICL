import time
import os
import dataclasses
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import pdb
import math
import openai
import torch
import json
import re
import random
import tiktoken
from accelerate import Accelerator

import logging
logging.getLogger("ray").setLevel(logging.ERROR)

@dataclasses.dataclass
class Sample:
    idx: int
    raw_question: str
    question: str
    label: str
    embed: torch.TensorType
    subset: str or None
    confidence: float or None
    entropy: torch.TensorType or None
    pseudo_label: str or None
    demonstration: str or None
    sorted_softmax_token2logprob: list or None
    usable: bool or None

def transfor_model2model_path(model_name):
    if model_name == "llama3.1-8b":
        model_path = f"./meta-llama/Meta-Llama-3.1-8B/"
    elif model_name == "qwen2.5-7b":
        model_path = f"./Qwen/Qwen2.5-7B/"
    elif model_name == "mistral-7b":
        model_path = f"./mistralai/Mistral-7B-v0.3/"
    return model_path

def load_llm(model_path, device, accelerator):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True).eval()
    model = accelerator.prepare(model)
    return tokenizer, model
    

def call_llm_server_func(
    prompt, tokenizer, model, device, accelerator, max_decode_steps=128, temperature=1.0, do_sample=False
):
    """The function to call llm with a list of input strings or token IDs."""
    res_completions = []
    
    if isinstance(prompt, str):
        prompt = [prompt]
    
    for p in prompt:
        input_ids = tokenizer([p], return_tensors="pt")["input_ids"].to(device)
        input_ids = accelerator.prepare(input_ids)
        response = model.generate(input_ids, max_new_tokens=max_decode_steps, do_sample=do_sample, temperature=temperature, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(response[0][input_ids.shape[1]:], skip_special_tokens=True)
        res_completions.append(response)
        
    return res_completions

def call_llm_logits_server_func(prompt, tokenizer, model, device, accelerator, labels, max_decode_steps=5, temperature=0):
    
    label_space_idx = torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize(i)) for i in labels])
    
    decoded_token2logprobs = {}
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    input_ids = accelerator.prepare(input_ids)
    with torch.no_grad():
        outputs = model(input_ids=input_ids).logits
        
    output_logits = outputs[0][-1].cpu()[label_space_idx].squeeze().to(torch.float32)
    entropy = - torch.sum(output_logits.softmax(dim=-1) * output_logits.log_softmax(dim=-1)).to(device)
    for label, logit in zip(labels, output_logits.cpu()):
        decoded_token2logprobs[label] = logit.item()
    return decoded_token2logprobs, entropy



def call_openai_server_func(prompt, model, client, labels=None, temperature=0, max_tokens=5):    
    
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    
    try:
        response = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        res_completion = response.choices[0].message.content
        if labels is not None:
            res_completion = parse_answer(res_completion, labels)
            return res_completion
        else:
            return [res_completion]
    except OSError as e:
        retry_time = 5
        print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_func(
            prompt, model=model, labels=labels, temperature=temperature
        )


def parse_answer(res_completion, labels):
    res_completion = res_completion.split('\n\n')[0]
    matches = []
    for label in labels:
        match = re.search(re.escape(label), res_completion)
        if match:
            matches.append((match.start(), label))
    
    if not matches:
        return random.choice(labels)
    
    matches.sort(key=lambda x: x[0])
    return matches[0][1]
    

def read_jsonl(path):
    data = []
    with open(path, "r") as file:
        lines = file.readlines()
        for line in lines:
            json_object = json.loads(line)
            data.append(json_object)
    return data


def write_jsonl(data, file_name):
    with open(file_name, 'w') as f:
        for d in data:
            json_str = json.dumps(d)
            f.write(json_str + "\n")
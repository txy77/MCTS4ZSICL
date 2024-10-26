import torch
from retriever import DynamicReteiever
import json
from tqdm import tqdm
import re
import random
import os
import sys
import pdb
import math
import tiktoken
from collections import defaultdict

ZS_ICL_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)
sys.path.insert(0, ZS_ICL_ROOT_PATH)

from src.utils import Sample, call_openai_server_func, call_llm_logits_server_func, read_jsonl


class DAIL:
    """
    Inference code for DAIL. You can inference your data with two steps:
    1). Init:             inferencer = DAIL(**kwargs)
    2). inference:        inferencer.run()
    """

    def __init__(self, args, sentence_model, target_tokenizer, target_model, subsets, label_space_map, task_description, is_choices, device, accelerator, client):
        self.args = args
        self.sentence_model = sentence_model
        self.target_tokenizer = target_tokenizer
        self.target_model = target_model
        self.subsets = subsets
        self.label_space_map = label_space_map
        self.task_description_map = task_description
        self.is_choices = is_choices
        self.tokens = 0
        if self.target_model == "gpt-4o-mini":
            self.encoding = tiktoken.encoding_for_model(args.model)
        self.device = device
        self.accelerator = accelerator
        self.client = client
        self.retriever = DynamicReteiever(args, target_tokenizer=target_tokenizer, target_model=target_model, device=self.device)

    def get_embedding(self, sentence):
        embedding = self.sentence_model.encode([sentence], convert_to_tensor=True)
        return embedding
    
    def get_query_response(self, query, subset):
        if self.target_model == "gpt-4o-mini":
            label = call_openai_server_func(query, self.target_model, self.client, labels=self.label_space_map[subset], temperature=0)
            self.tokens += len(self.encoding.encode(query))
            sorted_softmax_token2logprob = None
            entropy = None
        else:
            decoded_token2logprob, entropy = call_llm_logits_server_func(query, self.target_tokenizer, self.target_model, device=self.device, accelerator=self.accelerator, labels=self.label_space_map[subset])
            exp_values = {key: math.exp(value) for key, value in decoded_token2logprob.items()}
            sum_exp_values = sum(exp_values.values())
            softmax_token2logprob = {key: exp_value / sum_exp_values for key, exp_value in exp_values.items()}
            sorted_softmax_token2logprob = sorted(softmax_token2logprob.items(), key=lambda x: x[1], reverse=True)
            label = sorted_softmax_token2logprob[0][0]
        return label, sorted_softmax_token2logprob, entropy

    def inference(self, sample_id, sample):
        subset = sample.subset
        self.inference_num += 1
        query = self.retriever.get_final_query(sample)
        if self.args.dataset == "bbh" or self.args.dataset == "bbh-mini":
            query = f"Task description: {self.task_description_map[subset]}" + "\n\n" + query
        response, sorted_softmax_token2logprob, entropy = self.get_query_response(query, subset)
        sample.entropy = entropy
        sample.usable = True
        if self.args.dataset == "bbh" or self.args.dataset == "bbh-mini":
            if subset in self.is_choices.keys():
                sample.demonstration = sample.question + response + ")"
                sample.pseudo_label = f"({response})"
            else:
                sample.demonstration = sample.question + response
                sample.pseudo_label = response
            self.retriever.add_sample(sample_id, sample)
        elif self.args.dataset == "mmlu":
            sample.demonstration = sample.question + response
            sample.pseudo_label = response
            self.retriever.add_sample(sample_id, sample)
        return query, sample, sorted_softmax_token2logprob

    def preprocess(self, idx, sample):
        subset = sample["subset"]
        if self.args.dataset == "bbh" or self.args.dataset == "bbh-mini":
            if subset in self.is_choices.keys():
                prompt = "Q: {question}\nA: ("
            else:
                prompt = "Q: {question}\nA: "
            raw_question = sample["input"]
            question = prompt.format_map({"question": sample["input"]})
            label = sample["target"]
            
        elif self.args.dataset == "mmlu":
            prompt = "Question: {question}\nA.{A}    B.{B}    C.{C}    D.{D}\nAnswer: "
            raw_question = sample["input"]
            question = prompt.format_map({"question": sample["input"], "A": sample["A"], "B": sample["B"], "C": sample["C"], "D": sample["D"]})
            label = f"{sample['target']}"
        
        embed = self.get_embedding(question).squeeze()
        sample = Sample(idx, raw_question, question, label, embed, subset, None, None, None, None, None, False)
        self.sampleid2sample[idx] = sample
        return sample

    def run(self):
        self.test_dataset_num = 0
        self.subset_right_num = defaultdict(int)
        self.subset_test_num = defaultdict(int)
        global_right_sample_num = 0
        global_test_sample_num = 0
        self.inference_num = 0
        
        if self.args.dataset == "bbh" or self.args.dataset == "bbh-mini":
            
            results = {"avg": 0, "binary_choice": 0, "multiple_choice": 0}
            
            global_binary_choice_right_sample_num = 0
            global_binary_choice_total_sample_num = 0
            
            global_multiple_choice_right_sample_num = 0
            global_multiple_choice_total_sample_num = 0
            
        elif self.args.dataset == "mmlu":
            
            results = {"avg": 0, "stem": 0, "humanities": 0, "social_science": 0, "others": 0}
            
            global_humanities_right_sample_num = 0
            global_humanities_total_sample_num = 0
            
            global_stem_right_sample_num = 0
            global_stem_total_sample_num = 0
            
            global_social_science_right_sample_num = 0
            global_social_science_total_sample_num = 0
            
            global_others_right_sample_num = 0
            global_others_total_sample_num = 0
        
        self.result_dict = defaultdict(list)
        
        
        if self.args.dataset == "bbh" or self.args.dataset == "mmlu":
            for subset in self.subsets:
                self.dataset_list = []
                self.sampleid2sample = {}
                if self.args.dataset == "bbh":
                    with open(f"data/{self.args.dataset}/{subset}.json", "r") as f:
                        dataset = json.load(f)["examples"]
                elif self.args.dataset == "mmlu":
                    dataset = read_jsonl(f"data/{self.args.dataset}/{subset}/test.jsonl")
                    
                self.subset_right_num[subset] = 0
                self.subset_test_num[subset] = len(dataset)
                for d in dataset:
                    d["subset"] = subset
                    self.dataset_list.append(d)        
            
                for idx in tqdm(range(len(self.dataset_list)), desc=f"DAIL Inference {self.args.dataset}"):
                    subset = self.dataset_list[idx]["subset"]
                    sample = self.preprocess(idx, self.dataset_list[idx])
                    query, sample, sorted_softmax_token2logprob = self.inference(idx, sample)
                    self.sampleid2sample[idx] = sample
                    
                    res_data = {
                        "idx": sample.idx,
                        "query": query,
                        "res_prob": sorted_softmax_token2logprob,
                        "pred": sample.pseudo_label,
                        "gt_ans": sample.label,
                        "entropy": sample.entropy.item() if isinstance(sample.entropy, torch.Tensor) else sample.entropy, 
                        "acc": int(sample.pseudo_label == sample.label),
                    }
                    self.result_dict[subset].append(res_data)
                
                self.test_dataset_num += len(dataset)
            
                subset_save_dir = f"zsicl-output/{self.args.dataset}/{self.args.model}/{self.args.method}_{self.args.select_strategy}/"
                os.makedirs(subset_save_dir, exist_ok=True)
                for subset, res_list in self.result_dict.items():            
                    res_path = f"{subset_save_dir}/{subset}.jsonl"
                    res_list = sorted(res_list, key=lambda x: x["idx"])
                    with open(res_path, 'w') as file:
                        for res_data in res_list:
                            file.write(json.dumps(res_data) + '\n')
            
                for sample_idx, sample in self.sampleid2sample.items():
                    subset = sample.subset
                    label = sample.label
                    pseudo_label = sample.pseudo_label
                    global_test_sample_num += 1
                    if label == pseudo_label:
                        self.subset_right_num[subset] += 1
                        global_right_sample_num += 1
                    if self.args.dataset == "bbh":
                        if subset in self.is_choices.keys():
                            global_multiple_choice_total_sample_num += 1
                            if label == pseudo_label:
                                global_multiple_choice_right_sample_num += 1
                        else:
                            global_binary_choice_total_sample_num += 1
                            if label == pseudo_label:
                                global_binary_choice_right_sample_num += 1
                    elif self.args.dataset == "mmlu":
                        if self.is_choices[subset] == "STEM":
                            global_stem_total_sample_num += 1
                            if label == pseudo_label:
                                global_stem_right_sample_num += 1
                        elif self.is_choices[subset] == "Social Science":
                            global_social_science_total_sample_num += 1
                            if label == pseudo_label:
                                global_social_science_right_sample_num += 1
                        elif self.is_choices[subset] == "Humanities":
                            global_humanities_total_sample_num += 1
                            if label == pseudo_label:
                                global_humanities_right_sample_num += 1
                        elif self.is_choices[subset] == "Others":
                            global_others_total_sample_num += 1
                            if label == pseudo_label:
                                global_others_right_sample_num += 1
            
                acc = self.subset_right_num[subset] / self.subset_test_num[subset]
                results[subset] = acc
                print(f"{subset}: {acc}")
        
        elif self.args.dataset == "bbh-mini":
            self.dataset_list = []
            for subset in self.subsets:                
                self.sampleid2sample = {}
                if self.args.dataset == "bbh-mini":
                    with open(f"data/{self.args.dataset}/{subset}.json", "r") as f:
                        dataset = json.load(f)["examples"]
                for d in dataset:
                    d["subset"] = subset
                    self.dataset_list.append(d)
            
            random.seed(0)
            random.shuffle(self.dataset_list)
            
            for idx in tqdm(range(len(self.dataset_list)), desc=f"DAIL Inference {self.args.dataset}"):
                sample = self.preprocess(idx, self.dataset_list[idx])
                query, sample, sorted_softmax_token2logprob = self.inference(idx, sample)
                self.sampleid2sample[idx] = sample
                
            for sample_idx, sample in self.sampleid2sample.items():
                subset = sample.subset
                label = sample.label
                pseudo_label = sample.pseudo_label
                global_test_sample_num += 1
                if label == pseudo_label:
                    global_right_sample_num += 1
                if self.args.dataset == "bbh-mini":
                    if subset in self.is_choices.keys():
                        global_multiple_choice_total_sample_num += 1
                        if label == pseudo_label:
                            global_multiple_choice_right_sample_num += 1
                    else:
                        global_binary_choice_total_sample_num += 1
                        if label == pseudo_label:
                            global_binary_choice_right_sample_num += 1
            
        results["avg"] = global_right_sample_num / global_test_sample_num
        if self.args.dataset == "bbh" or self.args.dataset == "bbh-mini":
            results["binary_choice"] = global_binary_choice_right_sample_num / global_binary_choice_total_sample_num
            results["multiple_choice"] = global_multiple_choice_right_sample_num / global_multiple_choice_total_sample_num
        elif self.args.dataset == "mmlu":
            results["stem"] = global_stem_right_sample_num / global_stem_total_sample_num
            results["humanities"] = global_humanities_right_sample_num / global_humanities_total_sample_num
            results["social_science"] = global_social_science_right_sample_num / global_social_science_total_sample_num
            results["others"] = global_others_right_sample_num / global_others_total_sample_num
        return results
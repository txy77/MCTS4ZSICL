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
from collections import defaultdict

ZS_ICL_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)
sys.path.insert(0, ZS_ICL_ROOT_PATH)

from src.utils import Sample, call_llm_server_func, call_llm_logits_server_func, read_jsonl


class DAIL_CALI:
    """
    Inference code for DAIL. You can inference your data with two steps:
    1). Init:             inferencer = DAIL(**kwargs)
    2). inference:        inferencer.run()
    """

    def __init__(self, args, sentence_model, target_tokenizer, target_model, subsets, label_space_map, task_description, is_choices, device, use_vllm, accelerator):
        self.args = args
        self.sentence_model = sentence_model
        self.target_tokenizer = target_tokenizer
        self.target_model = target_model
        self.subsets = subsets
        self.label_space_map = label_space_map
        self.task_description_map = task_description
        self.is_choices = is_choices
        self.device = device
        self.use_vllm = use_vllm
        self.accelerator = accelerator
        self.retriever = DynamicReteiever(args, target_tokenizer=target_tokenizer, target_model=target_model, device=self.device)

    def get_embedding(self, sentence):
        embedding = self.sentence_model.encode([sentence], convert_to_tensor=True)
        return embedding

    def get_query_response(self, query, subset):
        if self.use_vllm:
            res_completions, decoded_token2logprobs = call_vllm_logits_server_func(query, self.target_model, self.label_space_map[subset])
        else: 
            res_completion, decoded_token2logprob, entropy = call_llm_logits_server_func(query, self.target_tokenizer, self.target_model, device=self.device, accelerator=self.accelerator, labels=self.label_space_map[subset])
        exp_values = {key: math.exp(value) for key, value in decoded_token2logprob.items()}
        sum_exp_values = sum(exp_values.values())
        softmax_token2logprob = {key: exp_value / sum_exp_values for key, exp_value in exp_values.items()}
        sorted_softmax_token2logprob = sorted(softmax_token2logprob.items(), key=lambda x: x[1], reverse=True)
        label = sorted_softmax_token2logprob[0][0]
        return label, sorted_softmax_token2logprob, entropy

    def inference(self, sample, use_demonstrations):
        subset = sample.subset
        if use_demonstrations:
            query = self.retriever.get_final_query(sample)
        else:
            query = sample.question
        if self.args.dataset == "bbh":
            query = f"Task description: {self.task_description_map[subset]}" + "\n\n" + query
        response, sorted_softmax_token2logprob, entropy = self.get_query_response(query, subset)
        sample.entropy = entropy
        sample.usable = True
        sample.sorted_softmax_token2logprob = sorted_softmax_token2logprob
        if self.args.dataset == "bbh":
            if subset in self.is_choices.keys():
                sample.demonstration = sample.question + response + ")"
                sample.pseudo_label = f"({response})"
            else:
                sample.demonstration = sample.question + response
                sample.pseudo_label = response
            self.retriever.add_sample(sample)
        elif self.args.dataset == "mmlu":
            sample.demonstration = sample.question + response
            sample.pseudo_label = response
            self.retriever.add_sample(sample)
        return query, sample, sorted_softmax_token2logprob

    def preprocess(self, idx, sample):
        subset = sample["subset"]
        if self.args.dataset == "bbh":
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
        results = {"avg": 0}
        self.test_dataset_num = 0
        self.dataset_list = []
        self.subset_right_num = {}
        self.subset_test_num = {}
        global_right_sample_num = 0
        global_test_sample_num = 0
        self.result_dict = defaultdict(list)
        
        for subset in self.subsets:
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

            self.label_bias = {}
            
            for idx in tqdm(range(len(self.dataset_list)), desc=f"Inference {self.args.dataset}"):
                
                subset = self.dataset_list[idx]["subset"]
                sample = self.preprocess(idx, self.dataset_list[idx])
                query, sample, sorted_softmax_token2logprob = self.inference(sample, use_demonstrations=True)
                self.sampleid2sample[idx] = sample
                
                for label2prob in sorted_softmax_token2logprob:
                    l = label2prob[0]
                    p = label2prob[1]
                    if l not in self.label_bias:
                        self.label_bias[l] = p
                    else:
                        self.label_bias[l] += p
                
                res_data = {
                    "idx": sample.idx,
                    "query": query,
                    "res_prob": sorted_softmax_token2logprob,
                    "pred": sample.pseudo_label,
                    "gt_ans": sample.label,
                    "entropy": sample.entropy.item(),
                    "acc": int(sample.pseudo_label == sample.label),
                }
                self.result_dict[subset].append(res_data)
            
            self.test_dataset_num += len(dataset)
            self.dataset_list.clear()
            
            for label, prob in self.label_bias.items():
                self.label_bias[label] = prob / len(self.sampleid2sample)
            
            acc_list = []
            for sample_id, sample in self.sampleid2sample.items():
                sorted_softmax_token2logprob = sample.sorted_softmax_token2logprob
                label_prob_res = {}
                for p in sorted_softmax_token2logprob:
                    label = p[0]
                    prob = p[1]
                    label_prob_res[label] = prob - self.label_bias[label]
                label_prob_res = sorted(label_prob_res.items(), key=lambda x: x[1], reverse=True)
                pseudo_label = label_prob_res[0][0]
                if subset in self.is_choices.keys():
                    pseudo_label = f"({pseudo_label})"
                self.sampleid2sample[sample_id].pseudo_label = pseudo_label
                gt_answer = sample.label
                acc = int(pseudo_label == gt_answer)
                acc_list.append(acc)
                
            subset_save_dir = f"zsicl-output/{self.args.dataset}/{self.args.model}/{self.args.method}_{self.args.select_strategy}/"
            os.makedirs(subset_save_dir, exist_ok=True)
            for subset, res_list in self.result_dict.items():
                res_path = f"{subset_save_dir}/{subset}.jsonl"
                res_list = sorted(res_list, key=lambda x: x["idx"])
                with open(res_path, 'w') as file:
                    for res_data in res_list:
                        file.write(json.dumps(res_data) + '\n')

            for acc in acc_list:
                global_test_sample_num += 1
                if acc == 1:
                    self.subset_right_num[subset] += 1
                    global_right_sample_num += 1
        
            acc = self.subset_right_num[subset] / self.subset_test_num[subset]
            results[subset] = acc
            print(f"{subset}: {acc}")
        results["avg"] = global_right_sample_num / global_test_sample_num
        return results
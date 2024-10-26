import torch
import torch.nn.functional as F
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
import numpy as np
from copy import deepcopy
from collections import Counter, defaultdict

ZS_ICL_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)
sys.path.insert(0, ZS_ICL_ROOT_PATH)

from src.utils import Sample, call_llm_logits_server_func, read_jsonl, call_openai_server_func

class Demo_selection:
    def __init__(self, args, target_tokenizer, target_model, device, accelerator):
        self.args = args
        self.target_tokenizer = target_tokenizer
        self.target_model = target_model
        self.device = device
        self.accelerator = accelerator
        self.demonstrations = {}
    
    @staticmethod
    def normalize(tensor):
        mean_value = torch.mean(tensor, dim=-1)
        mean_adjusted_tensor = tensor - mean_value
        std_value = torch.std(mean_adjusted_tensor, dim=-1)
        standardized_tensor = mean_adjusted_tensor / std_value
        return standardized_tensor
    
    def calc_similarity(self, sampleid2sample):        
        embed_list = []
        for sample_id, sample in sampleid2sample.items():
            embed = sample.embed
            embed_list.append(embed)
        
        embedding_matrix = torch.stack(embed_list) # (sample_num, embedding)
        n = len(embed_list)
        self.similarity_matrix = torch.zeros((n, n), device=embedding_matrix.device) # (sample_num, sample_num)
        self.mask = torch.zeros((n, n), device=embedding_matrix.device) # (sample_num, sample_num)
        
        for i in tqdm(range(0, n, 1000), desc="calc question similarity"):
            for j in range(0, n, 1000):
                end_i = min(i + 1000, n)
                end_j = min(j + 1000, n)
                self.similarity_matrix[i:end_i, j:end_j] = F.cosine_similarity(
                    embedding_matrix[i:end_i].unsqueeze(1),
                    embedding_matrix[j:end_j].unsqueeze(0),
                    dim=-1
                )
        self.similarity_matrix.fill_diagonal_(0)
    
    def add_mask(self, id):
        self.mask[id, :] = 1
        self.mask[:, id] = 1
        
    def reset_mask(self, ids):
        self.mask.zero_()
        for id in ids:
            self.mask[id, :] = 1
            self.mask[:, id] = 1
    
    def add_sample(self, sample_id, sample):
        self.demonstrations[sample_id] = sample
        
    def remove_sample(self, sample_id):
        del self.demonstrations[sample_id]
    
    def get_diverse_score(self, sample_id):
        self.dnum = min(len(self.demonstrations), self.args.shot_num)
        usable_similarity_matrix = self.mask * self.similarity_matrix
        row_values = usable_similarity_matrix[sample_id]
        non_zero_indices = torch.nonzero(row_values, as_tuple=False).flatten()
        non_zero_values = row_values[non_zero_indices]
        sorted_values, sorted_indices = torch.sort(non_zero_values, descending=True)
        sorted_values = sorted_values.tolist()[:min(len(sorted_values), self.args.diverse_candidate)]
        sorted_ids = non_zero_indices[sorted_indices].tolist()[:min(len(sorted_values), self.args.diverse_candidate)]
        
        confidence_score = []
        similarity_score = []
        
        for sorted_value, sorted_id in zip(sorted_values, sorted_ids):
            sample = self.demonstrations[sorted_id]
            similarity_score.append(sorted_value)
            if self.args.model != "gpt-4o-mini":
                confidence_score.append(sample.confidence)
            else:
                confidence_score.append(0)
        
        if len(similarity_score) == 0 or len(confidence_score) == 0:
            return 0, 0
        
        avg_similarity_score = sum(similarity_score) / len(similarity_score)
        avg_confidence_score = sum(confidence_score) / len(confidence_score)
        return avg_similarity_score, avg_confidence_score

class Search:
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
        self.device = device
        self.tokens = 0
        if self.target_model == "gpt-4o-mini":
            self.encoding = tiktoken.encoding_for_model(args.model)
        self.accelerator = accelerator
        self.client = client
        self.retriever = DynamicReteiever(args, target_tokenizer=target_tokenizer, target_model=target_model, device=self.device)
        self.demo_sele = Demo_selection(args, target_tokenizer, target_model, device, accelerator)

    def get_embedding(self, sentence):
        embedding = self.sentence_model.encode([sentence], convert_to_tensor=True)
        return embedding
    
    def Search(self, subset):
        
        if self.args.search_strategy == "Greedy":
            self.demo_sele.calc_similarity(self.sampleid2sample)
            self.sampleid2info = defaultdict(list)
            current_id = self.start_node
            current_sample = self.sampleid2sample[current_id]
            
            self.traversed_nodes = []
            pbar = tqdm(total=len(self.sampleid2sample), desc="Greedy Search", unit="node")
            while len(self.traversed_nodes) < len(self.dataset_list):
                query, current_sample, sorted_softmax_token2logprob, entropy = self.inference(current_sample, use_demonstrations=True)
                self.inference_num += 1
                self.sampleid2sample[current_id] = current_sample
                self.sampleid2info[current_id].append((current_sample.pseudo_label, sorted_softmax_token2logprob))
                
                self.traversed_nodes.append(current_id)
                self.demo_sele.add_sample(current_id, current_sample)
                self.demo_sele.add_mask(current_id)
                self.retriever.add_sample(current_id, current_sample)
                pbar.update(1)
                
                unsolved_sample_ids = list(set(self.sampleid2sample.keys()) - set(self.traversed_nodes))
                if len(unsolved_sample_ids) == 0:
                    break
                
                max_demo_score = 0
                evaluate_sample_id = -1
                
                for unsolved_sample_id in unsolved_sample_ids:
                    demonstration_similarity_score, demonstration_confidence_score = self.demo_sele.get_diverse_score(unsolved_sample_id)
                    demo_score = demonstration_similarity_score + demonstration_confidence_score
                    if demo_score > max_demo_score:
                        max_demo_score = demo_score
                        evaluate_sample_id = unsolved_sample_id
                
                current_id = evaluate_sample_id
                current_sample = self.sampleid2sample[current_id]
            
            iter_num = 0
            self.model_debias(iter_num, subset)
            pbar.close()
            
        
        elif self.args.search_strategy == "MC":
            self.demo_sele.calc_similarity(self.sampleid2sample)
            self.sampleid2info = defaultdict(list)
            current_id = self.start_node
            current_sample = self.sampleid2sample[current_id]
            
            self.traversed_nodes = []
            pbar = tqdm(total=len(self.sampleid2sample), desc="Single Monte Carlo Search", unit="node")
            while len(self.traversed_nodes) < len(self.dataset_list):
                query, current_sample, sorted_softmax_token2logprob, entropy = self.inference(current_sample, use_demonstrations=True)
                self.inference_num += 1
                self.sampleid2sample[current_id] = current_sample
                self.sampleid2info[current_id].append((current_sample.pseudo_label, sorted_softmax_token2logprob))
                
                self.traversed_nodes.append(current_id)
                self.demo_sele.add_sample(current_id, current_sample)
                self.demo_sele.add_mask(current_id)
                self.retriever.add_sample(current_id, current_sample)
                pbar.update(1)
                
                unsolved_sample_ids = list(set(self.sampleid2sample.keys()) - set(self.traversed_nodes))
                if len(unsolved_sample_ids) == 0:
                    break
                
                current_id = random.choice(unsolved_sample_ids)
                current_sample = self.sampleid2sample[current_id]
            
            iter_num = 0
            self.model_debias(iter_num, subset)
            pbar.close()
            
        elif self.args.search_strategy == "Beam_Search":
            self.demo_sele.calc_similarity(self.sampleid2sample)
            self.sampleid2info = defaultdict(list)
            
            pbar = tqdm(total=len(self.sampleid2sample), desc="Beam Search", unit="node")
            self.start_state = [self.start_node]
            start_sample = self.sampleid2sample[self.start_node]
            start_query, start_sample, start_sorted_softmax_token2logprob, start_entropy = self.inference(start_sample, use_demonstrations=True)
            self.inference_num += 1
            self.sampleid2sample[self.start_node] = start_sample
            self.sampleid2info[self.start_node].append((start_sample.pseudo_label, start_sorted_softmax_token2logprob))
            self.demo_sele.add_sample(self.start_node, self.sampleid2sample[self.start_node])
            self.demo_sele.add_mask(self.start_node)
            self.retriever.add_sample(self.start_node, self.sampleid2sample[self.start_node])
            current_states = [self.start_state]
            self.inference_num += 1
            
            while len(current_states) > 0:
                
                state2reward = {}
                
                for current_state in current_states:                
                    unsolved_sample_ids = list(set(self.sampleid2sample.keys()) - set(current_state))
                    
                    if len(unsolved_sample_ids) == 0:
                        break
                    
                    next_state2score = {}
                    
                    for unsolved_sample_id in unsolved_sample_ids:
                        similarity_score, confidence_score = self.demo_sele.get_diverse_score(unsolved_sample_id)
                        demo_score = similarity_score + confidence_score
                        next_state = current_state + [unsolved_sample_id]
                        next_state2score[tuple(next_state)] = demo_score
                    
                    sorted_next_state2score = sorted(next_state2score.items(), key=lambda x: x[1], reverse=True)
                    selected_sorted_next_state2score = sorted_next_state2score[: min(self.args.beam_search_expand_num, len(sorted_next_state2score))]
                
                    for next_state, score in selected_sorted_next_state2score:
                        current_id = next_state[-1]
                        current_sample = self.sampleid2sample[current_id]
                        query, current_sample, sorted_softmax_token2logprob, entropy = self.inference(current_sample, use_demonstrations=True)
                        self.inference_num += 1
                        self.sampleid2sample[current_id] = current_sample
                        self.sampleid2info[current_id].append((current_sample.pseudo_label, sorted_softmax_token2logprob))
                        confidence = sorted_softmax_token2logprob[0][1]
                        state2reward[tuple(next_state)] = confidence
                
                sorted_state2reward = sorted(state2reward.items(), key=lambda x: x[1], reverse=True)
                select_sorted_state2reward = sorted_state2reward[:min(self.args.select_num, len(unsolved_sample_ids)-1)]
                current_states = []
            
                for state, reward in select_sorted_state2reward:
                    add_id = state[-1]
                    add_sample = self.sampleid2sample[add_id]
                    current_states.append(list(state))
                    self.demo_sele.add_sample(add_id, add_sample)
                    self.retriever.add_sample(add_id, add_sample)
                
                pbar.update(1)
            
            iter_num = 0
            self.model_debias(iter_num, subset)
            pbar.close()
                
        elif self.args.search_strategy == "MCTS":
            self.demo_sele.calc_similarity(self.sampleid2sample)
            
            self.sampleid2info = defaultdict(list)
            self.visited = defaultdict(int)
            self.children = defaultdict(list)
            self.parent = defaultdict(tuple)
            self.action_cache = defaultdict(list)
            self.rewards = defaultdict(float)
            self.cum_rewards = defaultdict(list)
            
            self.start_state = [self.start_node]
            start_sample = self.sampleid2sample[self.start_node]
            start_query, start_sample, start_sorted_softmax_token2logprob, start_entropy = self.inference(start_sample, use_demonstrations=True)
            self.inference_num += 1
            self.sampleid2sample[self.start_node] = start_sample
            if self.args.model != "gpt-4o-mini":
                self.sampleid2info[self.start_node].append((start_sample.pseudo_label, start_sorted_softmax_token2logprob))
            else:
                self.sampleid2info[self.start_node].append(start_sample.pseudo_label)
            if self.args.model != "gpt-4o-mini":
                start_reward = start_sorted_softmax_token2logprob[0][1]
                self.rewards[tuple(self.start_state)] = start_reward
            else:
                self.rewards[tuple(self.start_state)] = 0
            self.demo_sele.add_sample(self.start_node, self.sampleid2sample[self.start_node])
            self.demo_sele.add_mask(self.start_node)
            self.retriever.add_sample(self.start_node, self.sampleid2sample[self.start_node])
            
            self.parent[tuple(self.start_state)] = -1
            
            for iter_num in range(self.args.iterative_num):
                print(f"--------------Iteration: {iter_num}--------------")
                # selection, expansion, simulation, back-propagation

                path = self._select(self.start_state)
                last_state = path[-1]
                unsolved_num = len(self.sampleid2sample) - len(last_state)
                self.pbar = tqdm(total=unsolved_num, desc=f"MCTS Traversing nodes Iteration {iter_num}", unit="node")
                
                if self.args.dataset == "bbh-mini":
                    subset = self.sampleid2sample[last_state[-1]].subset
                
                if not self.is_terminal_state(path[-1]):
                    path, is_terminal_state = self._expand(path)
                    path = self._simulate(path)
                self._back_propagate(path)
                self.model_debias(iter_num, subset)
                
    def is_terminal_state(self, state):
        if len(state) == len(self.sampleid2sample):
            return True
        else:
            return False
    
    def dq(self, state) -> float:
        if self.args.uct == "random":
            num = random.random()
            return num
        
        elif self.args.uct == "uct":
            if len(self.cum_rewards[tuple(state)]) == 0:
                expected_q = 0
            else:
                expected_q = np.mean(self.cum_rewards[tuple(state)])
            return expected_q
        
        elif self.args.uct == "demo_uct":
            last_id = state[-1]
            similarity_score, confidence_score = self.demo_sele.get_diverse_score(last_id)
            demo_score = similarity_score + confidence_score
            if len(self.cum_rewards[tuple(state)]) == 0:
                expected_q = 0
            else:
                expected_q = np.mean(self.cum_rewards[tuple(state)])
            dq = self.args.c_exp * expected_q + demo_score
            return dq
    
    def _uct(self, state) -> float:
        
        if self.args.uct == "random":
            num = random.random()
            return num
        
        elif self.args.uct == "uct":
            if self.parent[tuple(state)] == -1:
                N_parent = 0
            else:
                N_parent = self.visited[tuple(self.parent[tuple(state)])]
            if len(self.cum_rewards[tuple(state)]) == 0:
                expected_q = 0
            else:
                expected_q = np.mean(self.cum_rewards[tuple(state)])
            return expected_q + self.args.w_exp * np.sqrt(np.log(N_parent+1) / (self.visited[tuple(state)]+1))
        
        elif self.args.uct == "demo_uct":
            last_id = state[-1]
            similarity_score, confidence_score = self.demo_sele.get_diverse_score(last_id)
            demo_score = similarity_score + confidence_score
            if len(self.cum_rewards[tuple(state)]) == 0:
                expected_q = 0
            else:
                expected_q = np.mean(self.cum_rewards[tuple(state)])
            if self.parent[tuple(state)] == -1:
                N_parent = 0
            else:
                N_parent = self.visited[tuple(self.parent[tuple(state)])]
            uct = self.args.c_exp * expected_q + demo_score + self.args.w_exp * np.sqrt(np.log(N_parent+1) / (self.visited[tuple(state)]+1))
            return uct
    
    def _uct_select(self, state):
        
        max_uct =  0
        max_uct_state = None
        for child in self.children[tuple(state)]:
            uct = self._uct(child)
            if uct > max_uct:
                max_uct = uct
                max_uct_state = child
        return max_uct_state
    
    def get_all_leaf_paths(self, state):
        all_paths = []
        current_path = []

        def dfs(state):
            current_path.append(state)
            if len(self.children[tuple(state)]) == 0 or self.is_terminal_state(state):
                all_paths.append(deepcopy(current_path))
            else:
                for child in self.children[tuple(state)]:
                    dfs(child)
                    
            current_path.pop()

        dfs(state)
        return all_paths

    def cal_cum_reward(self, rewards):
        return sum(rewards)
    
    def _select(self, state):
        
        if self.args.uct == "uct" or self.args.uct == "demo_uct":        
            path = []
            while True:
                path.append(state)
                self.visited[tuple(state)] += 1
                if len(self.children[tuple(state)]) == 0 or self.is_terminal_state(state):
                    return path
                state = self._uct_select(state)
        elif self.args.uct == "random":
            path_list = self.get_all_leaf_paths(state)
            path = random.choice(path_list)
            return path
    
    def _expand(self, path):
        
        state = path[-1]
        
        if self.is_terminal_state(state):
            return path, True
        
        current_id = state[-1]
        current_sample = self.sampleid2sample[current_id]
        subset = current_sample.subset
        
        unsolved_sample_ids = list(set(self.sampleid2sample.keys()) - set(state))
        
        state2dq = defaultdict(float)
        for unsolved_sample_id in unsolved_sample_ids:
            next_state = state + [unsolved_sample_id]
            dq_score = self.dq(next_state)
            state2dq[unsolved_sample_id] = dq_score
        
        sorted_state2dq = sorted(state2dq.items(), key=lambda x: x[1], reverse=True)
        selected_sorted_state2dq = sorted_state2dq[: min(self.args.expansion_num, len(sorted_state2dq))]
        action_rewards = {}
        
        for i, (selected_sampleid, dq_score) in enumerate(selected_sorted_state2dq):
            
            child_state = state + [selected_sampleid]
            self.children[tuple(state)].append(child_state)
            self.parent[tuple(child_state)] = state
            
            if self.args.use_cache and len(self.action_cache[selected_sampleid]) >= self.args.traversed_times:
                rewards_list = self.action_cache[selected_sampleid]
                action_rewards[selected_sampleid] = sum(rewards_list) / len(rewards_list)
            else:
                self.inference_num += 1
                current_sample = self.sampleid2sample[selected_sampleid]
                query, current_sample, sorted_softmax_token2logprob, entropy = self.inference(current_sample, use_demonstrations=True)
                
                if self.args.model != "gpt-4o-mini":
                    self.sampleid2info[selected_sampleid].append((current_sample.pseudo_label, sorted_softmax_token2logprob))
                    confidence = sorted_softmax_token2logprob[0][1]
                    reward = confidence
                else:
                    self.sampleid2info[selected_sampleid].append(current_sample.pseudo_label)
                    reward = 0
                        
                action_rewards[selected_sampleid] = reward
                self.rewards[tuple(child_state)] = reward
                if self.args.use_cache and dq_score > self.args.epsilon:
                    self.action_cache[selected_sampleid].append(reward)
        
        sorted_action_rewards = sorted(action_rewards.items(), key=lambda x: x[1], reverse=True)        
        next_child_id = sorted_action_rewards[0][0]
        next_child_state = state + [next_child_id]
        self.demo_sele.add_sample(next_child_id, self.sampleid2sample[next_child_id])
        self.demo_sele.add_mask(next_child_id)
        self.retriever.add_sample(next_child_id, self.sampleid2sample[next_child_id])
        
        self.visited[tuple(next_child_state)] += 1
        path.append(next_child_state)
        
        self.pbar.update(1)
        
        return path, False
    
    def _simulate(self, path):
        
        while True:
            path, is_terminal_state = self._expand(path)
            if is_terminal_state:
                break
        
        return path
    
    def _back_propagate(self, path):
        rewards = []
        cum_rewards = []
        
        for state in reversed(path):
            rewards.append(self.rewards[tuple(state)])
            cum_reward = sum(rewards[::-1])
            cum_rewards.append(cum_reward)
            self.cum_rewards[tuple(state)].append(cum_reward / len(rewards))
    
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
    
    def inference(self, sample, use_demonstrations):
        subset = sample.subset
        if use_demonstrations:
            query = self.retriever.get_final_query(sample)
        else:
            query = sample.question
        if self.args.dataset == "bbh" or self.args.dataset == "bbh-mini":
            query = f"Task description: {self.task_description_map[subset]}" + "\n\n" + query           
            
        label, sorted_softmax_token2logprob, entropy = self.get_query_response(query, subset)
        
        sample.entropy = entropy
        sample.usable = True
        if (self.args.dataset == "bbh" or self.args.dataset == "bbh-mini") and subset in self.is_choices.keys() and self.args.model != "gpt-4o-mini":
            sorted_softmax_token2logprob = [(f'({key})', value) for key, value in sorted_softmax_token2logprob]
        sample.sorted_softmax_token2logprob = sorted_softmax_token2logprob
        if self.args.model != "gpt-4o-mini":
            confidence = sorted_softmax_token2logprob[0][1]
            sample.confidence = confidence
        if self.args.dataset == "bbh" or self.args.dataset == "bbh-mini":
            if subset in self.is_choices.keys():
                sample.demonstration = sample.question + label + ")"
                sample.pseudo_label = f"({label})"
            else:
                sample.demonstration = sample.question + label
                sample.pseudo_label = label
        elif self.args.dataset == "mmlu":
            sample.demonstration = sample.question + label
            sample.pseudo_label = label
            
        return query, sample, sorted_softmax_token2logprob, entropy

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
    
    def model_debias(self, iter_num, subset):
        
        if self.args.dataset == "bbh" or self.args.dataset == "mmlu":
        
            self.subset_right_num = {}
            self.subset_test_num = {}
            global_right_sample_num = 0
            global_test_sample_num = 0
            self.result_dict = defaultdict(list)
            
            self.subset_right_num[subset] = 0
            self.subset_test_num[subset] = len(self.sampleid2sample.keys())
            acc_list = []
            
            if self.args.model != "gpt-4o-mini":
                
                if self.args.aggregation:
                    for sample_idx in list(self.sampleid2info.keys()):
                        label_probabilities = {}
                        sample_info = self.sampleid2info[sample_idx]
                        sorted_softmax_values = [sorted_softmax_token2logprob for pred_label, sorted_softmax_token2logprob in sample_info]
                        for sorted_softmax_value in sorted_softmax_values:
                            for label_prob in sorted_softmax_value:
                                label = label_prob[0]
                                prob = label_prob[1]
                                if label not in label_probabilities:
                                    label_probabilities[label] = prob
                                else:
                                    label_probabilities[label] += prob
                        for label, prob in label_probabilities.items():
                            label_probabilities[label] = prob / len(sorted_softmax_values)
                        sorted_label_probabilities = sorted(label_probabilities.items(), key=lambda x: x[1], reverse=True)
                        self.sampleid2sample[sample_idx].sorted_softmax_token2logprob = sorted_label_probabilities
                    
                if self.args.calibration:
                    self.label_bias = {}
                    for sample_idx in list(self.sampleid2sample.keys()):
                        sample = self.sampleid2sample[sample_idx]
                        sample_token2logprob = sample.sorted_softmax_token2logprob
                        for label2prob in sample_token2logprob:
                            l = label2prob[0]
                            p = label2prob[1]
                            if l not in self.label_bias:
                                self.label_bias[l] = p
                            else:
                                self.label_bias[l] += p
                    
                    for label, prob in self.label_bias.items():
                        self.label_bias[label] = prob / len(self.sampleid2sample)
                    
                for sample_idx in list(self.sampleid2sample.keys()):
                    sample = self.sampleid2sample[sample_idx]
                    sorted_softmax_token2logprob = sample.sorted_softmax_token2logprob
                    if self.args.calibration:
                        label_prob_res = {}
                        for p in sorted_softmax_token2logprob:
                            label = p[0]
                            prob = p[1]
                            label_prob_res[label] = prob / self.label_bias[label]
                        label_prob_res = sorted(label_prob_res.items(), key=lambda x: x[1], reverse=True)
                        pseudo_label = label_prob_res[0][0]
                        self.sampleid2sample[sample_idx].pseudo_label = pseudo_label
                    else:
                        pseudo_label = self.sampleid2sample[sample_idx].pseudo_label
                    gt_answer = sample.label
                    acc = int(pseudo_label == gt_answer)
                    acc_list.append(acc)
            else:
                if self.args.aggregation:
                    for sample_idx in list(self.sampleid2info.keys()):
                        sample_info = self.sampleid2info[sample_idx]
                        counter = Counter(sample_info)
                        pred_label = counter.most_common(1)[0][0]
                        self.sampleid2sample[sample_idx].pseudo_label = pred_label
                        
                for sample_idx in list(self.sampleid2sample.keys()):
                    pseudo_label = self.sampleid2sample[sample_idx].pseudo_label
                    gt_answer = self.sampleid2sample[sample_idx].label
                    acc = int(pseudo_label == gt_answer)
                    acc_list.append(acc)
            
            if self.args.save_res:
                for sample_idx, acc in zip(list(self.sampleid2sample.keys()), acc_list):
                    sample = self.sampleid2sample[sample_idx]
                    res_data = {
                        "idx": sample_idx,
                        "res_prob": sample.sorted_softmax_token2logprob,
                        "pred": sample.pseudo_label,
                        "gt_ans": sample.label,
                        "acc": acc,
                    }
                    self.result_dict[subset].append(res_data)
                
                subset_save_dir = f"zsicl-output/{self.args.dataset}/{self.args.model}/{self.args.method}_{self.args.search_strategy}_{self.args.select_strategy}_{self.args.uct}_{self.args.expansion_num}_{self.args.w_exp}_{self.args.diverse_candidate}/{iter_num}"
                os.makedirs(subset_save_dir, exist_ok=True)
                for subset, res_list in self.result_dict.items():
                    res_path = f"{subset_save_dir}/{subset}.jsonl"
                    res_list = sorted(res_list, key=lambda x: x["idx"])
                    with open(res_path, 'w') as file:
                        for res_data in res_list:
                            file.write(json.dumps(res_data) + '\n')
            
            if self.args.dataset == "bbh":
                global_binary_choice_right_sample_num = 0
                global_binary_choice_total_sample_num = 0
                
                global_multiple_choice_right_sample_num = 0
                global_multiple_choice_total_sample_num = 0
                
            elif self.args.dataset == "mmlu":
                global_humanities_right_sample_num = 0
                global_humanities_total_sample_num = 0
                
                global_stem_right_sample_num = 0
                global_stem_total_sample_num = 0
                
                global_social_science_right_sample_num = 0
                global_social_science_total_sample_num = 0
                
                global_others_right_sample_num = 0
                global_others_total_sample_num = 0
            
            for acc in acc_list:
                global_test_sample_num += 1
                if acc == 1:
                    self.subset_right_num[subset] += 1
                    global_right_sample_num += 1
                if self.args.dataset == "bbh":
                    if subset in self.is_choices.keys():
                        global_multiple_choice_total_sample_num += 1
                        if acc == 1:
                            global_multiple_choice_right_sample_num += 1
                    else:
                        global_binary_choice_total_sample_num += 1
                        if acc == 1:
                            global_binary_choice_right_sample_num += 1
                elif self.args.dataset == "mmlu":
                    if self.is_choices[subset] == "STEM":
                        global_stem_total_sample_num += 1
                        if acc == 1:
                            global_stem_right_sample_num += 1
                    elif self.is_choices[subset] == "Social Science":
                        global_social_science_total_sample_num += 1
                        if acc == 1:
                            global_social_science_right_sample_num += 1
                    elif self.is_choices[subset] == "Humanities":
                        global_humanities_total_sample_num += 1
                        if acc == 1:
                            global_humanities_right_sample_num += 1
                    elif self.is_choices[subset] == "Others":
                        global_others_total_sample_num += 1
                        if acc == 1:
                            global_others_right_sample_num += 1
        
            acc = self.subset_right_num[subset] / self.subset_test_num[subset]
            self.results[iter_num][subset] = round(acc * 100, 2)
            print(f"Iter_num: {iter_num}, subset: {subset}, acc: {round(acc * 100, 2)}")
            
            self.results[iter_num]["global_test_sample_num"] += global_test_sample_num
            self.results[iter_num]["global_right_sample_num"] += global_right_sample_num
            
            if self.args.dataset == "bbh":
                self.results[iter_num]["global_binary_choice_total_sample_num"] += global_binary_choice_total_sample_num
                self.results[iter_num]["global_binary_choice_right_sample_num"] += global_binary_choice_right_sample_num
                self.results[iter_num]["global_multiple_choice_total_sample_num"] += global_multiple_choice_total_sample_num
                self.results[iter_num]["global_multiple_choice_right_sample_num"] += global_multiple_choice_right_sample_num
            elif self.args.dataset == "mmlu":
                self.results[iter_num]["global_stem_total_sample_num"] += global_stem_total_sample_num
                self.results[iter_num]["global_stem_right_sample_num"] += global_stem_right_sample_num
                self.results[iter_num]["global_stem_total_sample_num"] += global_stem_total_sample_num
                self.results[iter_num]["global_stem_right_sample_num"] += global_stem_right_sample_num
                self.results[iter_num]["global_social_science_total_sample_num"] += global_social_science_total_sample_num
                self.results[iter_num]["global_social_science_right_sample_num"] += global_social_science_right_sample_num
                self.results[iter_num]["global_humanities_total_sample_num"] += global_humanities_total_sample_num
                self.results[iter_num]["global_humanities_right_sample_num"] += global_humanities_right_sample_num
                self.results[iter_num]["global_others_total_sample_num"] += global_others_total_sample_num
                self.results[iter_num]["global_others_right_sample_num"] += global_others_right_sample_num
        
        elif self.args.dataset == "bbh-mini":
            global_right_sample_num = 0
            global_test_sample_num = 0
            self.result_dict = defaultdict(list)
            self.subset_right_num = defaultdict(int)
            self.subset_test_num = defaultdict(int)
            
            self.subset_right_num[subset] = 0
            self.subset_test_num[subset] = len(self.sampleid2sample.keys())
            
            if self.args.dataset == "bbh-mini":
                global_binary_choice_right_sample_num = 0
                global_binary_choice_total_sample_num = 0
                
                global_multiple_choice_right_sample_num = 0
                global_multiple_choice_total_sample_num = 0
            
            if self.args.calibration:
                pred_history = defaultdict(list)
                self.label_bias = defaultdict(lambda: defaultdict(float))
                
                for sample_idx in list(self.sampleid2sample.keys()):
                    subset = self.sampleid2sample[sample_idx].subset
                    sample_token2logprob = self.sampleid2sample[sample_idx].sorted_softmax_token2logprob
                    pred_history[subset].append(sample_token2logprob)
                    
                for subset, sorted_softmax_token2logprobs in pred_history.items():
                    for sorted_softmax_token2logprob in sorted_softmax_token2logprobs:
                        for p in sorted_softmax_token2logprob:
                            label = p[0]
                            prob = p[1]
                            self.label_bias[subset][label] += prob
                
                for subset, label2prob in self.label_bias.items():
                    for label, prob in label2prob.items():
                        self.label_bias[subset][label] = prob / len(pred_history[subset])
            
            for sample_idx in list(self.sampleid2sample.keys()):                
                if self.args.calibration:
                    sample = self.sampleid2sample[sample_idx]
                    subset = self.sampleid2sample[sample_idx].subset
                    sorted_softmax_token2logprob = self.sampleid2sample[sample_idx].sorted_softmax_token2logprob
                    label_prob_res = {}
                    for p in sorted_softmax_token2logprob:
                        label = p[0]
                        prob = p[1]
                        label_prob_res[label] = prob / self.label_bias[subset][label]
                    label_prob_res = sorted(label_prob_res.items(), key=lambda x: x[1], reverse=True)
                    pseudo_label = label_prob_res[0][0]
                    if subset in self.is_choices.keys() and self.args.dataset == "bbh-mini":
                        pseudo_label = f"({pseudo_label})"
                    self.sampleid2sample[sample_idx].pseudo_label = pseudo_label
                    gt_answer = self.sampleid2sample[sample_idx].label
                    acc = int(pseudo_label == gt_answer)
                    
                else:
                    sample = self.sampleid2sample[sample_idx]
                    subset = self.sampleid2sample[sample_idx].subset
                    pseudo_label = self.sampleid2sample[sample_idx].pseudo_label
                    gt_answer = self.sampleid2sample[sample_idx].label
                    acc = int(pseudo_label == gt_answer)
                
                global_test_sample_num += 1
                if acc == 1:
                    self.subset_right_num[subset] += 1
                    global_right_sample_num += 1
                if self.args.dataset == "bbh-mini":
                    if subset in self.is_choices.keys():
                        global_multiple_choice_total_sample_num += 1
                        if acc == 1:
                            global_multiple_choice_right_sample_num += 1
                    else:
                        global_binary_choice_total_sample_num += 1
                        if acc == 1:
                            global_binary_choice_right_sample_num += 1
                
            self.results[iter_num]["global_test_sample_num"] += global_test_sample_num
            self.results[iter_num]["global_right_sample_num"] += global_right_sample_num
            
            if self.args.dataset == "bbh-mini":
                self.results[iter_num]["global_binary_choice_total_sample_num"] += global_binary_choice_total_sample_num
                self.results[iter_num]["global_binary_choice_right_sample_num"] += global_binary_choice_right_sample_num
                self.results[iter_num]["global_multiple_choice_total_sample_num"] += global_multiple_choice_total_sample_num
                self.results[iter_num]["global_multiple_choice_right_sample_num"] += global_multiple_choice_right_sample_num
                

    def run(self):
        self.inference_num = 0
        self.results = defaultdict(lambda: defaultdict(float))
        
        if self.args.dataset == "bbh" or self.args.dataset == "mmlu":
            for subset in self.subsets:
                self.dataset_list = []
                self.sampleid2sample = {}
                self.demo_sele.demonstrations = {}
                self.retriever.demonstrations = {}
                if self.args.dataset == "bbh":
                    with open(f"data/{self.args.dataset}/{subset}.json", "r") as f:
                        dataset = json.load(f)["examples"]
                elif self.args.dataset == "mmlu":
                    dataset = read_jsonl(f"data/{self.args.dataset}/{subset}/test.jsonl")
                    
                for idx, d in enumerate(dataset):
                    d["idx"] = idx
                    d["subset"] = subset
                    self.dataset_list.append(d)
                    self.preprocess(idx, self.dataset_list[idx])
                
                self.start_node = 0
                self.Search(subset)
                
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
            
            for idx in range(len(self.dataset_list)):    
                self.preprocess(idx, self.dataset_list[idx])
            
            random.shuffle(self.dataset_list)
            self.start_node = 0
            self.Search(subset)
            
        for iter_num in range(self.args.iterative_num):
            self.results[iter_num]["avg"] = round(self.results[iter_num]["global_right_sample_num"] / self.results[iter_num]["global_test_sample_num"] * 100, 2)
            if self.args.dataset == "bbh" or self.args.dataset == "bbh-mini":
                self.results[iter_num]["binary_choice"] = round(self.results[iter_num]["global_binary_choice_right_sample_num"] / self.results[iter_num]["global_binary_choice_total_sample_num"] * 100, 2)
                self.results[iter_num]["multiple_choice"] = round(self.results[iter_num]["global_multiple_choice_right_sample_num"] / self.results[iter_num]["global_multiple_choice_total_sample_num"] * 100, 2)
        
        return self.results
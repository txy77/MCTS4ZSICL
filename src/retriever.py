import torch
import random
import pdb
from typing import List
from rank_bm25 import BM25Okapi

class DynamicReteiever:
    def __init__(self, args, target_tokenizer, target_model, device):
        self.args = args
        self.target_tokenizer = target_tokenizer
        self.target_model = target_model
        self.device = device
        self.demonstrations = {}
        self.dnum = 0

    def get_final_query(self, sample):
        demonstration_samples = self.get_demonstrations_from_bank(sample)
        demonstrations_list = [sample.demonstration for sample in demonstration_samples]
        demonstrations = "\n\n".join(demonstrations_list) + "\n\n" if demonstrations_list else ""
        query = demonstrations + sample.question
        return query

    def get_demonstrations_from_bank(self, sample):
        usable_demonstrations = [sample for sample in list(self.demonstrations.values()) if sample.usable == True]
        self.dnum = min(len(usable_demonstrations), self.args.shot_num)
        if self.dnum == 0:
            return []
        if self.args.select_strategy == "random":
            indices = self.get_random(sample)
        elif self.args.select_strategy == "bm25":
            indices = self.get_bm25(sample)
        elif self.args.select_strategy == "topk":
            indices = self.get_topk(sample)
        elif self.args.select_strategy == "dpp":
            if self.args.method == "DAIL":
                indices = self.get_dpp(sample, is_normalize=True)
            else:
                indices = self.get_dpp(sample, is_normalize=False)
            indices = self.get_dpp(sample)
        elif self.args.select_strategy == "diverse":
            indices = self.get_diverse(sample)
        else:
            print("select_strategy is not effective.")
            return
        samples = [self.demonstrations[i] for i in indices]
        return samples
    
    @staticmethod
    def normalize(tensor):
        mean_value = torch.mean(tensor, dim=-1)
        mean_adjusted_tensor = tensor - mean_value
        std_value = torch.std(mean_adjusted_tensor, dim=-1)
        standardized_tensor = mean_adjusted_tensor / std_value
        return standardized_tensor

    def get_random(self, sample):
        demonstration_keys = list(self.demonstrations.keys())
        indices = random.sample(demonstration_keys, self.dnum)
        return indices
    
    def get_bm25(self, sample):
        demonstration_keys = list(self.demonstrations.keys())
        tokenized_examples = [example.question.split() for example in self.demonstrations]
        tokenized_query = sample.question.split()
        bm25 = BM25Okapi(tokenized_examples)
        bm25_scores = torch.tensor(bm25.get_scores(tokenized_query))
        values, indices = torch.topk(bm25_scores, self.dnum, largest=True)
        indices = indices.tolist()
        indices = [demonstration_keys[i] for i in indices]
        return indices
    
    def get_topk(self, sample):
        demonstration_keys = list(self.demonstrations.keys())
        demonstration_embeds = torch.stack([self.demonstrations[key].embed for key in demonstration_keys], dim=0)
        topk_scores = torch.cosine_similarity(demonstration_embeds, sample.embed, dim=-1)
        values, indices = torch.topk(topk_scores, self.dnum, largest=True)
        indices = indices.tolist()
        indices = [demonstration_keys[i] for i in indices]
        return indices
    
    def get_dpp(self, sample, is_normalize=False):
        demonstration_keys = list(self.demonstrations.keys())
        demonstration_embeds = torch.stack([self.demonstrations[key].embed for key in demonstration_keys], dim=0)
        topk_scores = torch.cosine_similarity(demonstration_embeds, sample.embed, dim=-1)
        if self.args.model != "gpt-4o-mini" and is_normalize:
            normalized_topk_scores = self.normalize(topk_scores)
            entropy_scores = self.normalize(torch.stack([sample.entropy for sample in list(self.demonstrations.values())], dim=0))
            scores = normalized_topk_scores - self.args.alpha * entropy_scores
        else:
            scores = topk_scores
        values, indices = torch.topk(scores, min(len(self.demonstrations), self.args.dpp_candidates), largest=True)
        candidates = demonstration_embeds[indices]
        near_reps, rel_scores, kernel_matrix = self.get_kernel(sample.embed, candidates)
        rel_scores = rel_scores.cpu()
        kernel_matrix = kernel_matrix.cpu()
        samples_ids = torch.tensor(self.fast_map_dpp(kernel_matrix))
        samples_scores = rel_scores[samples_ids]
        _, ordered_indices = torch.sort(samples_scores, descending=False)
        sample_indices = samples_ids[ordered_indices]
        indices = indices[sample_indices]
        indices = indices.tolist()
        indices = [demonstration_keys[i] for i in indices]
        return indices

    def get_kernel(self, embed, candidates):
        near_reps = candidates
        embed = embed / torch.linalg.norm(embed)
        near_reps = near_reps / torch.linalg.norm(near_reps, keepdims=True, dim=1)
        rel_scores = torch.matmul(embed, near_reps.T)
        rel_scores = (rel_scores + 1) / 2
        rel_scores -= rel_scores.max()
        rel_scores = torch.exp(rel_scores / (2 * self.args.scale_factor))
        sim_matrix = torch.matmul(near_reps, near_reps.T)
        sim_matrix = (sim_matrix + 1) / 2
        kernel_matrix = rel_scores[None] * sim_matrix * rel_scores[:, None]
        return near_reps, rel_scores, kernel_matrix

    def fast_map_dpp(self, kernel_matrix):
        item_size = kernel_matrix.size()[0]
        cis = torch.zeros([self.dnum, item_size])
        di2s = torch.diag(kernel_matrix)
        selected_items = list()
        selected_item = torch.argmax(di2s)
        selected_items.append(int(selected_item))
        while len(selected_items) < self.dnum:
            k = len(selected_items) - 1
            ci_optimal = cis[:k, selected_item]
            di_optimal = torch.sqrt(di2s[selected_item])
            elements = kernel_matrix[selected_item, :]
            eis = (elements - torch.matmul(ci_optimal, cis[:k, :])) / di_optimal
            cis[k, :] = eis
            di2s -= torch.square(eis)
            selected_item = torch.argmax(di2s)
            selected_items.append(int(selected_item))
        return selected_items
    
    def get_diverse(self, sample):
        demonstration_keys = list(self.demonstrations.keys())
        demonstration_embeds = torch.stack([self.demonstrations[key].embed for key in demonstration_keys], dim=0)
        
        topk_scores = torch.cosine_similarity(demonstration_embeds, sample.embed, dim=-1)
        values, indices = torch.sort(topk_scores, descending=True)
        indices = indices.tolist()
        sorted_keys = [demonstration_keys[i] for i in indices[:min(len(indices), self.args.diverse_candidate)]]
        random.shuffle(sorted_keys)
        
        pseudo_labels = [self.demonstrations[k].pseudo_label for k in sorted_keys]
        selected_indices = []
        unique_pseudo_labels = set()
        
        for i, pseudo_label in enumerate(pseudo_labels):
            if pseudo_label not in unique_pseudo_labels:
                selected_indices.append(sorted_keys[i])
                unique_pseudo_labels.add(pseudo_label)
            if len(selected_indices) >= self.dnum:
                break
                
        if len(selected_indices) < self.dnum:
            for i, psuedo_label in enumerate(pseudo_labels):
                if indices[i] not in selected_indices:
                    selected_indices.append(sorted_keys[i])
                if len(selected_indices) == self.dnum:
                    break
        
        return selected_indices

    def add_sample(self, sample_id, sample):
        self.demonstrations[sample_id] = sample
            
    def make_empty(self):
        self.demonstrations = {}
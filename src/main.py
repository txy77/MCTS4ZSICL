import argparse
from sentence_transformers import SentenceTransformer
import torch
import pdb
import os
import sys
import random
import time
import numpy as np
from openai import OpenAI
from accelerate import Accelerator

ZS_ICL_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)
sys.path.insert(0, ZS_ICL_ROOT_PATH)

from src.method.Base import method2class
from src.utils import transfor_model2model_path, load_llm
from data_utils.bbh_utils import load_bbh_config
from data_utils.mmlu_utils import load_mmlu_config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="llama3.1-8b")
    parser.add_argument("--method", type=str, default="Search")
    parser.add_argument("--dataset", type=str, default="bbh")
    parser.add_argument("--shot_num", type=int, default=3)
    
    # openai
    parser.add_argument("--api_key", type=str)
    
    # Demonstration Selection Method
    parser.add_argument("--select_strategy", type=str, default="dpp")
    
    # Hyper parameters for DAIL dpp
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--dpp_candidates", type=int, default=10)
    parser.add_argument("--diverse_candidate", type=int, default=50)
    parser.add_argument("--scale_factor", type=float, default=0.1)
    
    # Hyper parameters for Search
    parser.add_argument("--search_strategy", type=str, default="no")
    
    # Hyper parameters for Beam_Search
    parser.add_argument("--beam_search_expand_num", type=int, default=0)
    parser.add_argument("--select_num", type=int, default=0)
    
    # Hyper parameters for MCTS Search
    parser.add_argument("--expansion_num", type=int, default=0)
    parser.add_argument('--iterative_num', type=int, default=0)
    parser.add_argument('--uct', type=str, default="no")
    parser.add_argument('--w_exp', type=float, default=5)
    parser.add_argument('--c_exp', type=float, default=1)
    
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--traversed_times', type=int, default=1)
    parser.add_argument('--epsilon', type=float, default=1.5)
    
    # debias
    parser.add_argument('--aggregation', action='store_true')
    parser.add_argument('--calibration', action='store_true')
    
    parser.add_argument('--save_res', action='store_true')
    
    arguments = parser.parse_args()
    return arguments




if __name__ == "__main__":
    args = get_args()
    if args.seed == 42:
        random.seed(args.seed)
    
    # dataset
    if args.dataset == "bbh" or args.dataset == "bbh-mini":
        subsets, label_space_map, is_choices, task_description, task_type = load_bbh_config()
    if args.dataset == "mmlu":
        task_subsets, stem_subjects, social_sciences_subjects, humanities_subjects, other_subjects = load_mmlu_config()
        subsets = {}
        label_space_map = {}
        is_choices = {}
        for task_subset in task_subsets:
            subsets[task_subset] = r'[A-D]'
            label_space_map[task_subset] = ['A', 'B', 'C', 'D']
            if task_subset in stem_subjects:
                is_choices[task_subset] = "STEM"
            elif task_subset in social_sciences_subjects:
                is_choices[task_subset] = "Social Science"
            elif task_subset in humanities_subjects:
                is_choices[task_subset] = "Humanities"
            elif task_subset in other_subjects:
                is_choices[task_subset] = "Others"
            else:
                raise ValueError("Unknown subject")
            task_description = None
    
    # model
    accelerator = Accelerator()
    device = torch.device(args.device)
    sentence_model = SentenceTransformer("BAAI/bge-large-en").to(device)
    
    if args.model == "gpt-4o-mini":
        client = OpenAI(api_key=args.api_key)
        target_tokenizer = None
        target_model = args.model
    else:
        model_path = transfor_model2model_path(args.model)
        target_tokenizer, target_model = load_llm(model_path, device, accelerator)
        client = None
    
    start_time = time.time()
    
    assert args.method in {"ZS", "FS", "SelfICL", "DAIL", "DAIL_CALI", "Search"}
    inferencer = method2class[args.method](args, sentence_model=sentence_model, target_tokenizer=target_tokenizer, target_model=target_model, subsets=subsets, label_space_map=label_space_map, task_description=task_description, is_choices=is_choices, device=device, accelerator=accelerator, client=client)
    results = inferencer.run()
    
    end_time = time.time()
    execution_time = end_time - start_time

    results = {"model": args.model, "method": args.method, "dataset": args.dataset, "shot_num": args.shot_num, "search_strategy": args.search_strategy, "expansion_num": args.expansion_num, "select_num": args.select_num, "iterative_num": args.iterative_num, "uct": args.uct, "diverse_candidate": args.diverse_candidate, "w_exp": args.w_exp, "select_strategy": args.select_strategy, "alpha": args.alpha, "dpp_candidates": args.dpp_candidates, "scale_factor": args.scale_factor, "calibration": args.calibration, "use_cache": args.use_cache, "results": results, "execution_time": execution_time, "inference_num": inferencer.inference_num, "token": inferencer.tokens}
    print("-------------------------final-results-----------------------")
    print(results)
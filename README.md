# DAWN-ICL: Strategic Planning of Problem-solving Trajectories for Zero-Shot In-Context Learning

## 😀 Overview

+ We are the first to formalize ZS-ICL as a planning problem, which is closer to real-world scenarios.

+ We propose a novel demonstration-aware MCTS for ZS-ICL to achieve a more effective and efficient search for the problem-solving trajectories.

+ Extensive experiments demonstrate the effectiveness of our approach on in-domain and cross-domain scenarios, and it even outperforms ICL using human-annotated demonstrations.

<p align="center">
  <img src="./assets/dawnicl.png" width="75%" height="75% title="The overview of DAWN-ICL" alt="">
</p>

## 🚀 Quick Start

### Requirements

- python == 3.11.9
- pytorch == 2.3.1
- transformers == 4.42.4
- accelerate == 0.33.0
- openai==1.35.14

### Download Models

Download models from huggingface, open the src/utils.py file and update the directory paths in lines 34-41.

### Parameter Settings

+ model: large language models
+ method: method of zero-shot in-context learning: ZS, FS, SelfICL, DAIL, Search
+ dataset: evaluation dataset: bbh, bbh-mini, mmlu
+ shot_num: the shot number of in-context learning
+ select_strategy: the demonstration selection strategy of in-context learning
+ diverse_candidate: the number of retrieved candidates: $k_d$
+ search_strategy: the search strategy of zero-shot in-context learning: Greedy, MC, Beam_Search, MCTS
+ expansion_num: the expansion number of MCTS: $k_a$
+ iterative_num: the iteration number of MCTS
+ use_cache: whether to use the cache strategy
+ aggregation: whether to use the aggregation strategy
+ calibration: whether to use the calibration strategy

### Run

You can get the results of our method by running the following command

```bash
bash run_dawn_icl.sh
```
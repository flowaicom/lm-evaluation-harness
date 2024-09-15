from datasets import Dataset
from typing import List, Dict, Any
from lm_eval.tasks.flow_judge_evals.utils import format_vars

# Import metric functions relevant to the task
from lm_eval.tasks.flow_judge_evals.metrics import (
    pearson_corr,
    pearson_corr_agg,
    spearman_corr,
    spearman_corr_agg,
    kendalltau_corr,
    kendalltau_corr_agg
)

def format_orig_score(doc) -> str:
    score = doc["orig_score"]
    return str(score)

def format_prompt_prometheus(doc: Dict[str, Any]) -> str:
        prompt = doc["instruction"]
        return prompt
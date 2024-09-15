from datasets import Dataset
from typing import List, Dict, Any
from lm_eval.tasks.flow_judge_evals.utils import format_vars, format_rubric, format_prompt, format_target

# Import metric functions relevant to the task
from lm_eval.tasks.flow_judge_evals.metrics import (
    accuracy,
    accuracy_agg,
    f1_binary,
    f1_agg_binary,
    precision_binary,
    precision_agg_binary,
    recall_binary,
    recall_agg_binary,
    f1_macro_3likert,
    f1_agg_macro_3likert,
    f1_macro_5likert,
    f1_agg_macro_5likert,
    f1_micro_3likert,
    f1_agg_micro_3likert,
    f1_micro_5likert,
    f1_agg_micro_5likert,
    precision_macro_3likert,
    precision_agg_macro_3likert,
    precision_macro_5likert,
    precision_agg_macro_5likert,
    precision_micro_3likert,
    precision_agg_micro_3likert,
    precision_micro_5likert,
    precision_agg_micro_5likert,
    recall_macro_3likert,
    recall_agg_macro_3likert,
    recall_macro_5likert,
    recall_agg_macro_5likert,
    recall_micro_3likert,
    recall_agg_micro_3likert,
    recall_micro_5likert,
    recall_agg_micro_5likert,
    pearson_corr,
    pearson_corr_agg,
    spearman_corr,
    spearman_corr_agg,
    kendalltau_corr,
    kendalltau_corr_agg
)

def preprocess_records(dataset: Dataset) -> Dataset:
    def _helper(doc: Dict[str, Any]):
        # Remove the "type" key from inputs and output
        inputs = [{k: v for k, v in d.items() if k != "type"} for d in doc["inputs"]]
        output = {k: v for k, v in doc["output"].items() if k != "type"}
        
        prompt_variables = {
            "INPUTS": format_vars(inputs) if inputs else None,
            "OUTPUT": format_vars([output]),
            "EVALUATION_CRITERIA": doc["criteria"],
            "RUBRIC": format_rubric(doc["rubric"])
        }
        
        return prompt_variables
    
    return dataset.map(_helper)
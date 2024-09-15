from typing import Dict, Any, List
from datasets import Dataset
from lm_eval.tasks.flow_judge_evals.utils import format_vars, format_rubric, format_prompt

# Import metric functions relevant to the task
from lm_eval.tasks.flow_judge_evals.metrics import (
    pearson_corr,
    pearson_corr_agg,
    spearman_corr,
    spearman_corr_agg,
    kendalltau_corr,
    kendalltau_corr_agg
)

def create_rubric(record: Dict) -> List[Dict]:
    """Create a rubric from the record."""
    return [
        {"score": i, "description": record[f"orig_score{i}_description"]}
        for i in range(1, 6)
    ]

def preprocess_records(dataset: Dataset) -> Dataset:
    def _helper(doc: Dict[str, Any]):
        processed_record = {
            "id": doc["id"],
            "input": [{"name": "Instruction", "content": doc["orig_instruction"]}],
            "output": [{"name": "Response", "content": doc["orig_response"]}],
            "score": int(doc["orig_score"]),
            "rubric": create_rubric(doc),
            "criteria": doc["orig_criteria"]
        }
        prompt_variables = {
            "INPUTS": format_vars(processed_record["input"]),
            "OUTPUT": format_vars(processed_record["output"]),
            "EVALUATION_CRITERIA": processed_record["criteria"],
            "RUBRIC": format_rubric(processed_record["rubric"])
        }
        return prompt_variables
    
    return dataset.map(_helper)

def format_orig_score(doc) -> str:
    score = doc["orig_score"]
    return str(score)
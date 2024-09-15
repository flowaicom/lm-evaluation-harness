from typing import Dict, Any
from datasets import Dataset
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
    pearson_corr,
    pearson_corr_agg,
    spearman_corr,
    spearman_corr_agg,
    kendalltau_corr,
    kendalltau_corr_agg
)

EVALUATION_CRITERIA =  "Evaluate whether the information provided in the answer is factually accurate and directly supported by the context given in the document, without any fabricated or hallucinated details."

RUBRIC = [
    { 
        "score": 0, 
        "description": "The answer is not supported by the document. It contains inaccuracies, fabrications, or details that are not present in the document."
    },
    { 
        "score": 1, 
        "description": "The answer is fully supported by the document. It is factually accurate and all details are directly derived from the document."
    } 

]  

def preprocess_records(dataset: Dataset) -> Dataset:
    def _helper(doc: Dict[str, Any]):
        prompt_variables = {
            "INPUTS": format_vars([
                {
                    "name": "Passage",
                    "content": doc["passage"]
                },
                {
                    "name": "Question",
                    "content": doc["question"]
                }
            ]),
            "OUTPUT": format_vars([
                {
                    "name": "Answer",
                    "content": doc["answer"]
                }
            ]),
            "EVALUATION_CRITERIA": EVALUATION_CRITERIA,
            "RUBRIC": format_rubric(RUBRIC)
        }
        
        return prompt_variables
    
    return dataset.map(_helper)
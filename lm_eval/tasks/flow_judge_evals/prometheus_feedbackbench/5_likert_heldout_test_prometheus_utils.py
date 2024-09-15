from datasets import Dataset
from typing import List, Dict, Any
from lm_eval.tasks.flow_judge_evals.utils import format_vars, format_target

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


def format_rubric_criteria(criteria: str, rubric: List[Dict]) -> str:
    rubric_and_criteria = f"[{criteria}]\n"
    rubric_strs = []
    
    # reverse for binary for consistency
    if len(rubric) == 2:
        rubric = sorted(rubric, key=lambda x: x["score"], reverse=False)
    
    for score in rubric:
        score_value = score["score"]
        rubric_strs.append(f"- Score {score_value}: {score['description']}")
    rubric_and_criteria += "\n".join(rubric_strs)
    return rubric_and_criteria
   

def preprocess_records(dataset: Dataset) -> Dataset:
    def _helper(doc: Dict[str, Any]):
        # Remove the "type" key from inputs and output
        inputs = [{k: v for k, v in d.items() if k != "type"} for d in doc["inputs"]]
        output = {k: v for k, v in doc["output"].items() if k != "type"}
        
        prompt_variables = {
            "INPUTS": format_vars(inputs) if inputs else None,
            "OUTPUT": format_vars([output]),
            "CRITERIA_AND_RUBRIC": format_rubric_criteria(doc["criteria"], doc["rubric"])
            
        }
        
        return prompt_variables
    
    return dataset.map(_helper) 

# Source: https://github.com/prometheus-eval/prometheus-eval/blob/b7a431a553b320e0a7cc49c6c5d3c54b1b840d39/libs/prometheus-eval/prometheus_eval/prompts.py#L27
ABSOLUTE_PROMPT_WO_REF = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{INPUTS}

###Response to evaluate:
{OUTPUT}

###Score Rubrics:
{CRITERIA_AND_RUBRIC}

###Feedback: """

def format_prompt_prometheus(doc: Dict[str, Any]) -> str:
        return ABSOLUTE_PROMPT_WO_REF.format(**doc)

from typing import Dict, Any
from datasets import Dataset
from lm_eval.tasks.flow_judge_evals.utils import format_vars, format_rubric
import json


EVALUATION_CRITERIA = "Evaluate whether the information provided in the response is factually accurate and directly supported by the context given in the related passages." # manually defined

RUBRIC = [
    {
        "score": 0,
        "description": "The response contains information that is not supported by the passages, includes fabricated details, or misinterprets the information from the passages."
    },
    {
        "score": 1,
        "description": "The response is factually accurate and directly supported by the information provided in the passages, without any fabricated or hallucinated details."
    }
]

def preprocess_records(dataset: Dataset) -> Dataset:
    def _helper(doc: Dict[str, Any]):
        source_info = json.loads(doc['source_info'])
        prompt_variables = {
            "INPUTS": format_vars([
                {
                    "name": "passages",
                    "content": source_info["passages"]
                },
                {
                    "name": "question",
                    "content": source_info["question"]
                }
            ]),
            "OUTPUT": format_vars([
                {
                    "name": "answer",
                    "content": doc["response"]
                }
            ]),
            "EVALUATION_CRITERIA": EVALUATION_CRITERIA,
            "RUBRIC": format_rubric(RUBRIC)
        }
        return prompt_variables
    
    return dataset.map(_helper)
from typing import Dict, Any
from datasets import Dataset
from lm_eval.tasks.flow_judge_evals.utils import format_vars, format_rubric

EVALUATION_CRITERIA = """Based on the provided JSON file about a local business, does the overview only contain information that is supported by or directly inferable from the JSON file?"""

RUBRIC = [
     {
        "score": 0,
        "description": "The overview contains statements or claims that cannot be directly found in or logically inferred from the provided context. There is hallucinated or fabricated information present in the response that does not have support in the given context."
    },
    {
        "score": 1,
        "description": "The overview contains only statements and claims that are directly stated in or logically inferable from the provided context. There is no hallucinated or fabricated information present in the response that cannot be traced back to or deduced from the context."

    }
] 

def preprocess_records(dataset: Dataset) -> Dataset:
    def _helper(doc: Dict[str, Any]):
        prompt_variables = {
            "INPUTS": format_vars([
                {
                    "name": "JSON file",
                    "content": doc["source_info"]
                }
            ]),
            "OUTPUT": format_vars([
                {
                    "name": "Overview",
                    "content": doc["response"]
                }
            ]),
            "EVALUATION_CRITERIA": EVALUATION_CRITERIA,
            "RUBRIC": format_rubric(RUBRIC)
        }
        return prompt_variables
    
    return dataset.map(_helper)
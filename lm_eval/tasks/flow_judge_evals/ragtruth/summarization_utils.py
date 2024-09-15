from typing import Dict, Any
from datasets import Dataset
from lm_eval.tasks.flow_judge_evals.utils import format_vars, format_rubric
import ast

EVALUATION_CRITERIA = """Based on the provided news article does the summary contain only information that is supported by or directly inferable from the news article?"""

RUBRIC = [
    {
        "score": 0,
        "description": "The summary contains statements or claims that cannot be directly found in or logically inferred from the provided context. There is hallucinated or fabricated information present in the summary that does not have support in the given context."
    },
    {
        "score": 1,
        "description": "The summary contains only statements and claims that are directly stated in or logically inferable from the provided context. There is no hallucinated or fabricated information present in the summary that cannot be traced back to or deduced from the context."
    }
] 


def format_source_string(input_string: str) -> str:
    try:
        # Use ast.literal_eval to safely evaluate the string
        evaluated_string = ast.literal_eval(input_string)
        
        # If the evaluated result is a string, return it
        if isinstance(evaluated_string, str):
            return evaluated_string
        else:
            raise ValueError("The input is not a valid string representation")
    except (SyntaxError, ValueError) as e:
        # If there's an error, return the original string
        print(f"Error processing string: {e}")
        return input_string

def preprocess_records(dataset: Dataset) -> Dataset:
    def _helper(doc: Dict[str, Any]):
        prompt_variables = {
            "INPUTS": format_vars([
                {
                    "name": "News article",
                    "content": format_source_string(doc["source_info"])
                }
            ]),
            "OUTPUT": format_vars([
                {
                    "name": "Summary",
                    "content": doc["response"]
                }
            ]),
            "EVALUATION_CRITERIA": EVALUATION_CRITERIA,
            "RUBRIC": format_rubric(RUBRIC)
        }
        return prompt_variables
    
    return dataset.map(_helper)
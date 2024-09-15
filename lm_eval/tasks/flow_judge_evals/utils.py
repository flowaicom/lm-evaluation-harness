from typing import List, Dict, Any
from lm_eval.tasks.flow_judge_evals.prompts import USER_PROMPT_NO_INPUTS_TEMPLATE, USER_PROMPT_TEMPLATE

def format_vars(variables: List[Dict]) -> str:
    """Format variables into XML-like tags with content."""
    var_strs = []
    for var in variables:
        var_tag = var["name"].lower().replace(" ", "_")
        var_strs.append(f"<{var_tag}>\n{var['content']}\n</{var_tag}>")
    return "\n".join(var_strs)

def format_rubric(rubric: List[Dict]) -> str:
    """Format rubric items into a numbered list."""
    rubric_strs = []
    
    # reverse for binary for consistency
    if len(rubric) == 2:
        rubric = sorted(rubric, key=lambda x: x["score"], reverse=False)
    
    for score in rubric:
        score_value = score["score"]
        rubric_strs.append(f"- Score {score_value}: {score['description']}")
    return "\n".join(rubric_strs)

def format_prompt(doc: Dict[str, Any]) -> str:
    """Format the flow judge prompt based on whether inputs are present."""
    if doc["INPUTS"]:
        return USER_PROMPT_TEMPLATE.format(**doc)
    else:
        return USER_PROMPT_NO_INPUTS_TEMPLATE.format(**doc)

def format_target(doc) -> str:
    """Extract and return the score as a string from the document."""
    score = doc["score"]
    return str(score)
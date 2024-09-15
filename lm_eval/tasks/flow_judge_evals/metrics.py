from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging

def filter_and_log(items, metric_name):
    """This function filters out items with empty predictions or references and logs the number of ignored items.
    
    If there is an important number of ignored items (more than 20%), the function will return None for the metric value."""
    filtered_items = [(pred, ref) for pred, ref in items if pred != '' and ref != '']
    predictions, references = zip(*filtered_items) if filtered_items else ([], [])
    
    total_items = len(items)
    ignored_items = total_items - len(filtered_items)
    
    logging.info(f"Ignored {ignored_items} out of {total_items} predictions due to an output parsing error.")
    
    if ignored_items > 0.2 * total_items:
        logging.warning(f"More than 20% of items ignored. Returning None for {metric_name}.")
        return None, None
    
    if not predictions:
        logging.warning(f"No valid predictions after filtering. Returning None for {metric_name}.")
        return None, None
    
    return predictions, references

# Note function names are used as keys in the metric dictionary
def accuracy(predictions, references):
    return (predictions[0], references[0])

def accuracy_agg(items):
    predictions, references = filter_and_log(items, "accuracy score")
    return 0.0 if predictions is None else accuracy_score(references, predictions)

# F1 Score
def f1_binary(predictions, references):
    return (predictions[0], references[0])

def f1_agg_binary(items):
    predictions, references = filter_and_log(items, "F1 score (binary)")
    return 0.0 if predictions is None else f1_score(references, predictions, average='binary', labels=[0, 1])

def f1_macro_3likert(predictions, references):
    return (predictions[0], references[0])

def f1_agg_macro_3likert(items):
    predictions, references = filter_and_log(items, "F1 score - macro")
    return 0.0 if predictions is None else f1_score(references, predictions, average='macro', labels=[1, 2, 3])

def f1_macro_5likert(predictions, references):
    return (predictions[0], references[0])

def f1_agg_macro_5likert(items):
    predictions, references = filter_and_log(items, "F1 score - macro")
    return 0.0 if predictions is None else f1_score(references, predictions, average='macro', labels=[1, 2, 3, 4, 5])

def f1_micro_3likert(predictions, references):
    return (predictions[0], references[0])

def f1_agg_micro_3likert(items):
    predictions, references = filter_and_log(items, "F1 score - micro")
    return 0.0 if predictions is None else f1_score(references, predictions, average='micro', labels=[1, 2, 3])

def f1_micro_5likert(predictions, references):
    return (predictions[0], references[0])

def f1_agg_micro_5likert(items):
    predictions, references = filter_and_log(items, "F1 score - micro")
    return 0.0 if predictions is None else f1_score(references, predictions, average='micro', labels=[1, 2, 3, 4, 5])

# Precision
def precision_binary(predictions, references):
    return (predictions[0], references[0])

def precision_agg_binary(items):
    predictions, references = filter_and_log(items, "Precision score (binary)")
    return 0.0 if predictions is None else precision_score(references, predictions, average='binary', labels=[0, 1])

def precision_macro_3likert(predictions, references):
    return (predictions[0], references[0])

def precision_agg_macro_3likert(items):
    predictions, references = filter_and_log(items, "Precision score - macro")
    return 0.0 if predictions is None else precision_score(references, predictions, average='macro', labels=[1, 2, 3])

def precision_macro_5likert(predictions, references):
    return (predictions[0], references[0])

def precision_agg_macro_5likert(items):
    predictions, references = filter_and_log(items, "Precision score - macro")
    return 0.0 if predictions is None else precision_score(references, predictions, average='macro', labels=[1, 2, 3, 4, 5])

def precision_micro_3likert(predictions, references):
    return (predictions[0], references[0])

def precision_agg_micro_3likert(items):
    predictions, references = filter_and_log(items, "Precision score - micro")
    return 0.0 if predictions is None else precision_score(references, predictions, average='micro', labels=[1, 2, 3])

def precision_micro_5likert(predictions, references):
    return (predictions[0], references[0])

def precision_agg_micro_5likert(items):
    predictions, references = filter_and_log(items, "Precision score - micro")
    return 0.0 if predictions is None else precision_score(references, predictions, average='micro', labels=[1, 2, 3, 4, 5])

# Recall
def recall_binary(predictions, references):
    return (predictions[0], references[0])

def recall_agg_binary(items):
    predictions, references = filter_and_log(items, "Recall score (binary)")
    return 0.0 if predictions is None else recall_score(references, predictions, average='binary', labels=[0, 1])

def recall_macro_3likert(predictions, references):
    return (predictions[0], references[0])

def recall_agg_macro_3likert(items):
    predictions, references = filter_and_log(items, "Recall score - macro")
    return 0.0 if predictions is None else recall_score(references, predictions, average='macro', labels=[1, 2, 3])

def recall_macro_5likert(predictions, references):
    return (predictions[0], references[0])

def recall_agg_macro_5likert(items):
    predictions, references = filter_and_log(items, "Recall score - macro")
    return 0.0 if predictions is None else recall_score(references, predictions, average='macro', labels=[1, 2, 3, 4, 5])

def recall_micro_3likert(predictions, references):
    return (predictions[0], references[0])

def recall_agg_micro_3likert(items):
    predictions, references = filter_and_log(items, "Recall score - micro")
    return 0.0 if predictions is None else recall_score(references, predictions, average='micro', labels=[1, 2, 3])

def recall_micro_5likert(predictions, references):
    return (predictions[0], references[0])

def recall_agg_micro_5likert(items):
    predictions, references = filter_and_log(items, "Recall score - micro")
    return 0.0 if predictions is None else recall_score(references, predictions, average='micro', labels=[1, 2, 3, 4, 5])

# Correlations
def pearson_corr(predictions, references):
    return (predictions[0], references[0])

def pearson_corr_agg(items):
    predictions, references = filter_and_log(items, "Pearson correlation")
    return 0.0 if predictions is None else pearsonr(references, predictions)[0]

def spearman_corr(predictions, references):
    return (predictions[0], references[0])

def spearman_corr_agg(items):
    predictions, references = filter_and_log(items, "Spearman correlation")
    return 0.0 if predictions is None else spearmanr(references, predictions)[0]

def kendalltau_corr(predictions, references):
    return (predictions[0], references[0])

def kendalltau_corr_agg(items):
    predictions, references = filter_and_log(items, "Kendall's Tau correlation")
    return 0.0 if predictions is None else kendalltau(references, predictions)[0]

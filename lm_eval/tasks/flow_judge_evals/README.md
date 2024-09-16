# Instructions for Flow-Judge-v0.1 Evaluations

# Install dependencies

```bash
pip install -e ".[vllm,hf_transfer,api]
```

# Flow-Judge-v0.1 evaluation tasks

| Task Name | Type | Description |
|-----------|------|-------------|
| `binary_heldout_test` | Pass/Fail | Evaluation on held-out test data, which contains evaluations spanning different domains and metrics |
| `qa_ragtruth` | Pass/Fail | Evaluation of hallucination detection on the Question-Answering subset of RAGTruth |
| `data2txt_ragtruth` | Pass/Fail | Evaluation of hallucination detection on the Data-to-text subset of RAGTruth dataset |
| `summarization_ragtruth` | Pass/Fail | Evaluation of hallucination detection on the Summarization subset of RAGTruth |
| `halueval_halubench` | Pass/Fail | Evaluation of hallucination detection on HaluEval subset of HaluBench |
| `covid_qa_halubench`| Pass/Fail | Evaluation of hallucination detection on the COVID-19 Question-Answering subset of HaluBench |
| `pubmedqa_halubench` | Pass/Fail | Evaluation of hallucination detection on the PubMedQA Question-Answering subset of HaluBench |
| `3_likert_heldout_test` | 3-Likert | Evaluation on held-out test data, which contains evaluations spanning different domains and metrics |
| `5_likert_heldout_test` | 5-Likert | Evaluation on held-out test data, which contains evaluations spanning different domains and metrics |
| `feedbackbench` | 5-Likert | Evaluation on FeedbackBench, which contains evaluations spanning different domains and metrics |
| `feedbackbench_prometheus` | 5-Likert | Prometheus 2 - FeedbackBench evaluation |
| `5_likert_heldout_test_prometheus` | 5-Likert | Prometheus 2 - Evaluation on held-out test data |


# How to reproduce evaluations

Example:

```bash
./lm_eval/tasks/flow_judge_evals/run_vllm_eval.sh binary_heldout_test,covid_qa_halubench flowaicom/Flow-Judge-v0.1
```

## vLLM

Choose tasks and a model. Then run the following command:

```bash
./lm_eval/tasks/flow_judge_evals/run_vllm_eval.sh <task_names> <model_path_or_name>
```

Note. You need a cuda-enabled GPU to run this command. You can choose which GPU to use by setting the `CUDA_VISIBLE_DEVICES` environment variable. You should also set the number of threads to use for the model with `OMP_NUM_THREADS`.

## OpenAI Chat Completions
Choose tasks and a model. Then run the following command:

```bash
./lm_eval/tasks/flow_judge_evals/run_openai_eval.sh <task_names> <model_path_or_name>
```

Note. You need to provide your own OpenAI API key to run this command. You can set it in the `export OPENAI_API_KEY="sk-..."` line at the top of the script.
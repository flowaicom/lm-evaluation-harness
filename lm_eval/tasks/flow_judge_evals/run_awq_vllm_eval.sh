#!/usr/bin/env bash

# Eval tasks
TASKS="$1"  # Take from command line, no default

# Configuration and Model arguments
PRETRAINED="$2"  # Take from command line, no default
TOKENIZER="$PRETRAINED"
DTYPE="bfloat16"
DISABLE_SLIDING_WINDOW=True
GPU_MEMORY_UTILIZATION=0.90
ENFORCE_EAGER=True
TRUST_REMOTE_CODE=False
MAX_MODEL_LEN=8192
MAX_GEN_TOKS=2048
MAX_NUM_SEQS=256
QUANTINZATION="awq_marlin"

MODEL_ARGS=(
    "pretrained=$PRETRAINED"
    "tokenizer=$TOKENIZER"
    "dtype=$DTYPE"
    "disable_sliding_window=$DISABLE_SLIDING_WINDOW"
    "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    "enforce_eager=$ENFORCE_EAGER"
    "trust_remote_code=$TRUST_REMOTE_CODE"
    "max_model_len=$MAX_MODEL_LEN"
    "max_gen_toks=$MAX_GEN_TOKS"
    "max_num_seqs=$MAX_NUM_SEQS"
    "quantization=$QUANTINZATION"
)

# Generation config
TEMPERATURE=0.1
TOP_P=0.95
DO_SAMPLE=True
UNTIL="<|endoftext|>"
SEED=42

GEN_KWARGS=(
    "temperature=$TEMPERATURE"
    "top_p=$TOP_P"
    "do_sample=$DO_SAMPLE"
    "until=$UNTIL"
    "seed=$SEED"
)

# Environment variables
export CUDA_VISIBLE_DEVICES="0"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export OMP_NUM_THREADS=32 # set to number of cores

# Run lm_eval
USE_FLASH_ATTENTION=1 python -m lm_eval \
    --model vllm \
    --model_args $(IFS=,; echo "${MODEL_ARGS[*]}") \
    --tasks $TASKS \
    --apply_chat_template \
    --log_samples \
    --batch_size auto \
    --max_batch_size 40 \
    --gen_kwargs $(IFS=,; echo "${GEN_KWARGS[*]}") \
    --output_path output
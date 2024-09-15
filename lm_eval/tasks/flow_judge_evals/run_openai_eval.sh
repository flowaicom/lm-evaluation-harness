#!/usr/bin/env bash

# Eval tasks
TASKS="$1"  # Take from command line, no default

# Configuration and Model arguments
MODEL_ID="$2"  # Take from command line, no default
NUM_CONCURRENT_REQUESTS=10


MODEL_ARGS=(
    "model=$MODEL_ID"
    "num_concurrent=$NUM_CONCURRENT_REQUESTS"
)

# Generation config
TEMPERATURE=0.0
TOP_P=1.0
PRESENCE_PENALTY=0.0
FREQUENCY_PENALTY=0.0
MAX_NEW_TOKENS=2048
STOP=null

GEN_KWARGS=(
    "temperature=$TEMPERATURE"
    "top_p=$TOP_P"
    "max_tokens=$MAX_NEW_TOKENS"
    "presence_penalty=$PRESENCE_PENALTY"
    "frequency_penalty=$FREQUENCY_PENALTY"
    "stop=$STOP"
)

# Environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set."
    exit 1
fi

# Run lm_eval
python -m lm_eval \
    --model openai-chat-completions \
    --model_args "$(IFS=,; echo "${MODEL_ARGS[*]}")" \
    --tasks "$TASKS" \
    --log_samples \
    --apply_chat_template \
    --gen_kwargs "$(IFS=,; echo "${GEN_KWARGS[*]}")" \
    --output_path output
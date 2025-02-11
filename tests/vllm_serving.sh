#!/bin/bash

# Run this script in one tab first, then run the script calling the method in another tab
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export RAY_memory_monitor_refresh_ms=0;
server_type=vllm.entrypoints.openai.api_server
model_name="/mnt/e/projects/vllm/checkpoints/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
download_dir=/mnt/e/projects/vllm/checkpoints
num_GPUs=1

python -m $server_type \
    --model $model_name \
    --tokenizer $model_name \
    --gpu-memory-utilization=0.9 \
    --max-num-seqs=1200 \
    --disable-log-requests \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size $num_GPUs \
    --download-dir $download_dir

#!/bin/bash

model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
cache_dir=/root/.cache/huggingface:/root/.cache/huggingface

# clean-up after use
sudo docker rm my_vllm_container

sudo docker run --runtime nvidia --gpus all \
	--name my_vllm_container \
	-v $cache_dir \
 	--env "HUGGING_FACE_HUB_TOKEN={your_hf_token}" \
	-p 8000:8000 \
	--ipc=host \
	vllm/vllm-openai:latest \
	--model $model_name \
    --tokenizer $tokenizer \
    --dtype float16 \
	--max_model_len 32000 \
    --disable-log-requests
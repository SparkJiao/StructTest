#!/bin/bash
cache_dir=~/.cache/huggingface:/root/.cache/huggingface

while getopts ':m:h' opt; do
  case "$opt" in
    m)
      model_name=$OPTARG
      echo "Setting model name to $model_name"
      ;;

    ?|h)
      echo "Usage: $(basename $0) [-m arg]"
      exit 1
      ;;
  esac
done

# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NGPU=$(nvidia-smi --list-gpus | wc -l)

export HF_TOKEN={your_hf_keys}


model_full_name=$(cat config.json | jq -r .'model_configs.'\"$model_name\"'.model_name')
# model_full_name=$model_name
echo "model full name: ""$model_full_name";


vllm serve $model_full_name \
    --dtype float16 \
    --tensor_parallel_size ${NGPU} \
    --disable-log-requests

#!/bin/sh

timeout=30 # default to 30s timeout
while getopts ':n:m:t:h' opt; do
  case "$opt" in
    n)
      num_proc=$OPTARG
      echo "Setting num_proc to $num_proc"
      ;;

    m)
      model_name=$OPTARG
      echo "Setting model name to $model_name"
      ;;

    t)
      timeout=$OPTARG
      echo "Setting timeout to $timeout"
      ;;

    ?|h)
      echo "Usage: $(basename $0) [-n arg] [-m arg]"
      exit 1
      ;;
  esac
done

subtask="nested_html_encoding"
 
echo $subtask
python -m src.pipeline.run_parallel --model_name $model_name --task html_encoding --subtask $subtask \
--num_proc $num_proc \
--timeout $timeout \
--load_format_properties_from_local_disk \
--few_shot \
--extra_mode random_easy

python -m src.pipeline.run_parallel --model_name $model_name --task html_encoding --subtask $subtask \
--num_proc $num_proc  \
--timeout $timeout \
--load_format_properties_from_local_disk \
--few_shot \
--extra_mode random_hard \

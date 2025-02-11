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


for subtask in "length" "bullet_points" "numbered_points" "questions" "bullet_points_length" "numbered_points_length" "indented_bullet_points";
   do
      echo $subtask, $model_name
      python -m src.pipeline.run_parallel --model_name $model_name --task summarization --subtask $subtask \
      --num_proc $num_proc --timeout $timeout \
      --load_format_properties_from_local_disk \
      
   done

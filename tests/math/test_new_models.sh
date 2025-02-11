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


for subtask in "suffix_phrase" "bullet_points";
do
      echo $subtask, $model_name
      python tests/math/math_follow.py --model_name $model_name --math_eval_type $subtask \
      --num_proc $num_proc --timeout $timeout
done

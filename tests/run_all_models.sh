#!/bin/bash
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
      echo "Usage: $(basename $0) [-n arg] [-m arg] [-t arg]"
      exit 1
      ;;
  esac
done

# summarization
sh tests/summarization/test_new_models.sh -m $model_name -n $num_proc -t $timeout
# code
sh tests/code/test_new_models.sh -m $model_name -n $num_proc -t $timeout
# html
sh tests/html/test_new_models.sh -m $model_name -n $num_proc -t $timeout
# math
sh tests/math/test_new_models.sh -m $model_name -n $num_proc -t $timeout

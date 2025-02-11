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

for subtask in "add_print_statements" "replace_variables" "test_case_inputs_gen" "add_print_statements_v2" "replace_variables_v2" "test_case_inputs_gen_v2" "simulate_execute_v2";
do
   echo $subtask
   python -m src.pipeline.run_parallel --model_name $model_name --task code --subtask $subtask --num_proc $num_proc --timeout $timeout
done

for subtask in "simulate_execute";
do
   echo $subtask
   python -m src.pipeline.run_parallel --model_name $model_name --task code --subtask $subtask --num_proc $num_proc --timeout $timeout
   echo "For simulate execute, please ignore the micro accuracy above and use macro accuracy below:"
   python tests/code/simulate_exec_macro_acc.py --model_name $model_name
done

import argparse
from collections import defaultdict
import json
import csv


def calculate_macro_accuracy(data):
    """
    Calculate the macro accuracy based on merged problem_id correctness.

    Args:
        data (list): A list of dictionaries containing `id` and `result` fields.
                     The `id` field is in the format `{problem_id}_{test_id}`.
                     The `result` field is a boolean indicating correctness.

    Returns:
        float: The macro accuracy of the results.
    """

    # Step 1: Group results by problem_id
    grouped_results = defaultdict(list)
    for item in data:
        problem_id = item['index'].split('_')[0]
        grouped_results[problem_id].append(item['result'])

    # Step 2: Evaluate correctness for each problem_id
    problem_correctness = []
    for problem_id, results in grouped_results.items():
        # A problem is correct only if all its test results are True
        problem_correctness.append(all(results))

    # Step 3: Calculate macro accuracy
    macro_accuracy = sum(problem_correctness) / len(problem_correctness) if problem_correctness else 0.0

    return macro_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()
    clean_model_name = args.model_name.split("/")[-1]
    args.input_file = f"outputs/output_{clean_model_name}_code_simulate_execute.csv" 

    all_data = []
    with open(args.input_file, mode='r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            row["result"] = eval(row["result"])
            all_data.append(row)

    macro_accuracy = calculate_macro_accuracy(all_data)
    print(f"Macro accuracy (%): {macro_accuracy*100:.4f}")


if __name__ == "__main__":
    main()
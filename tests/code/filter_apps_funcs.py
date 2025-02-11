from datasets import load_dataset
import argparse
import sys
import json
from tqdm import tqdm
import random


# sys.set_int_max_str_digits(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--seed", type=str, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    data = load_dataset("codeparrot/apps", split="test").to_list()

    outputs = []
    is_fn = 0
    for item in tqdm(data):
        if item["solutions"]:
            item["solutions"] = json.loads(item["solutions"])

        if item["input_output"]:
            item["input_output"] = json.loads(item["input_output"])

            if "fn_name" in item["input_output"]:
                outputs.append(item)

    print(len(outputs))
    print(is_fn)

    if args.output_file:
        json.dump(outputs, open(args.output_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()

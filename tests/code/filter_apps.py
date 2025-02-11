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
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--seed", type=str, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    data = load_dataset("codeparrot/apps", split="test").to_list()

    outputs = []
    is_fn = 0
    for item in tqdm(data):
        if item["solutions"]:
            item["solutions"] = json.loads(item["solutions"])
            sol_lens = []
            for sol in item["solutions"]:
                lines = sol.split("\n")
                lines = [line for line in lines if len(line.strip())]
                sol_lens.append(len(lines))
        else:
            sol_lens = [0]
        item["sol_lens"] = sol_lens

        tmp = 0
        code_candidates = []
        for i, _len in enumerate(sol_lens):
            if 50 <= _len <= 200:
                tmp += 1
                code_candidates.append(item["solutions"][i])

        if item["input_output"]:
            item["input_output"] = json.loads(item["input_output"])

        if tmp and item["input_output"]:
            item["code"] = random.choice(code_candidates)
            outputs.append(item)
            if "fn_name" in item["input_output"]:
                is_fn += 1

    print(len(outputs))
    print(is_fn)

    new_outputs = random.sample(outputs, int(args.num_samples))

    if args.output_file:
        json.dump(new_outputs, open(args.output_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()

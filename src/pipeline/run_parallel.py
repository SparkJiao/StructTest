import argparse
import os.path

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, IterableDatasetDict, IterableDataset
from functools import partial
from multiprocessing import Pool
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resources.utils import seed_everything, map_task_to_dataset_name, map_task_to_tags, sample_format, load_format_properties_from_local_disk, \
    check_exists_or_quit
from pipeline.prompting import prepare_prompt
from pipeline.online_inference import OnlineGenerator
from pipeline.evaluation import evaluate_generation
from dataset.apps import load_code_dataset
from pathlib import Path


def process_example(example, idx, args, model_name, dataset_name):
    # Find corresponding output tags for the given task
    start_tag, end_tag = map_task_to_tags(args.task, args.subtask)
    args.start_tag = start_tag
    args.end_tag = end_tag
    if args.task == "html_encoding":
        start_tag, end_tag = "<html>", "</html>"
    else:
        start_tag, end_tag = args.start_tag, args.end_tag
    # if the example is already properly processed
    if example.get("completed") is not None: #example.get("completed") is True:
        if args.re_eval:
            res = evaluate_generation(
                subtask=args.subtask,
                generated=example['generated'],
                start_tag=start_tag,
                end_tag=end_tag,
                **example.get('format_properties', {}),
                **example.get('meta_data', {}),
            )  # 0 or 1
            example['result'] = res
        return {'idx': example['index'], 'result': example['result'], 'generated': example['generated'],
                'format_properties': example.get('format_properties', {})}

    generator = OnlineGenerator(model_name=model_name, dataset=dataset_name, task=args.task)

    document = example["document"]  # TODO: Can we specify the inputs according to different task/dataset?
    if 'format_properties' in example:  # use existing format properties to reduce randomness in sampling
        format_properties = example['format_properties']
    else:
        if args.task not in ['code']:
            print("No format properties found")
        format_properties = sample_format(args.subtask, args.extra_mode)

    # `meta_data` is used to contain some extra inputs for evaluation. Maybe we can combine it with `document` or `format_properties`?
    if 'meta_data' in example:
        meta_data = example['meta_data']
    else:
        meta_data = {}

    prompt = prepare_prompt(
        task=args.task,
        subtask=args.subtask,
        document=document,
        generator=generator,
        no_extra_comment_prompt=args.no_extra_comment,
        few_shot=args.few_shot,
        surrounding_tags=args.surrounding_tags,
        start_tag=args.start_tag,
        end_tag=args.end_tag,
        **format_properties,
        **meta_data,
    )

    prompt = [{"role": "user", "content": prompt}]
    # print(f"format properties: {format_properties}, prompt: {prompt}")
    generated = generator.generate(prompt, temperature=args.temperature, timeout=args.timeout)
    # Evaluate the generated summary
    try:
        res = evaluate_generation(
            subtask=args.subtask,
            generated=generated,
            start_tag=start_tag,
            end_tag=end_tag,
            **format_properties,
            **meta_data,
        )  # 0 or 1
    except Exception as e:
        print("Exception in evaluate_generation:")
        print(e)
        res = 0

    return {'idx': idx, 'result': res, 'generated': generated, 'format_properties': format_properties}


def vanilla_mp_process_proxy(example, args, model_name, dataset_name):
    example, idx = example
    if "id" in example:
        idx = example["id"]
    return process_example(example, idx, args, model_name, dataset_name)


def main():
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument(
        "--seed", type=int, default=42, help="The random seed to fix"
    )
    parser.add_argument(
        "--debug_size", type=int, default=-1, help="If set to >=1, controls the max number of data points."
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="The model to be used"
    )
    parser.add_argument(
        "--task", type=str, required=True, help="The task to be executed",
        choices=["summarization", "html_encoding", "code"]
    )
    parser.add_argument(
        "--subtask", type=str, required=True, help="The subtask to be executed",
        choices=["length", "bullet_points", "numbered_points", "questions",
                 "bullet_points_length", "numbered_points_length", "indented_bullet_points",
                 "easy_html_encoding", "nested_html_encoding",
                 "add_print_statements", "add_docstring", "replace_variables", "test_case_inputs_gen", "simulate_execute",
                 "add_print_statements_v2", "add_docstring_v2", "test_case_inputs_gen_v2", "simulate_execute_v2", "replace_variables_v2",
                 "simulate_execute_obfuscation_v1", "simulate_execute_obfuscation_v2"]
    )
    parser.add_argument(
        "--extra_mode", type=str, default="", help="Any extra mode, depending on the sub-task"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.01, help="The sampling temperature during generation"
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="timeout in seconds for vllm_post request or openai API"
    )
    parser.add_argument(
        "--num_proc", type=int, default=1, help="number of parallel processes"
    )
    parser.add_argument(
        "--no_extra_comment", action="store_true", help="Sentence in the prompt to prevent extra text generation."
    )
    parser.add_argument(
        "--re_eval", action="store_true", help="Force re-evaluation on generated samples, do not re-generated."
    )
    parser.add_argument(
        "--surrounding_tags", action="store_true", help="Start and end tags for prompt generation."
    )
    parser.add_argument(
        "--few_shot", action="store_true", help="Few shot examples in the prompt."
    )
    parser.add_argument(
        "--load_format_properties_from_local_disk", action="store_true", help="load dataset from local disk, this "
                                                                              "ensures the format options per data "
                                                                              "instance is stable"
    )
    parser.add_argument("--batch_size", help="The batch size for processing and caching.", type=int, default=64)

    args = parser.parse_args()
    clean_model_name = args.model_name.split("/")[-1]
    if args.extra_mode:
        output_path = f"outputs/output_{clean_model_name}_{args.task}_{args.subtask}_{args.extra_mode}.csv"
        cache_path = f"outputs/output_{clean_model_name}_{args.task}_{args.subtask}_{args.extra_mode}.jsonl"
    else:
        output_path = f"outputs/output_{clean_model_name}_{args.task}_{args.subtask}.csv"
        cache_path = f"outputs/output_{clean_model_name}_{args.task}_{args.subtask}.jsonl"

    if args.few_shot:
        base = os.path.splitext(output_path)[0]
        output_path = base + '_fewshot.csv'
        cache_path = base + "_fewshot.jsonl"

    if not args.re_eval:
        check_exists_or_quit(output_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)  # Assuming seed is set globally or with a default value
    # TODO: Is there any elegant way to unify the dataset loading procedure?
    if args.task in ["summarization", "html_encoding"]:
        dataset_name = map_task_to_dataset_name(args.task)
        dataset = load_dataset(dataset_name, split="train")
        if args.load_format_properties_from_local_disk:
            dataset = load_format_properties_from_local_disk(args.subtask, args.extra_mode, dataset_name, dataset)
    elif args.task == "code":
        dataset_name = map_task_to_dataset_name(args.task)
        dataset = load_code_dataset(args.subtask)
    else:
        raise ValueError("Unsupported task name. Please specify a supported task.")

    # TODO: change this temporary solution --> use a new dataset for html encoding
    if args.task == 'html_encoding':
        dataset = dataset.select(range(300))
        print(f"loaded dataset: {dataset}, {dataset.features}") 
    
    n_points = len(dataset)
    if args.debug_size >= 1:
        n_points = min(n_points, args.debug_size)
        if any(isinstance(dataset, x) for x in [DatasetDict, Dataset, IterableDatasetDict, IterableDataset]):
            dataset = dataset.select(range(n_points))
        else:
            dataset = dataset[:n_points]

    # if exists, mark completed rows <--> incomplete row is with generated as `ERROR` 
    flag_load_from_eval_file = False
    if Path(output_path).exists():
        print(f"Loading existing eval file: {output_path}")
        flag_load_from_eval_file = True
        df = pd.read_csv(output_path).sort_values(by='index')
        if 'generation' in df:
            df = df.rename(columns={"generation": "generated"})
        for key in ("index", "generated", "result", "completed"):
            if key == 'completed':
                new_column = [True if v != 'ERROR' else False for v in df['generated'].to_list()]
            else:
                new_column = df[key].to_list()
            if type(dataset) == list:
                for example, col in zip(dataset, new_column):
                    example[key] = col
            else:
                dataset = dataset.add_column(key, new_column)

    if any(isinstance(dataset, x) for x in [DatasetDict, Dataset, IterableDatasetDict, IterableDataset, list]):
        dataset = [(example, idx) for idx, example in tqdm(enumerate(dataset), desc="Indexing dataset")]

    _annotate = partial(vanilla_mp_process_proxy, args=args, dataset_name=dataset_name, model_name=args.model_name)
    
    processed_dataset = []
    # try to load from cached file when csv output is not available
    if (not flag_load_from_eval_file) and os.path.exists(cache_path):
        cache_records = [json.loads(line) for line in open(cache_path, "r").readlines()]
        processed_dataset.extend(cache_records)
        cached_ids = set([record['idx'] for record in cache_records])
        new_input_dataset = []
        for example in dataset:
            exp, idx = example
            if "id" in exp:
                idx = exp["id"]
            if idx not in cached_ids:
                new_input_dataset.append(example)
        dataset = new_input_dataset
        print(f"Loaded {len(cached_ids)} cached records from {cache_path}")

    pbar = tqdm(total=len(dataset), desc="Processing dataset")
    if os.path.exists(cache_path):
        _writer = open(cache_path, "a")
    else:
        _writer = open(cache_path, "w")

    for i in range(0, len(dataset), args.batch_size):
        batch = dataset[i:i + args.batch_size]
        with ThreadPoolExecutor(max_workers=args.num_proc) as executor:
            futures = []
            for example in batch:
                future = executor.submit(_annotate, example)
                futures.append(future)

            for future in as_completed(futures):
                _record = future.result()
                if _record['generated'] != "ERROR": # write valid results to cache
                    _writer.write(json.dumps(_record) + "\n")
                processed_dataset.append(_record)
                pbar.update()

    processed_dataset = {
        'idx': [x['idx'] for x in processed_dataset],
        'generated': [x['generated'] for x in processed_dataset],
        'result': [x['result'] for x in processed_dataset],
        'format_properties': [x['format_properties'] for x in processed_dataset]
    }

    result = processed_dataset['result']
    result = [res if res == 1 else 0 for res in result]

    generated = processed_dataset['generated']
    format_properties = processed_dataset['format_properties']
    index = processed_dataset['idx']

    smoothed_result = []
    for item in result:
        if item < 0:  # cases when we do not have any tags generated
            smoothed_result.append(0)
        else:
            smoothed_result.append(item)

    # Calculate average evaluation score
    avg_acc = 100 * sum(smoothed_result) / len(smoothed_result)
    print(f"Average Accuracy (%): {avg_acc}, # {len(smoothed_result)}")

    df = pd.DataFrame({"index": index, "generated": generated, "result": result, 'format_properties': format_properties})
    df = df.sort_values(by='index')
    print(f"Exporting results to: {output_path}")
    df.to_csv(output_path, index=False)
    err_size = len(df[df['generated'] == 'ERROR'])
    print(f"Error generation: {err_size} / {len(df)}")

    with open(output_path.replace(".csv", ".metrics.txt"), "w") as f:
        f.write(f"Average Accuracy (%): {avg_acc}\n")


if __name__ == "__main__":
    main()

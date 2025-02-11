import copy
import json
import sys
import logging

from datasets import load_dataset
from datasets.arrow_dataset import Dataset

# sys.set_int_max_str_digits(0)

logger = logging.getLogger(__name__)


class APPsReader:
    def __init__(self, split: str = "train", use_starter_code: bool = False):
        # self.train_sub_val_ids = set(json.load(open("apps_train_sub_val_ids.json")))
        self.split = split
        self.use_starter_code = use_starter_code

    def __call__(self, file_path):
        data = load_dataset(file_path, split=self.split, trust_remote_code=True).to_list()
        logger.info(len(data))

        missing_solutions = 0
        missing_test_cases = 0
        outputs = []
        for item in data:
            if item["solutions"]:
                item["solutions"] = json.loads(item["solutions"])
            else:
                missing_solutions += 1
                continue

            if item["input_output"]:
                item["input_output"] = json.loads(item["input_output"])  # Currently we do not need ground-truth test cases.
                if "fn_name" in item["input_output"]:
                    item["fn_name"] = item["input_output"]["fn_name"]
                    if self.use_starter_code:
                        assert item["starter_code"]
                        item["question"] += f"\n\n{item['starter_code']}"
                    else:
                        item["question"] += f"\n\nYou should name the function as `{item['fn_name']}`."
                item["input_output"] = json.dumps(item["input_output"], ensure_ascii=False)  # For PyArrow dataset usage.
            else:
                missing_test_cases += 1
                continue

            outputs.append(item)

        print(f"Missing solutions: {missing_solutions}")
        print(f"Missing test cases: {missing_test_cases}")

        return outputs


class APPsFlatTestCasesReader(APPsReader):
    def __call__(self, file_path):
        data = super().__call__(file_path)
        outputs = []
        for item in data:
            if item["input_output"]:
                inputs = item["input_output"]["inputs"]
                inputs = [str(_input) for _input in inputs]
                item["test_inputs"] = inputs
                outputs.append(item)

        return outputs


class MBPPReader:
    def __call__(self, file_path):
        data = json.load(open(file_path, encoding='utf-8'))

        # for item in data:
        #     item["input_output"]["inputs"] = [eval(x) for x in item["input_output"]["inputs"]]
        #     item["input_output"]["outputs"] = [eval(x) for x in item["input_output"]["outputs"]]

        return data


def load_code_dataset(subtask: str):
    if subtask in ["add_print_statements", "add_docstring", "replace_variables", "test_case_inputs_gen", "simulate_execute", "simulate_execute_obfuscation_v1",
                   "simulate_execute_obfuscation_v2"]:
        # data = APPsReader(split="train")(f"codeparrot/apps")
        data = MBPPReader()("src/resources/data/code/mbpp-sanitized-256-var-replace-exec-app-style.json")
        if subtask in ["add_print_statements"]:
            # For APPs
            # for item in data:
            #     item["document"] = item["solutions"][0]  # TODO: This is a hack. If there is a more flexible way, do not duplicate the field.
            #     item["meta_data"] = {
            #         "source_code": item["solutions"][0],
            #     }
            for item in data:
                item["document"] = item["code"]
                item["meta_data"] = {
                    "source_code": item["code"],
                }
        elif subtask in ["add_docstring"]:
            for item in data:
                item["document"] = item["code"]
                item["meta_data"] = {}
        elif subtask in ["replace_variables"]:
            for item in data:
                item["document"] = item["code"]
                item["meta_data"] = {
                    "source_code": item["code"],
                    "var_mapping": item["var_mapping"],
                    "_EXTRA_KEY_:var_mapping": "\n".join([f"{pair.split(':')[0]} -> {pair.split(':')[1]}" for pair in item["var_mapping"]]),
                }
        elif subtask in ["test_case_inputs_gen"]:
            for item in data:
                item["document"] = item["code"]
                item["meta_data"] = {
                    "source_code": item["code"],
                    "gen_n": 5,
                    "fn_name": item["input_output"]["fn_name"],
                    "_EXTRA_KEY_:problem": item["prompt"],
                }
        elif subtask in ["simulate_execute"]:
            # For test only
            # data = MBPPReader()("src/resources/data/code/mbpp-sanitized-256-var-replace-exec-app-style-v2.0-random-var-obfuscation-v1.json")
            # data = MBPPReader()("src/resources/data/code/mbpp-sanitized-256-var-replace-exec-app-style-v2.0-random-var-obfuscation-v2.json")
            # data = MBPPReader()("src/resources/data/code/mbpp-sanitized-256-var-replace-exec-app-style-v2.0-random-var-obfuscation-v2.1.json")
            all_data = []
            for item in data:
                item["document"] = item["code"]
                for t_i, (_input, _output) in enumerate(zip(item["input_output"]["inputs"], item["input_output"]["outputs"])):
                    # output = eval(item["input_output"]["outputs"][0])
                    # output = item["input_output"]["outputs"][0]
                    output = eval(_output)
                    if isinstance(output, tuple):
                        output = list(output)

                    # _input = item["input_output"]["inputs"][0]
                    tmp = copy.deepcopy(item)
                    tmp["meta_data"] = {
                        "output": output,
                        "call_based": True,
                        "_EXTRA_KEY_:inputs": _input,
                    }
                    tmp["id"] = f"{item['task_id']}_{t_i}"
                    all_data.append(tmp)

            data = all_data
        elif subtask in ["simulate_execute_obfuscation_v1"]:
            data = MBPPReader()("src/resources/data/code/mbpp-sanitized-256-var-replace-exec-app-style-v2.0-random-var-obfuscation-v1.json")
            all_data = []
            for item in data:
                item["document"] = item["code"]
                for t_i, (_input, _output) in enumerate(zip(item["input_output"]["inputs"], item["input_output"]["outputs"])):
                    output = eval(_output)
                    if isinstance(output, tuple):
                        output = list(output)

                    tmp = copy.deepcopy(item)
                    tmp["meta_data"] = {
                        "output": output,
                        "call_based": True,
                        "_EXTRA_KEY_:inputs": _input,
                    }
                    tmp["id"] = f"{item['task_id']}_{t_i}"
                    all_data.append(tmp)

            data = all_data
        elif subtask in ["simulate_execute_obfuscation_v2"]:
            data = MBPPReader()("src/resources/data/code/mbpp-sanitized-256-var-replace-exec-app-style-v2.0-random-var-obfuscation-v2.1.json")
            all_data = []
            for item in data:
                item["document"] = item["code"]
                for t_i, (_input, _output) in enumerate(zip(item["input_output"]["inputs"], item["input_output"]["outputs"])):
                    output = eval(_output)
                    if isinstance(output, tuple):
                        output = list(output)

                    tmp = copy.deepcopy(item)
                    tmp["meta_data"] = {
                        "output": output,
                        "call_based": True,
                        "_EXTRA_KEY_:inputs": _input,
                    }
                    tmp["id"] = f"{item['task_id']}_{t_i}"
                    all_data.append(tmp)

            data = all_data
    elif subtask in ["add_print_statements_v2", "add_docstring_v2", "replace_variables_v2", "test_case_inputs_gen_v2", "simulate_execute_v2"]:
        # Version 2.0
        if subtask in ["add_print_statements_v2"]:
            data = json.load(open("src/resources/data/code/apps-r200-s42-Copy1.add_print.v2.0.json", encoding='utf-8'))
            for item in data:
                item["document"] = item["code"]
                item["meta_data"] = {
                    "source_code": item["code"],
                }
        elif subtask in ["add_docstring_v2"]:
            data = json.load(open("src/resources/data/code/distract_docstring_standard.v2.0.json", encoding='utf-8'))
            for item in data:
                item["document"] = item["distractor_code"]
                item["meta_data"] = {
                    "fn_name": item["function_name"],
                    "_EXTRA_KEY_:fn_name": item["function_name"],
                }
        elif subtask in ["replace_variables_v2"]:
            data = json.load(open("src/resources/data/code/r_vars.apps.mbpp257.v2.0.json", encoding='utf-8'))
            for item in data:
                item["document"] = item["code"]
                item["meta_data"] = {
                    "source_code": item["code"],
                    "var_mapping": item["var_mapping_random"],
                    "_EXTRA_KEY_:var_mapping": "\n".join([f"{pair.split(':')[0]} -> {pair.split(':')[1]}" for pair in item["var_mapping_random"]]),
                }
        elif subtask in ["test_case_inputs_gen_v2"]:
            data = json.load(open("src/resources/data/code/apps-r199-s42.json", encoding='utf-8'))
            for item in data:
                item["document"] = item["code"]
                item["meta_data"] = {
                    "source_code": item["code"],
                    "gen_n": 5,
                    "_EXTRA_KEY_:problem": item["question"],
                }
                if "fn_name" in item["input_output"]:
                    item["meta_data"]["fn_name"] = item["input_output"]["fn_name"]
                else:
                    item["meta_data"]["fn_name"] = None
        elif subtask in ["simulate_execute_v2"]:
            data = json.load(open("src/resources/data/code/apps-r199-s42.json", encoding='utf-8'))
            for item in data:
                item["document"] = item["code"]

                if "fn_name" in item["input_output"]:
                    # output = eval(item["input_output"]["outputs"][0])
                    output = item["input_output"]["outputs"][0]  # TODO: This is different with MBPP format.
                    if isinstance(output, tuple):
                        output = list(output)

                    _input = item["input_output"]["inputs"][0]
                    if isinstance(_input, list):
                        _input = str(_input)[1:-1]
                else:
                    output = item["input_output"]["outputs"][0]
                    _input = item["input_output"]["inputs"][0]

                item["meta_data"] = {
                    "output": output,
                    "call_based": False,
                    "_EXTRA_KEY_:inputs": _input,
                }

        # return Dataset.from_list(data)
    else:
        raise NotImplementedError(f"Unknown subtask: {subtask}")

    return data

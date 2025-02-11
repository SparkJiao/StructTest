
"""
* Reasoning steps
    * as *
    * as numbers
* Calculation
    * as python blob
    * as <<>>
    * as cal: 
* As final results
    * The answer is:
    * "###
    * Final answer is
    * Latex answer in \boxed
    * Answer in $$
    
"""

import os
import numpy as np
import backoff
import openai
import torch
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
import argparse
import re
import json
from functools import partial

torch.manual_seed(42)
# Set the seed for NumPy
np.random.seed(42)

openai_client = None

NUM_GPUS = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(**kwargs):
    global openai_client
    if openai_client is None:
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return openai_client.chat.completions.create(**kwargs)


def call_model_chatgpt(messages, model, max_tokens=1024):
    try:
        results = completions_with_backoff(
            model=model,
            messages=messages,
            timeout=60,
            max_tokens=max_tokens,
        )
        result = results.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        result = "ERROR: API error outputs"
    return result



class RegistryBase(type):

    REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        new_cls = type.__new__(cls, name, bases, attrs)
        if new_cls.__name__ in cls.REGISTRY:
            raise ValueError(f'{new_cls.__name__} already in {cls.REGISTRY}')
        cls.REGISTRY[new_cls.__name__] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)
    
    @classmethod
    def get_registry_names(cls):
        return dict(cls.REGISTRY).keys()
    
    @classmethod
    def get_class(cls, name):
        assert name in cls.REGISTRY, f'{name} not in {cls.REGISTRY.keys()}'
        return cls.REGISTRY[name]



class MathFormat(metaclass=RegistryBase):
    """
    Applying certain prompting for mat
    """
    example_question = "What is the total cost of purchasing equipment for all sixteen players on the football team, considering that each player requires a $25 jersey, a $15.20 pair of shorts, and a pair of socks priced at $6.80?"
    example_response = "Each player requires a $25 jersey, a $15.20 pair of shorts, and a pair of socks priced at $6.80. So the total cost for each player is $25 + $15.20 + $6.80 = $47. Since there are sixteen players on the football team, the total cost for all of them is 16 * $47 = $752. #### 752"

    system_user_prompts = [
        {
            # format in system prompt
            "system": "You are a helpful assistant.",
            "user": """{question}"""
        },
        {
            # format in system prompt
            "system": "You are a helpful assistant.",
            "user": """{question}"""
        },
    ]
    
    def math_to_conversation(self, question):
        """
        turn question into conversation prefix [system, user]
        """
        system_user_prompt = np.random.choice(self.system_user_prompts)
        conversation = [
            {"role": "system", "content": system_user_prompt['system']},
            {"role": "user", "content": system_user_prompt['user'].format(question=question)},
        ]
        return conversation

    def extract_final_answer(self, question, conversation, response):
        raise NotImplementedError
    

    def evaluate_response(self, question, conversation, response, gold_response=None, **kwargs) -> bool:
        """
        Return if the response complies with the specific format
            True or False
        Return if the response is accurate
        """
        # label = gold_response_to_label(gold_response)
        raise NotImplementedError



class Suffix1PhrasesMathFormat(MathFormat):
    # allowed_suffixes = ["The answer is:", "###", "The final answer is", "Result:"]
    allowed_suffix = "The answer is:"
    suffix_instruction = """Explain your answer and finally state your final result without any extra unit after the phrase "{suffix}"."""
    system_user_prompts = [
        {
            # format in system prompt
            "system": "You are a helpful assistant.",
            "user": """{question}"""
        },
    ]

    def extract_final_answer(self, question, conversation, response):
        if self.allowed_suffix not in response:
            return ""
        answer = response.split(self.allowed_suffix)[-1].strip()
        if answer.endswith("."):
            answer = answer[:-1]
        return answer
    
    def math_to_conversation_data(self, question):
        """
        turn question into conversation prefix [system, user]
        Returns:
            json object with "conversation" as well as any necessary meta data equipped to evaluate the response
        """
        system_user_prompt = np.random.choice(self.system_user_prompts)
        suffix_instruction = self.suffix_instruction.format(suffix=self.allowed_suffix)
        
        if np.random.randint(2) == 0:
            system = system_user_prompt['system'] + f"\n{suffix_instruction}"
            user = system_user_prompt['user'].format(question=question)
        else:
            system = system_user_prompt['system']
            user = (
                suffix_instruction + "\n" + system_user_prompt['user'].format(question=question)
                if np.random.randint(2) == 0 else
                system_user_prompt['user'].format(question=question) + f"\n{suffix_instruction}"
            )

        conversation = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        return conversation
    
    def evaluate_response(self, question, conversation, response, gold_response=None, **kwargs) -> bool:
        """
        Return if the response complies with the specific format
            True or False
        """
        last_line = response.strip().split("\n")[-1]
        suffix_in_response = self.allowed_suffix in last_line
        label = gold_response_to_label(gold_response)
        answer = self.extract_final_answer(question, conversation, response)
        accurate = answer == label
        return suffix_in_response, accurate


class Suffix2PhrasesMathFormat(Suffix1PhrasesMathFormat):
    allowed_suffix = "###"


class Suffix3PhrasesMathFormat(Suffix1PhrasesMathFormat):
    allowed_suffix = "The final answer is"


class Suffix4PhrasesMathFormat(Suffix1PhrasesMathFormat):
    allowed_suffix = "Result:"


class Suffix5PhrasesMathFormat(Suffix1PhrasesMathFormat):
    allowed_suffix = "\\boxed{"
    suffix_instruction = """Explain your answer and finally state your final result without any extra unitas latex expression inside \\boxed{{}}."""
    def extract_final_answer(self, question, conversation, response):
        if self.allowed_suffix not in response:
            return ""
        answer = response.split(self.allowed_suffix)[-1].strip()
        if "}" not in answer:
            return ""
        else:
            answer = answer.split("}")[0]
            return answer



class Suffix6PhrasesMathFormat(Suffix1PhrasesMathFormat):
    allowed_suffix = "Result: $"
    suffix_instruction = """Explain your answer and finally state your final result without any extra unitin a new line the format of "Result: <result>" with Latex dollar-sign notations"""

    def evaluate_response(self, question, conversation, response, gold_response=None, **kwargs) -> bool:
        last_line = response.strip().split("\n")[-1]
        correct = last_line.startswith("Result: $") and (last_line.endswith("$") or last_line.endswith("$."))
        answer = self.extract_final_answer(question, conversation, response)
        label = gold_response_to_label(gold_response)
        accurate = answer == label
        return correct, accurate
    
    def extract_final_answer(self, question, conversation, response):
        if self.allowed_suffix not in response:
            return ""
        answer = response.split(self.allowed_suffix)[-1].strip()
        if answer.startswith("$"):
            if "$$" not in answer:
                return ""
            else:
                answer = answer[1:].split("$$")[0].strip()
                return answer
        else:
            if "$$" in answer:
                return ""
            answer = answer.split("$")[0].strip()
            return answer


class Suffix7PhrasesMathFormat(Suffix1PhrasesMathFormat):
    allowed_suffix = "<result>"
    suffix_instruction = """Explain your answer and finally state your final result without any extra unitas latex expression inside <result> tag."""



# ! Pending: number of reasoning steps

class BulletPointsMathFormat(MathFormat):
    # checking bullet points
    num_bullet_points = [3, 10]
    # style can be different things
    """
    Produce reasoning steps in different styles and with strict number of steps range
        - markdown, JSON
        - in [3-10] steps

    1. Markdown ## style
    2. Markdown 1.
    3. JSON list
    4. JSON dict with index value reasoning
    5. Step 1, Step 2....

    """
    bullet_style = {

    }
    def math_to_conversation_data(self, question):
        """
        turn question into conversation prefix [system, user]
        Returns:
            json object with "conversation" as well as any necessary meta data equipped to evaluate the response
        """
        system_user_prompt = np.random.choice(self.system_user_prompts)
        suffix_instruction = self.suffix_instruction.format(suffix=self.allowed_suffix)
        
        if np.random.randint(2) == 0:
            system = system_user_prompt['system'] + f"\n{suffix_instruction}"
            user = system_user_prompt['user'].format(question=question)
        else:
            system = system_user_prompt['system']
            user = (
                suffix_instruction + "\n" + system_user_prompt['user'].format(question=question)
                if np.random.randint(2) == 0 else
                system_user_prompt['user'].format(question=question) + f"\n{suffix_instruction}"
            )

        conversation = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        return conversation
    
    def extract_bullet_points(self, question, conversation, response):
        raise NotImplementedError
    
    def evaluate_response(self, question, conversation, response, gold_response=None, **kwargs) -> bool:
        """
        Return if the response complies with the specific format
            True or False
        """
        question_id = kwargs.get("question_id", None)
        try:
            bullet_points = self.extract_bullet_points(question, conversation, response, question_id=question_id)
        except Exception as e:
            return False, False

        correct = self.num_bullet_points[0] <= len(bullet_points) <= self.num_bullet_points[1]
        answer = self.extract_final_answer(question, conversation, response)
        label = gold_response_to_label(gold_response)
        accurate = answer == label
        return correct, accurate



class BulletPoints1MathFormat(BulletPointsMathFormat):
    suffix_instruction = """Explain your answer using {bullet_low} to {bullet_high} bullet points.\n{style_instruction}.\nFinally state your final result without any extra unit after the phrase "{suffix}"."""

    system_user_prompts = [
        {
            # format in system prompt
            "system": "You are a helpful assistant.",
            "user": """{question}"""
        },
    ]
    allowed_suffix = "The answer is:"
    style_instruction = """
You style should be in Markdown style using header `### <step_number>` as starting point of each step, where <step_number> is a number indicating the step number index beginning with 1. Each step should be separated by double new lines.
"""
    line_numbered_pattern = r'^### \d+'
    expected_prefix_pattern = "### {point_count}"
    def __init__(self):
        super().__init__()
        self.log_counter = 0

    def math_to_conversation_data(self, question):
        """
        turn question into conversation prefix [system, user]
        Returns:
            json object with "conversation" as well as any necessary meta data equipped to evaluate the response
        """
        system_user_prompt = np.random.choice(self.system_user_prompts)
        suffix_instruction = self.suffix_instruction.format(
            bullet_low=self.num_bullet_points[0],
            bullet_high=self.num_bullet_points[1],
            style_instruction=self.style_instruction.strip(),
            suffix=self.allowed_suffix,
        )
        
        if np.random.randint(2) == 0:
            system = system_user_prompt['system'] + f"\n{suffix_instruction}"
            user = system_user_prompt['user'].format(question=question)
        else:
            system = system_user_prompt['system']
            user = (
                suffix_instruction + "\n" + system_user_prompt['user'].format(question=question)
                if np.random.randint(2) == 0 else
                system_user_prompt['user'].format(question=question) + f"\n{suffix_instruction}"
            )

        conversation = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        return conversation

    def extract_final_answer(self, question, conversation, response):
        if self.allowed_suffix not in response:
            return ""
        answer = response.split(self.allowed_suffix)[-1].strip()
        if answer.endswith("."):
            answer = answer[:-1]
        return answer
    
    def extract_bullet_points(self, question, conversation, response, question_id=None):
        pattern = self.line_numbered_pattern
        lines = response.strip().split("\n\n")
        
        steps = []
        in_markdown_list = False
        point_count = 1

        for li, line in enumerate(lines):
            stripped_line = line.strip()
            if re.match(pattern, stripped_line):
                in_markdown_list = True
                expected_prefix = self.expected_prefix_pattern.format(point_count=point_count)
                if not stripped_line.startswith(expected_prefix):
                    # return None # if out of sequence, return False
                    raise ValueError(f'Out of sequence for {response}')
                point_count += 1
                steps.append(stripped_line.split(expected_prefix)[-1])
            # If in_markdown_list is True and the next markdown argument is not met
            elif in_markdown_list and not re.match(pattern, stripped_line) and stripped_line != '':
                # ending part
                # return False  # Found a non-empty line that doesn't match the pattern
                # break
                # but a step may have new lines, which is fine ?
                steps[-1] = steps[-1] + "\n\n" + line
        
        # if question_id is not None and question_id % 10 == 0:
        #     print(f'{self.__class__.__name__}:-----\n{response}\n>>>>>>>>>{json.dumps(steps, indent=1)}\nGIVEN\n{conversation=}')
        return steps


def is_markdown_list_format(text):
    # Create a regex pattern to match "### Argument <n>"
    pattern = r'^### Argument \d+'

    # Split the text into lines
    lines = text.strip().split('\n')

    # Flag to keep track of list order
    in_markdown_list = False
    argument_count = 1

    # Iterate over each line
    for line in lines:
        # Strip leading and trailing whitespaces
        stripped_line = line.strip()

        # Check if the line matches the markdown argument format
        if re.match(pattern, stripped_line):
            in_markdown_list = True  # Set flag True as soon as we hit the first argument
            expected_argument = f"### Argument {argument_count}"
            
            if stripped_line != expected_argument:
                return False  # If out of sequence, return False

            argument_count += 1

        # If in_markdown_list is True and the next markdown argument is not met
        elif in_markdown_list and not re.match(pattern, stripped_line) and stripped_line != '':
            return False  # Found a non-empty line that doesn't match the pattern

    # Return True if all arguments are in sequence
    return True


class BulletPoints2MathFormat(BulletPoints1MathFormat):
    allowed_suffix = "####"


class BulletPoints3MathFormat(BulletPoints1MathFormat):
    allowed_suffix = "Final result:"


class BulletPoints4MathFormat(BulletPoints1MathFormat):
    allowed_suffix = "\\boxed{"
    def extract_final_answer(self, question, conversation, response):
        if self.allowed_suffix not in response:
            return ""
        answer = response.split(self.allowed_suffix)[-1].strip()
        if "}" not in answer:
            return ""
        else:
            answer = answer.split("}")[0]
            return answer


class BulletPoints5MathFormat(BulletPoints1MathFormat):

    system_user_prompts = [
        {
            # format in system prompt
            "system": "You are a helpful assistant.",
            "user": """{question}"""
        },
    ]
    allowed_suffix = "The answer is:"
    style_instruction = """
You style should be in Markdown style using header `<step-number>. <reasoning step>` as starting point of each step, where <step_number> is a number indicating the step number index beginning with 1. Each step should be separated by double new lines.
"""
    line_numbered_pattern = r'^\d+\.\s(.*)'
    # line_numbered_pattern = r'^### \d+'
    expected_prefix_pattern = "{point_count}. "


class BulletPoints6MathFormat(BulletPoints5MathFormat):
    allowed_suffix = "####"


class BulletPoints7MathFormat(BulletPoints5MathFormat):
    allowed_suffix = "Final result:"


class BulletPoints8MathFormat(BulletPoints5MathFormat):
    allowed_suffix = "\\boxed{"
    def extract_final_answer(self, question, conversation, response):
        if self.allowed_suffix not in response:
            return ""
        answer = response.split(self.allowed_suffix)[-1].strip()
        if "}" not in answer:
            return ""
        else:
            answer = answer.split("}")[0]
            return answer


# -----

class BulletPoints9MathFormat(BulletPoints1MathFormat):

    system_user_prompts = [
        {
            # format in system prompt
            "system": "You are a helpful assistant.",
            "user": """{question}"""
        },
    ]
    allowed_suffix = "The answer is:"
    style_instruction = """
You style should be in Markdown style using header `**Step <step-number>** <reasoning step>` as starting point of each step, where <step_number> is a number indicating the step number index beginning with 1. Each step should be separated by double new lines.
"""
    line_numbered_pattern = r'^\*\*Step \d+\*\*\s+(.*)'
    # line_numbered_pattern = r'^### \d+'
    expected_prefix_pattern = "**Step {point_count}**"


class BulletPoints10MathFormat(BulletPoints9MathFormat):
    allowed_suffix = "####"


class BulletPoints11MathFormat(BulletPoints9MathFormat):
    allowed_suffix = "Final result:"


class BulletPoints12MathFormat(BulletPoints9MathFormat):
    allowed_suffix = "\\boxed{"
    def extract_final_answer(self, question, conversation, response):
        if self.allowed_suffix not in response:
            return ""
        answer = response.split(self.allowed_suffix)[-1].strip()
        if "}" not in answer:
            return ""
        else:
            answer = answer.split("}")[0]
            return answer


class BulletPoints13MathFormat(BulletPoints1MathFormat):
    # JSON list
    allowed_suffix = "The answer is:"
    style_instruction = """
You style should be a JSON list, where each element is a string representing a reasoning step. The JSON list must be encapsulated within the ```json <your json list> ``` format.
"""
    def extract_bullet_points(self, question, conversation, response, question_id=None):
        pattern = r'```json\s*([\s\S]*?)\s*```'

        match = re.search(pattern, response)
        if not match:
            raise ValueError("JSON snippet not found in the correct format.")

        # Extract the JSON string from the match group
        json_string = match.group(1).strip()

        try:
            # Attempt to parse the JSON string
            steps = json.loads(json_string)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format detected.")
        
        if not isinstance(steps, list):
            raise ValueError("Invalid JSON format detected not list.")

        if not all(isinstance(x, str) for x in steps):
            raise ValueError("Invalid JSON format detected not list, not alls tring")
        
        # if question_id is not None and question_id % 10 == 0:
        #     print(f'{self.__class__.__name__}:-----\n{response}\n>>>>>>>>>{json.dumps(steps, indent=1)}\nGIVEN\n{conversation=}')
        return steps


def extract_json_snippet(text):
    # Define a pattern to capture everything within ```json ... ```
    pattern = r'```json\s*([\s\S]*?)\s*```'
    
    # Search for the pattern in the provided text
    match = re.search(pattern, text)
    if not match:
        raise ValueError("JSON snippet not found in the correct format.")

    # Extract the JSON string from the match group
    json_string = match.group(1).strip()

    try:
        # Attempt to parse the JSON string
        json_data = json.loads(json_string)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format detected.")
    
    return json_data

class BulletPoints14MathFormat(BulletPoints13MathFormat):
    allowed_suffix = "####"


class BulletPoints15MathFormat(BulletPoints13MathFormat):
    allowed_suffix = "Final result:"


class BulletPoints16MathFormat(BulletPoints13MathFormat):
    allowed_suffix = "\\boxed{"
    def extract_final_answer(self, question, conversation, response):
        if self.allowed_suffix not in response:
            return ""
        answer = response.split(self.allowed_suffix)[-1].strip()
        if "}" not in answer:
            return ""
        else:
            answer = answer.split("}")[0]
            return answer


class BulletPoints17MathFormat(BulletPoints1MathFormat):
    # JSON list
    allowed_suffix = "The answer is:"
    style_instruction = """
You style should be a JSON dict, where each key-value pair represents a reasoning step. \
Specifically, the key is an integer indicating the step number beginning with 1, and value is a string representing the reasoning content. \
The JSON dict must be encapsulated within the ```json <your json list> ``` format.
"""
    def extract_bullet_points(self, question, conversation, response, question_id=None):
        pattern = r'```json\s*([\s\S]*?)\s*```'

        match = re.search(pattern, response)
        if not match:
            raise ValueError("JSON snippet not found in the correct format.")

        # Extract the JSON string from the match group
        json_string = match.group(1).strip()

        try:
            # Attempt to parse the JSON string
            step_dict = json.loads(json_string)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format detected.")
        
        # if question_id is not None and question_id % 10 == 0:
        #     print(f'{self.__class__.__name__}:-----\n{response}\nGIVEN\n{conversation=}')

        if not isinstance(step_dict, dict):
            print(f'not dict')
            raise ValueError("Invalid JSON format detected not list.")

        if not all(isinstance(v, str) for k, v in step_dict.items()):
            print(f'not dict')
            raise ValueError("Invalid JSON format detected , not alls tring")

        for k in step_dict.keys():
            _ = int(k)
        
        steps = [
            step_dict.get(i + 1, step_dict.get(str(i + 1), None))
            for i in range(len(step_dict))
        ]
        if any(x is None for x in steps):
            print(f'{steps=} none')
            raise ValueError("Invalid JSON format detected , not alls tring")

        if question_id is not None and question_id % 10 == 0:
            print(f'>>>>>>>>>{json.dumps(steps, indent=1)}\n')
        
        return steps


class BulletPoints18MathFormat(BulletPoints17MathFormat):
    allowed_suffix = "####"


class BulletPoints19MathFormat(BulletPoints17MathFormat):
    allowed_suffix = "Final result:"


class BulletPoints20MathFormat(BulletPoints17MathFormat):
    allowed_suffix = "\\boxed{"
    def extract_final_answer(self, question, conversation, response):
        if self.allowed_suffix not in response:
            return ""
        answer = response.split(self.allowed_suffix)[-1].strip()
        if "}" not in answer:
            return ""
        else:
            answer = answer.split("}")[0]
            return answer


# ! Pending: calculation format
"""
Model should produce certain caculator/python interpreter friendly snippet or expression that can produce 100% accurate numbers
e.g:
"Please put any caculation expression in << and >> symbols." ---> "The answer is <<10*pi>> = 31.4" --> 31.4 (31.4 is produced by calculator, not model)


try different easier or harder formats
start with [ end with }
produce a python snippet using ```python ``` snippet
"""


def gold_response_to_label(response):
    label = response.split("####")[-1].strip()
    return label


def process_example_add_format_properties(example, idx, formats_dataset):
    format_properties = formats_dataset[idx]['format_properties']
    example['format_properties'] = format_properties
    return example

def process_example_format_properties(example, idx, allowed_formats):
    # print(f"Inside process_example_format_properties")
    assert 'format_properties' not in example, "error with format_properties in process_example_format_properties"
    math_format_name = np.random.choice(allowed_formats)
    return {"format_properties": {"math_format_name": math_format_name}}

from pathlib import Path
from datasets import load_from_disk
def load_format_properties_from_local_disk(subtask, dataset_name, dataset, allowed_formats):
    # print("inside load_format_properties_from_local_disk()")
    local_data_path = f"data/{dataset_name.replace('/', '_')}_format_properties/{subtask}"
    # print(f"local data path: {local_data_path}")
    # if not exist, create a local dataset version with format properties
    if not Path(local_data_path).exists():
        # print("inside load_format_properties_from_local_disk: path does not exist")
        format_properties_dataset = dataset.map(process_example_format_properties, with_indices=True, fn_kwargs={"allowed_formats": allowed_formats}, remove_columns=dataset.column_names, keep_in_memory=True)
        format_properties_dataset.save_to_disk(local_data_path)
    else:
        # print(f"inside load_format_properties_from_local_disk: path exist: {local_data_path}")
        format_properties_dataset = load_from_disk(local_data_path)
    # add format properties into original dataset
    dataset = dataset.map(process_example_add_format_properties, with_indices=True, fn_kwargs={"formats_dataset": format_properties_dataset}, keep_in_memory=True)
    return dataset

from pipeline.online_inference import OnlineGenerator
def eval_example(example, idx, args, model_name):
    math_format_name = example['format_properties']['math_format_name']
    math_format = MathFormat.get_class(math_format_name)()
    question = example['question']
    gold_response = example['answer']
    # gold_label = gold_response_to_label(gold_response)
    conversation = math_format.math_to_conversation_data(question)
    generator = OnlineGenerator(model_name=model_name, dataset='', task="math")
    # generated = generator.generate(conversation, temperature=1e-6)
    generated = generator.generate(conversation, temperature=args.temperature, timeout=args.timeout)
    
    eval_value, accuracy_value = math_format.evaluate_response(question, conversation, generated, gold_response, question_id=idx)
    return {"format_result": eval_value, "accuracy_result": accuracy_value, 'generated': generated, 'idx': idx}

import pandas as pd
def run_eval_parallel(dataset, model_name, allowed_formats, subtask, args):
    clean_model_name = model_name.split("/")[-1]
    output_path = f"outputs/output_{clean_model_name}_math_{subtask}.csv"
    print(allowed_formats)
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        format_results = df['format_results']
        acc_results = df['acc_results']
    else:
        processed_dataset = dataset.map(eval_example, with_indices=True,
                    fn_kwargs={"model_name": model_name, "args": args},
                    num_proc=args.num_proc, load_from_cache_file=False)
        
        format_results = processed_dataset['format_result']
        acc_results = processed_dataset['accuracy_result']
        result = [1 if res == True else 0 for res in format_results]

        generated = processed_dataset['generated']
        format_properties = processed_dataset['format_properties']
        index = processed_dataset['idx']

        df = pd.DataFrame({"index": index, "generated": generated, "result": result, "format_results": format_results, "acc_results": acc_results, 'format_properties': format_properties})
        df = df.sort_values(by='index')
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"Exporting results to: {output_path}")
        df.to_csv(output_path, index=False)
    
    format_accuracy = np.array(format_results).astype(float).mean()
    score_accuracy = np.array(acc_results).astype(float).mean()
    strict_accuracy = (np.array(format_results) & np.array(acc_results)).astype(float).mean()
    print(f"{format_accuracy=} | {score_accuracy=}")
    return {
        "format_accuracy": format_accuracy * 100,
        "score_accuracy": score_accuracy * 100,
        "strict_accuracy": strict_accuracy * 100,
    }

def run_evaluation(args, dataset, model_name):
    styles = args.math_eval_type.split(",")
    results = {}
    for i, style in enumerate(styles):
        if style == "suffix_phrase":
            allowed_formats = [f"Suffix{i + 1}PhrasesMathFormat" for i in range(6)]
        elif style == "bullet_points":
            allowed_formats = [f"BulletPoints{i + 1}MathFormat" for i in range(20)]
        
        dataset = load_format_properties_from_local_disk(style, "math", dataset, allowed_formats)
        eval_result = run_eval_parallel(dataset, model_name, allowed_formats, style, args)
        results[style] = eval_result
    print(f'Final:\n{args.model_name=}\n{json.dumps(results, indent=1)}')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt-4o")
    # parser.add_argument("--math_eval_type", default="suffix_phrase", choices=['suffix_phrase', 'bullet_points'])
    parser.add_argument("--math_eval_type", default="suffix_phrase")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument(
        "--temperature", type=float, default=0.01, help="The sampling temperature during generation"
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="timeout in seconds for vllm_post request or openai API"
    )
    args = parser.parse_args()
    print(args)

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    # inference_func, inference_style = get_inference_function(model_type=args.model_type, model_name=args.model_name)
    run_evaluation(args, dataset, args.model_name)

if __name__ == "__main__":
    main()
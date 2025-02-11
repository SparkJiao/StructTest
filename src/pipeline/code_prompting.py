from transformers import PreTrainedTokenizer
from typing import Optional
from functools import partial


def code_prompt_prepare_base(
        document: str,
        context_length: int = 4096,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        prompt_file: str = "resources/code/add_print_0shot.txt",
        key_field: str = "[[code_snippet]]",
        **kwargs,
):
    prompt_template = open(prompt_file).read()
    prompt = prompt_template.replace(key_field, document)
    for k, v in kwargs.items():
        if k.startswith("_EXTRA_KEY_:"):
            extra_key = k.split("_EXTRA_KEY_:")[1]
            prompt = prompt.replace(f"[[{extra_key}]]", v)
    return prompt


# TODO: Do we need add an interface to change the default prompt file?
default_functions = {
    # "add_print_statements": partial(code_prompt_prepare_base, prompt_file="src/resources/prompts/code/add_print_0shot.txt", key_field="[[code_snippet]]"),
    "add_print_statements": partial(code_prompt_prepare_base, prompt_file="src/resources/prompts/code/add_print_1shot_tag.txt", key_field="[[code_snippet]]"),
    "add_print_statements_v2": partial(code_prompt_prepare_base, prompt_file="src/resources/prompts/code/add_print_1shot_tag.txt",
                                       key_field="[[code_snippet]]"),
    # "add_docstring": partial(code_prompt_prepare_base, prompt_file="src/resources/prompts/code/docstring_gen_0shot.txt", key_field="[[Function]]"),
    "add_docstring": partial(code_prompt_prepare_base, prompt_file="src/resources/prompts/code/docstring_gen_1shot_tag.txt", key_field="[[code_snippet]]"),
    "add_docstring_v2": partial(code_prompt_prepare_base, prompt_file="src/resources/prompts/code/docstring_gen_1shot_distract_tag.txt",
                                key_field="[[code_snippet]]"),
    # "replace_variables": partial(code_prompt_prepare_base, prompt_file="src/resources/prompts/code/replace_variable_0shot.txt", key_field="[[code_snippet]]"),
    "replace_variables": partial(code_prompt_prepare_base, prompt_file="src/resources/prompts/code/replace_variable_1shot.txt", key_field="[[code_snippet]]"),
    "replace_variables_v2": partial(code_prompt_prepare_base, prompt_file="src/resources/prompts/code/replace_variable_1shot.txt",
                                    key_field="[[code_snippet]]"),
    "test_case_inputs_gen_v2": partial(code_prompt_prepare_base, prompt_file="src/resources/prompts/code/test_case_input_gen_2shot.txt",
                                       key_field="[[code_snippet]]"),
    "test_case_inputs_gen": partial(code_prompt_prepare_base, prompt_file="src/resources/prompts/code/test_case_input_gen_call_based_1shot_tag.txt",
                                    key_field="[[code_snippet]]"),
    # "simulate_execute": partial(code_prompt_prepare_base, prompt_file="src/resources/prompts/code/simulate_program.txt", key_field="[[code_snippet]]"),
    "simulate_execute": partial(code_prompt_prepare_base, prompt_file="src/resources/prompts/code/simulate_program_1shot_tag.txt",
                                key_field="[[code_snippet]]"),
    "simulate_execute_v2": partial(code_prompt_prepare_base, prompt_file="src/resources/prompts/code/simulate_program_1shot_tag.txt",
                                   key_field="[[code_snippet]]"),
    "simulate_execute_obfuscation_v1": partial(code_prompt_prepare_base, prompt_file="src/resources/prompts/code/simulate_program_1shot_tag.txt",
                                               key_field="[[code_snippet]]"),
    "simulate_execute_obfuscation_v2": partial(code_prompt_prepare_base, prompt_file="src/resources/prompts/code/simulate_program_1shot_tag.txt",
                                               key_field="[[code_snippet]]"),
}


def code_prompt_prepare(subtask, document, generator, **kwargs):
    if subtask not in default_functions:
        raise ValueError(f"Unknown subtask: {subtask}")

    tokenizer = generator.tokenizer
    context_length = generator.context_length - generator.max_summary_length

    return default_functions[subtask](document=document, context_length=context_length, tokenizer=tokenizer, **kwargs)

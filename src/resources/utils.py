import os
import random
from collections import deque
from html.parser import HTMLParser
from operator import index

import numpy as np
from sqlalchemy import column

from src.resources.text_processing import count_delims, isolate_html, basic_postprocessing
from src.resources.sampling_ranges import n_sentences_interval, symbols, n_points_interval, \
    n_sentences_per_point_interval, n_subpoints_interval, n_tags_count_interval_v1, n_tags_count_interval_v2
from pathlib import Path
from datasets import load_from_disk, load_dataset
import pandas as pd
import json
hf_token = json.load(open('config.json', 'r'))['hf_token']


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONASSEED"] = str(seed)
    np.random.seed(seed)


def map_model_name_to_context_window(model_name):
    lower_model_name = model_name.lower()
    if "llama-2" in lower_model_name:
        return 4096
    elif "llama-3" in lower_model_name:
        return 8192
    else:
        return 16000


def map_task_to_dataset_name(task):
    if task == "summarization" or "html_encoding":
        return "compositional-format-following/summarization_eval"
    if task == "code":
        return "compositional-format-following/code_eval"
    else:
        raise ValueError("Unsupported task name. Please specify a supported task.")


def map_task_to_tags(task, subtask=""):
    if task == "summarization":
        start_tag, end_tag = "<summary>", "</summary>"
        return start_tag, end_tag
    elif task == "html_encoding":
        start_tag, end_tag = "<html_encoding>", "</html_encoding>"
        return start_tag, end_tag
    elif task == "code":
        start_tag, end_tag = "<ans>", "</ans>"
        return start_tag, end_tag
    else:
        raise ValueError("Unsupported task name. Please specify a supported task.")


def map_dataset_name_to_summary_length(task):
    # return 1024 # debugging for tiny models
    if task == "html_encoding":
        return 4096
    elif task == "code":
        return 4096
    elif task == "math":
        return 4096
    return 512


def sample_format(subtask, extra_mode):
    format_params = {}
    if subtask == "length":
        n_sentences = np.random.randint(low=n_sentences_interval[0], high=n_sentences_interval[1]+1)
        format_params["n_sentences"] = n_sentences
    elif subtask == "bullet_points":
        n_points = np.random.randint(low=n_points_interval[0], high=n_points_interval[1]+1)
        symbol = symbols[np.random.randint(len(symbols))]
        format_params["n_points"] = n_points
        format_params["symbol"] = symbol
    elif subtask == "numbered_points":
        n_points = np.random.randint(low=n_points_interval[0], high=n_points_interval[1]+1)
        format_params["n_points"] = n_points
    elif subtask == "bullet_points_length":
        n_points = np.random.randint(low=n_points_interval[0], high=n_points_interval[1]+1)
        symbol = symbols[np.random.randint(len(symbols))]
        n_sentences_per_point = np.random.randint(low=n_sentences_per_point_interval[0], high=n_sentences_per_point_interval[1]+1)
        n_sentences_total = n_points * n_sentences_per_point
        format_params["n_points"] = n_points
        format_params["symbol"] = symbol
        format_params["n_sentences_total"] = n_sentences_total
        format_params["n_sentences_per_point"] = n_sentences_per_point
    elif subtask == "numbered_points_length":
        n_points = np.random.randint(low=n_points_interval[0], high=n_points_interval[1]+1)
        n_sentences_per_point = np.random.randint(low=n_sentences_per_point_interval[0], high=n_sentences_per_point_interval[1]+1)
        n_sentences_total = n_points * n_sentences_per_point
        format_params["n_points"] = n_points
        format_params["n_sentences_total"] = n_sentences_total
        format_params["n_sentences_per_point"] = n_sentences_per_point
    elif subtask == "indented_bullet_points":
        n_points = np.random.randint(low=n_points_interval[0], high=n_points_interval[1]+1)
        symbol = symbols[np.random.randint(len(symbols))]
        n_subpoints = np.random.randint(low=n_subpoints_interval[0], high=n_subpoints_interval[1]+1)
        format_params["n_points"] = n_points
        format_params["symbol"] = symbol
        format_params["n_subpoints"] = n_subpoints
    elif subtask == "nested_html_encoding":
        if extra_mode == "random_easy":
            n_tags_count_interval = n_tags_count_interval_v1
        elif extra_mode == "random_hard":
            n_tags_count_interval = n_tags_count_interval_v2

        while True:
            number_of_head_tags = np.random.randint(low=n_tags_count_interval[0], high=n_tags_count_interval[1] + 1)
            number_of_body_tags = np.random.randint(low=n_tags_count_interval[0], high=n_tags_count_interval[1] + 1)
            number_of_div_tags = np.random.randint(low=n_tags_count_interval[0], high=n_tags_count_interval[1] + 1)

            number_of_title_tags = np.random.randint(low=n_tags_count_interval[0], high=n_tags_count_interval[1] + 1)
            number_of_h1_tags = np.random.randint(low=n_tags_count_interval[0], high=n_tags_count_interval[1] + 1)
            number_of_h2_tags = np.random.randint(low=n_tags_count_interval[0], high=n_tags_count_interval[1] + 1)
            number_of_p_tags = np.random.randint(low=n_tags_count_interval[0], high=n_tags_count_interval[1] + 1)
            number_of_footer_tags = np.random.randint(low=n_tags_count_interval[0], high=n_tags_count_interval[1] + 1)

            format_params["no_of_tags"] = {
                'html': 1,
                'head': number_of_head_tags,
                'title': number_of_title_tags,
                'body': number_of_body_tags,
                'div': number_of_div_tags,
                'h1': number_of_h1_tags,
                'h2': number_of_h2_tags,
                'p': number_of_p_tags,
                'footer': number_of_footer_tags
            }

            n_body = number_of_body_tags
            n_div = n_body * number_of_div_tags
            n_h1 = n_div * number_of_h1_tags
            n_h2 = n_div * number_of_h2_tags
            n_p = n_div * number_of_p_tags
            n_footer = n_body * number_of_footer_tags
            format_params["ground_truth"] = {
                'html': 1,
                'head': number_of_head_tags,
                'title': number_of_head_tags*number_of_title_tags,
                'body': n_body,
                'div': n_div,
                'h1': n_h1,
                'h2': n_h2,
                'p': n_p,
                'footer': n_footer
            }
            if sum(list(format_params["ground_truth"].values())) <= 700: # we don't want to generate too many tags.
                break

        format_params["eval_mode"] = "nested_eval"
        # print(f"Format params: {format_params}")
    return format_params


def nested_bullet_points_counter(file_name, bullet_ids=["*", "+"], nesting_delim="\t", levels=[4,3], verbose=False):
    '''
    Implement count of nested bullet points from the provided text-blob
    params:
        file_name: file location
        bullet_ids = list of bullet ids for each level.
        nesting_delims = by default it's a tab
        levels = number of elements in each nesting, list index corresponds to levels and content is the count of items.
    output:
        returns a list where each item position indicates the level and
        the corresponding item value is the number of items in that level.
    '''
    # print(len(levels))
    count = [0]*len(levels)
    with open(file_name, "r") as file:
        for line in file:
            if verbose:
                print(line)
            if line.startswith(bullet_ids[0]):
                level = 0
                count[level] += 1
            elif line.startswith(nesting_delim):
                delims_count, delim_splits = count_delims(line, nesting_delim, verbose)
                if verbose:
                    print("delims_count:", delims_count, "delim_splits:", delim_splits[len(delim_splits)-1].strip()[0],
                          "bullet_ids[level]:", bullet_ids[delims_count])
                if delim_splits[len(delim_splits)-1].strip().split()[0] == bullet_ids[delims_count]:
                    count[delims_count] += 1
    return count


def process_example_add_format_properties(example, idx, formats_dataset):
    format_properties = formats_dataset[idx]['format_properties']
    example['format_properties'] = format_properties
    return example


def process_example_format_properties(example, idx, subtask, extra_mode):
    # print(f"Inside process_example_format_properties")
    assert 'format_properties' not in example, "error with format_properties in process_example_format_properties"
    format_properties = sample_format(subtask, extra_mode)
    return {"format_properties": format_properties}


def load_format_properties_from_local_disk(subtask, extra_mode, dataset_name, dataset):
    # print("inside load_format_properties_from_local_disk()")
    if extra_mode:
        local_data_path = f"data/{dataset_name.replace('/', '_')}_format_properties/{subtask}_{extra_mode}"    
    else:
        local_data_path = f"data/{dataset_name.replace('/', '_')}_format_properties/{subtask}"
    # print(f"local data path: {local_data_path}")
    # if not exist, create a local dataset version with format properties
    if not Path(local_data_path).exists():
        # print("inside load_format_properties_from_local_disk: path does not exist")
        dataset = load_dataset(dataset_name, split="train")
        # print(f"dataset: {dataset}")
        format_properties_dataset = dataset.map(process_example_format_properties, with_indices=True, fn_kwargs={"subtask": subtask, "extra_mode": extra_mode}, remove_columns=dataset.column_names, keep_in_memory=True)
        format_properties_dataset.save_to_disk(local_data_path)
    else:
        # print(f"inside load_format_properties_from_local_disk: path exist: {local_data_path}")
        format_properties_dataset = load_from_disk(local_data_path)
    # add format properties into original dataset
    dataset = dataset.map(process_example_add_format_properties, with_indices=True, fn_kwargs={"formats_dataset": format_properties_dataset}, keep_in_memory=True)
    return dataset


def check_exists_or_quit(output_path):
    if Path(output_path).exists():
        df = pd.read_csv(output_path)
        print(f'Skip as {output_path} already exists !')
        avg_acc = df['result'].map(lambda x: 1 if (x == 1 or str(x).lower() == 'true') else 0).mean() * 100
        print(f"Average Accuracy (%): {avg_acc}, # {len(df)}")
        err_size = len(df[df['generated'] == 'ERROR'])
        if err_size > 0:
            print(f"Error generation: {err_size} / {len(df)}")
        exit(0)


class MyHTMLParser(HTMLParser):
    def __init__(self):
        self.verbose = False
        self.tag_closures = {}
        self.nested_check = {}
        self.tag_nested_trace = {}
        self.nested_in_head_flag = False
        self.nested_in_body_flag = False
        self.nested_in_div_flag = False
        self.nested_in_html_flag = False
        super().__init__()

    def handle_starttag(self, tag, attrs):
        if self.verbose:
            print("Encountered a start tag:", tag)
            print("Tag closure dict status:", self.tag_closures)

        if tag not in self.tag_closures.keys():
            self.tag_closures[tag] = []
            self.tag_closures[tag].append('start')

        else:
            self.tag_closures[tag].append('start')
        self.is_nested(tag)

    def handle_endtag(self, tag):
        if self.verbose:
            print("Encountered an end tag:", tag)
            print("Tag closure dict status:", self.tag_closures)

        if tag not in self.tag_closures.keys():
            self.tag_closures[tag] = []
            self.tag_closures[tag].append('end')

        else:
            self.tag_closures[tag].append('end')

    def handle_data(self, data):
        if self.verbose:
            print("Encountered some data:", data)

    def is_nested(self, tag):
        if tag in ['head', 'body']:
            try:
                if 'start' == self.tag_closures['html'][len(self.tag_closures['html'])-1]: #and 'end' not in self.tag_closures['html']:
                    self.nested_in_html_flag = True
                else:
                    self.nested_in_html_flag = False
                if tag not in self.tag_nested_trace.keys():
                    self.tag_nested_trace[tag] = []
                    self.tag_nested_trace[tag].append(self.nested_in_html_flag)
                else:
                    self.tag_nested_trace[tag].append(self.nested_in_html_flag)
            except:
                # print(f"key:{tag} not found yet")
                pass
        elif tag == 'title':
            try:
                if 'start' == self.tag_closures['head'][len(self.tag_closures['head'])-1]: #and 'end' not in self.tag_closures['head']:
                    if self.verbose:
                        print(f"self.tag_closures when condition satisfies for tag {tag}: {self.tag_closures}")
                    self.nested_in_head_flag = True
                else:
                    if self.verbose:
                        print(f"self.tag_closures else for {tag}: {self.tag_closures}")
                    self.nested_in_head_flag = False
                if self.verbose:
                    print(f"self.nested_in_head_flag: {self.nested_in_head_flag}")
                if tag not in self.tag_nested_trace.keys():
                    self.tag_nested_trace[tag] = []
                    self.tag_nested_trace[tag].append(self.nested_in_head_flag)
                else:
                    self.tag_nested_trace[tag].append(self.nested_in_head_flag)
            except:
                print(f"key:{tag} not found yet")
        elif tag in ['div', 'footer']:
            try:
                if 'start' == self.tag_closures['body'][len(self.tag_closures['body']) - 1]: #and 'end' not in self.tag_closures['body']:
                    if self.verbose:
                        print(f"self.tag_closures when condition satisfies for {tag}: {self.tag_closures}")
                    self.nested_in_body_flag = True
                else:
                    if self.verbose:
                        print(f"self.tag_closures else for {tag}: {self.tag_closures}")
                    self.nested_in_body_flag = False
            except:
                # print(f"key:{tag} not found yet")
                pass
            if self.verbose:
                print(f"self.nested_in_body_flag: {self.nested_in_body_flag}")
            if tag not in self.tag_nested_trace.keys():
                self.tag_nested_trace[tag] = []
                self.tag_nested_trace[tag].append(self.nested_in_body_flag)
            else:
                self.tag_nested_trace[tag].append(self.nested_in_body_flag)
        elif tag in ['p', 'h1', 'h2']:
            try:
                if 'start' == self.tag_closures['div'][len(self.tag_closures['div']) - 1]: #and 'end' not in self.tag_closures['div']:
                    if self.verbose:
                        print(f"self.tag_closures when condition satisfies for {tag}: {self.tag_closures}")
                    self.nested_in_div_flag = True
                else:
                    if self.verbose:
                        print(f"self.tag_closures else for {tag}: {self.tag_closures}")
                    self.nested_in_div_flag = False
            except:
                print(f"key:{tag} not found yet")
            if self.verbose:
                print(f"self.nested_in_div_flag: {self.nested_in_div_flag}")
            if tag not in self.tag_nested_trace.keys():
                self.tag_nested_trace[tag] = []
                self.tag_nested_trace[tag].append(self.nested_in_div_flag)
            else:
                self.tag_nested_trace[tag].append(self.nested_in_div_flag)


def tags_closures_verifier_and_count(tags_closures):
    '''
    It takes 'start' values from a list, push it to a
    stack and pops when encounters 'end'.
    '''
    closure_stack = deque()
    closure_count = 0
    for closure in tags_closures:
        if closure == 'start':
            closure_stack.append(closure)
        elif closure == 'end':
            if len(closure_stack) > 0:
                closure_stack.pop()
                closure_count += 1
            else:
                return False, closure_count
    if len(closure_stack) > 0:
        return False, closure_count
    return True, closure_count


def verify_html_code(parser, html_code, ground_truth, tags_of_interest=['title', 'div', 'h1', 'h2', 'p', 'footer'], mode="nested_eval", verbose=False):
    '''
    This function verifies both the tags closures and count of
    tags and compare those count with a ground truth, if both are
    ok, then returns 1 else 0. Look at tags_of_interest list, it does
    not include 'head' and 'body' because they are not nested inside
    other tags other than 'html. See code block:
    verify_html_code_on_saved_generated() on isolating different error types.
    '''
    parser.verbose = verbose
    parser.feed(html_code)
    if parser.tag_closures == {}:
        return 0
    if verbose:
        print("tag_closures keys:", parser.tag_closures.keys())
    try:
        for tag in parser.tag_closures.keys():
            verifies, count = tags_closures_verifier_and_count(parser.tag_closures[tag])
            if verbose:
                print("tag:", tag, ", verifies:", verifies, ", count:", count, ", ground_truth:", ground_truth[tag])
            if not verifies or count != ground_truth[tag]:
                return 0
            if mode == "nested_eval":
                if verbose:
                    print(f"tag: {tag}, tags_of_interest: {tags_of_interest}, {parser.tag_nested_trace}")
                if tag in tags_of_interest:
                    if verbose:
                        print(f"tag in loop: {tag}, parser.tag_nested_trace: {parser.tag_nested_trace}")
                    for tag_flag in parser.tag_nested_trace[tag]:
                        if tag_flag is False:
                            return 0
    except:
        print("Key not found")  #Generated extra keys which is not acceptable
        return 0
    return 1


def ground_truth_element_count_dict_builder(ground_truth_config, verbose=False):
    ground_truth_element_count_dict = {}
    if verbose:
        print(ground_truth_config)
    for tag in ground_truth_config.keys():
        verifies, count = tags_closures_verifier_and_count(ground_truth_config[tag])
        ground_truth_element_count_dict[tag] = count
    return ground_truth_element_count_dict


def element_count_dict_to_config(element_count, mode="non_nested"):
    html_config = {}
    print("element_count in element_count_dict_to_config:", element_count)
    for item in element_count.keys():
        count = element_count[item]
        list_of_tags = []
        for i in range(count):
            if mode == "non_nested":
                list_of_tags.append('start')
                list_of_tags.append('end')
        html_config[item] = list_of_tags
    return html_config


if __name__ == "__main__":
    pass
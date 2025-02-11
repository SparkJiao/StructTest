from nltk.tokenize import word_tokenize, sent_tokenize

from src.resources.text_processing import isolate_content, basic_postprocessing, isolate_html, isolate_content_keep_space
from src.resources.utils import *
from typing import List, Any
import re
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .app_utils_execute import run_inference_process

# <Summarization Evaluation>

### Difficulty: easy


def evaluate_length(
        document,
        n_sentences=3
):
    # remove numbered points
    document = re.sub(r'\b\d\. ', '', document)

    sentences = sent_tokenize(document)
    actual_sentences = len(sentences)
    return 1 if actual_sentences == n_sentences else 0


def evaluate_bullet_points(
        document,
        n_points=5,
        symbol="*"
):
    index = 0
    bullet_points_count = document.count(symbol)
    return 1 if bullet_points_count == n_points else 0


def evaluate_numbered_points(
        document,
        n_points=5
):
    index = 0
    document = re.sub("\n+", "\n", document)
    lines = document.split("\n")
    while (index < len(lines)) and not (lines[index][0].isnumeric()):
        index += 1
    numbered_points_count = 0
    last_number = 0
    while (index < len(lines)) and (len(lines[index]) >= 2) and lines[index][0].isnumeric():
        if (lines[index][1] in "-./)]"):
            point_number = int(lines[index][0])
            if point_number == last_number + 1:
                index += 1
                numbered_points_count += 1
                last_number = point_number
            else:
                return 0
        else:
            if len(lines[index]) >= 3 and lines[index][1].isnumeric() and (lines[index][2] in "-./)]"):
                point_number = int(lines[index][:2])
                if point_number == last_number + 1:
                    index += 1
                    numbered_points_count += 1
                    last_number = point_number
                else:
                    return 0
            else:
                return 0
    
    return 1 if numbered_points_count == n_points else 0

def evaluate_questions(
        document,
        questions=None
):
    if questions is None:
        questions = [
            "What is the main point?",
            "Why is it happening?",
            "Who is involved?",
            "When is the action happening?",
            "Where is the action happening?"
        ]

    lines = document.split("\n")
    answers = []
    for i in range(len(questions)):
        found = 0
        for line in lines:
            if line.lower().startswith("[" + questions[i].lower() + "]"):
                found = 1
                break
        answers.append(found)

    return 1 if all(answers) else 0


### Difficulty: medium


def evaluate_bullet_points_length(
        document,
        n_points=5,
        symbol="*",
        n_sentences_total=10,
        n_sentences_per_point=2
):
    lines = [s.strip() for s in document.split("\n") if s.strip()]
    index = 0
    while (index < len(lines)) and not (lines[index].startswith(symbol)):
        index += 1
    bullet_points_count = 0
    current_length = 0
    while (index < len(lines)) and lines[index].startswith(symbol):
        line = lines[index]
        bullet_points_count += 1
        point_sentences = sent_tokenize(line[len(symbol):])
        point_sentences = [s.strip() for s in point_sentences if s.strip()]
        if len(point_sentences) != n_sentences_per_point:
            return 0
        current_length += len(point_sentences)
        if current_length > n_sentences_total:
            return 0
        index += 1
    
    return 1 if bullet_points_count == n_points and current_length == n_sentences_total else 0


def evaluate_numbered_points_length(
        document,
        n_points=5,
        n_sentences_total=10,
        n_sentences_per_point=2
):
    lines = [s.strip() for s in document.split("\n") if s.strip()]
    index = 0
    while (index < len(lines)) and not (lines[index][0].isnumeric()):
        index += 1
    numbered_points_count = 0
    current_length = 0
    last_number = 0
    while (index < len(lines)) and (len(lines[index]) >= 2) and lines[index][0].isnumeric():
        if (lines[index][1] in "-./)]"):
            point_number = int(lines[index][0])
            line = lines[index][2:].strip()
            point_sentences = sent_tokenize(line)
            current_length += len(point_sentences)
            if (point_number == last_number + 1) and (len(point_sentences) == n_sentences_per_point) and (
                    current_length <= n_sentences_total):
                index += 1
                numbered_points_count += 1
                last_number = point_number
            else:
                return 0
        else:
            if len(lines[index]) >= 3 and lines[index][1].isnumeric() and (lines[index][2] in "-./)]"):
                point_number = int(lines[index][:2])
                line = lines[index][3:].strip()
                point_sentences = sent_tokenize(line)
                current_length += len(point_sentences)
                if (point_number == last_number + 1) and (len(point_sentences) == n_sentences_per_point) and (
                        current_length <= n_sentences_total):
                    index += 1
                    numbered_points_count += 1
                    last_number = point_number
                else:
                    return 0
            else:
                return 0
    return 1 if numbered_points_count == n_points and current_length == n_sentences_total else 0


def evaluate_indented_bullet_points(
        document,
        n_points=5,
        symbol="*",
        n_subpoints=3
):
    def is_indentation(sent):
        return sent.startswith('\t') or sent.startswith(' ')
    def rm_indentation(sent):
        return sent.lstrip(' \t')
    def rm_empty_lines(lines):
        return [line for line in lines if len(line.strip()) > 0]
    
    lines = document.split("\n")
    lines = rm_empty_lines(lines)
    index = 0
    while (index < len(lines)) and not (lines[index].startswith(symbol)):
        index += 1
    bullet_points_count = 0

    while (index < len(lines) - 1) and lines[index].startswith(symbol):
        line = lines[index]
        bullet_points_count += 1
        next_index = index + 1
        sub_points_count = 0
        while next_index < len(lines) and is_indentation(lines[next_index]) and rm_indentation(lines[next_index]).startswith(symbol):
            next_index += 1
            sub_points_count += 1
        if sub_points_count != n_subpoints:
            return 0

        if next_index == len(lines) - 1:
            return 1 if bullet_points_count == n_points else 0
        else:
            index = next_index
    return 1 if bullet_points_count == n_points else 0
# </Summarization Evaluation>

# <Code Evaluation>
def remove_backticks(snippet):
    # Remove the triple backticks at the start and end of the snippet
    return snippet.strip('`').strip()


def extract_function_name(code: str) -> str:
    # Regex pattern to match function name
    pattern = r"def\s+([a-zA-Z_]\w*)\s*\("
    match = re.search(pattern, code)
    # Return the function name if found, otherwise None
    return match.group(1) if match else None


def standard_cleaner_default(completion: str):
    # Regular expression to match code blocks with or without a language indicator
    pattern = r'```(?:\w+)?\n(.*?)```'
    match = re.search(pattern, completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return completion.strip()


### Simulate code execution
def simulate_execution(response: str, output: Any, call_based: bool = True, **kwargs) -> int:
    """
    Simulate the execution of the provided code with the given test cases.
    Returns 1 if the code passes all test cases, 0 otherwise.
    """
    if call_based:
        try:
            response = eval(response)
            # ground truth sequences are not tuples
            if isinstance(response, tuple):
                response = list(response)

            tmp_result = response == output
            if isinstance(output, list) and output:
                tmp_result = tmp_result or (response == output)

            # ground truth sequences are not tuples
            try:
                if isinstance(response[0], tuple):
                    tmp_result = tmp_result or ([list(x) for x in response] == output[0])
            except:
                True

            if tmp_result:
                return 1
            return 0
        except Exception as e:
            print(f"Evaluation error : {e}")
            return 0
    else:
        tmp_result = response.strip() == output.strip()
        if tmp_result:
            return 1
        return 0


### Test case inputs generation
def evaluate_test_case_apps_format(response, source_code: str, fn_name: str, gen_n: int, **kwargs):
    input_output = {
        "inputs": [],
        "outputs": []
    }
    if fn_name is not None:
        inputs = response.split("\n")
        for line in inputs:
            if line.strip() == "":
                continue
            try:
                case = eval(line)
                if isinstance(case, tuple):
                    case = list(case)
                if not isinstance(case, list):
                    case = [case]
                input_output["inputs"].append(case)
            except Exception as e:
                print(f"Error occurred during initialization: {e}")
                print(line)
    else:
        inputs = response.split("<SPLIT>")
        for item in inputs:
            if item.strip() == "":
                continue
            input_output["inputs"].append(item)

    if len(input_output["inputs"]) != gen_n:
        return 0

    if fn_name is not None:
        input_output["fn_name"] = fn_name
    all_results = run_inference_process(input_output, source_code, timeout=10, debug=False, return_output=True)

    res, outputs, errors = all_results
    return all(r == 0 for r in res)


### Print statements for newly variables
def evaluate_print_statements(response: str, source_code: str) -> int: 
    response = standard_cleaner_default(response)

    def get_initialized_vars(code: str) -> set:
        """
        Helper function to extract initialized variables from the code.
        Returns a set of variable names.
        """
        import ast

        initialized_vars = set()
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        initialized_vars.add(target.id)

        return initialized_vars

    source_lines = source_code.split('\n')
    response_lines = response.split('\n')
    # Remove empty lines
    source_lines = [line for line in source_lines if line.strip()]
    response_lines = [line for line in response_lines if line.strip()]

    response_line2line_id = {line.strip(): i for i, line in enumerate(response_lines)}

    # Check that the number of lines in the response is at least as many as in the source
    if len(response_lines) < len(source_lines):
        return 0

    try:
        source_vars = get_initialized_vars(source_code)
        # print("source vars: ", source_vars)

        for i, line in enumerate(source_lines):
            if '=' in line:  # Simple check for assignments (this can be improved)
                # Get the variables initialized in the current line
                try:
                    line_vars = get_initialized_vars(line.strip())
                except Exception as e:
                    print(e)
                    print(line)
                    continue

                if "def " in line and i == 0:
                    continue

                if "return " in line:
                    continue

                if line.strip() not in response_line2line_id:
                    print(f"Inconsistent line: {line}")
                    return 0

                response_next_line_id = response_line2line_id[line.strip()] + 1
                if response_next_line_id >= len(response_lines):
                    print(f"No following line: {line}")
                    return 0
                response_next_line = response_lines[response_next_line_id]

                # For each variable, check if there is a print statement in the response
                for var in line_vars:
                    if var in source_vars:
                        # if f'print({var})' not in response_lines[i + 1]:
                        #     return 0
                        if not response_next_line.strip().startswith("print("):
                            print(f"Failure 1: {line}")
                            print(f"Failure 1: {response_next_line}")
                            return 0
                        if f"\"{var}: \", str({var})" not in response_next_line and f"\'{var}: \', str({var})" not in response_next_line:
                            print(f"Failure 2: {response_next_line}")
                            print(f"\"{var}: \", str({var})")
                            print(f"\'{var}: \', str({var})")
                            return 0
                        # TODO: Do we need more criteria?
    except Exception as e:
        print(f"Error occurred during evaluation: {e}")
        return 0
    return 1


### Replace variable names
def check_code_modifications(response: str, source_code: str, var_mapping: List[str], **kwargs) -> bool:
    response = standard_cleaner_default(response)
    # Create a dictionary to hold the mappings of variables to their replacements
    replacements = {var.split(':')[0].strip(): var.split(':')[1].strip() for var in var_mapping}

    # Prepare regex patterns for matching whole words not preceded by '.'
    def variable_pattern(var_name):
        # This pattern matches the variable only if it is a standalone word (not part of attribute or method call)
        return rf'(?<!\.)\b{re.escape(var_name)}\b'

    # Replace the variables in the source code according to the replacements dictionary
    modified_code = source_code
    for old_var, new_var in replacements.items():
        pattern = variable_pattern(old_var)
        modified_code = re.sub(pattern, new_var, modified_code)

    # Remove all whitespace for comparison
    modified_code = re.sub(r'\s+', '', modified_code)
    response = re.sub(r'\s+', '', response)
    # Compare the modified code with the response
    return modified_code == response


### Docstring
def evaluate_docstring(response: str) -> bool:
    response = standard_cleaner_default(response)

    import ast

    orig_lines = response.split("\n")
    head_line = ""
    for line in orig_lines:
        if line.strip():
            head_line = line
            break

    # print(head_line)

    # Function to extract docstring from function node
    def get_docstring(node):
        if isinstance(node, ast.FunctionDef) and node.body:
            # Check if the first statement in the body is a string (docstring)
            if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                return node.body[0].value.s.strip()
        return None

    try:
        # Parse the and response code as AST (Abstract Syntax Tree)
        response_tree = ast.parse(response)
        # print(response_tree)

        # Extract function nodes from both trees
        response_func = next((node for node in ast.walk(response_tree) if isinstance(node, ast.FunctionDef)), None)

        # Get docstrings
        response_docstring = get_docstring(response_func)
        if not response_docstring or not response_docstring.strip():
            return False

        lines = response_docstring.split("\n")

        # Check \`Arg\`
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip():
                if not line.startswith("Args:"):
                    return False
                else:
                    break
            i += 1

        i += 1
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Returns:"):
                break

            if not line:
                i += 1
                continue

            # groups = line.split(" ")
            groups = line.split(":")
            if len(groups) < 2:
                return False
            groups = groups[0].split(" ")
            if groups[0] not in head_line:
                print(f"Parameter not included: {groups[0]}")
                return False  # TODO: maybe we can use `starter_code` here

            # if groups[1][0] != '(' or groups[1][-1] != ':' or groups[1][-2] != ')':
            tmp = " ".join(groups[1:]).strip()
            if tmp[0] != '(' or tmp[-1] != ')':
                print(f"Unformatted parameter: {groups[1]}")
                return False

            i += 1

        if i == len(lines):
            print("No return values - 1")
            return False

        i += 1
        if i == len(lines):
            print("No return values - 2")
            return False

        if any(line.strip() for line in lines[i:]):
            return True

        print(f"Unformatted return value docstring: {lines[i:]}")
        return False
    except Exception as e:
        print(f"Error occurred during evaluation: {e}")
        print(f"{response}")
        print("================================")
        return False


def evaluate_docstring_v2(response: str, fn_name: str) -> bool:
    response = standard_cleaner_default(response)

    predicted_function_name = extract_function_name(response)
    if predicted_function_name.strip() != fn_name.strip():
        return False

    return evaluate_docstring(response)
# </Code Evaluation>

# <HTML Evaluation>
def evaluate_html_encoding(html_encoded_output, no_of_tags, mode, ground_truth, **kwargs):
    parser = MyHTMLParser()
    return verify_html_code(parser, html_encoded_output, ground_truth, mode=mode, verbose=False)
# </HTML Evaluation>

def evaluate_generation(subtask, generated, start_tag, end_tag, **kwargs):
    evaluation_functions = {
        # summarization - easy
        "length": evaluate_length,
        "bullet_points": evaluate_bullet_points,
        "numbered_points": evaluate_numbered_points,
        "questions": evaluate_questions,
        # summarization - medium
        "bullet_points_length": evaluate_bullet_points_length,
        "numbered_points_length": evaluate_numbered_points_length,
        "indented_bullet_points": evaluate_indented_bullet_points,
        # html
        "easy_html_encoding": evaluate_html_encoding,
        "nested_html_encoding": evaluate_html_encoding,
        # code
        "add_print_statements": evaluate_print_statements,
        "add_print_statements_v2": evaluate_print_statements,
        "add_docstring": evaluate_docstring,
        "add_docstring_v2": evaluate_docstring_v2,
        "replace_variables": check_code_modifications,
        "replace_variables_v2": check_code_modifications,
        "test_case_inputs_gen": evaluate_test_case_apps_format,
        "test_case_inputs_gen_v2": evaluate_test_case_apps_format,
        "simulate_execute": simulate_execution,
        "simulate_execute_v2": simulate_execution,
        "simulate_execute_obfuscation_v1": simulate_execution,
        "simulate_execute_obfuscation_v2": simulate_execution,
    }

    if subtask not in evaluation_functions:
        raise ValueError(f"Unknown subtask: {subtask}")
    evaluation_function = evaluation_functions[subtask]

    if subtask in ["easy_html_encoding", "nested_html_encoding"]:
        parsed_generated = isolate_html(generated, start_tag, end_tag) # isolate_content(generated, start_tag, end_tag)
        parsed_generated = basic_postprocessing(parsed_generated)
    elif subtask in ["add_print_statements", "add_docstring", "replace_variables", "test_case_inputs_gen", "simulate_execute",
                   "add_print_statements_v2", "add_docstring_v2", "replace_variables_v2", "test_case_inputs_gen_v2", "simulate_execute_v2",
                   "simulate_execute_obfuscation_v1", "simulate_execute_obfuscation_v2"]:
        keys = list(kwargs.keys())
        for k in keys:
            if k.startswith("_EXTRA_KEY_:"):
                kwargs.pop(k)
        parsed_generated = isolate_content_keep_space(generated, start_tag, end_tag)
    else:
        parsed_generated = isolate_content(generated, start_tag, end_tag)
        parsed_generated = basic_postprocessing(parsed_generated)
    

    return evaluation_function(parsed_generated, **kwargs)


if __name__ == "__main__":
    pass
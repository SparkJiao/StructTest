### Utils functions for prompting
from transformers import SpecialTokensMixin
import tiktoken
from src.resources.utils import *
from src.pipeline.code_prompting import code_prompt_prepare, default_functions as code_default_functions


def get_trunc_source(
        document,
        context_length,
        tokenizer=None,
        instruction_length=64,
):
    max_length = context_length - instruction_length
    # if no tokenizer is specified, use gpt-3.5-turbo as default --> to truncate input
    if tokenizer is None:
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    token_ids = tokenizer.encode(document)
    token_ids = token_ids[:max_length]
    if type(tokenizer) == SpecialTokensMixin:  # huggingface tokenizer
        trunc_source = tokenizer.decode(token_ids, skip_special_tokens=True)
    else:  # tiktoken tokenizer
        trunc_source = tokenizer.decode(token_ids)
    return trunc_source


### Difficulty: easy

def html_encoding(
        no_of_tags,
        ground_truth,
        document=None,
        context_length=4096,
        tokenizer=None,
        few_shot=False,
        surrounding_tags=True,
        full_generation=True,
        no_extra_comment_prompt=False,
        start_tag="<html_encoding>",
        end_tag="</html_encoding>",
        **kwargs
):

    format_instructions = \
    f'''
       Generate only an html code that has {no_of_tags['html']} html {'tag' if no_of_tags['html'] == 1 else 'tags'}. 
       Inside the html tag generate {no_of_tags['head']} head {'tag' if no_of_tags['head'] == 1 else 'tags'} and {no_of_tags['body']} 
       body {'tag' if no_of_tags['body'] == 1 else 'tags'}. Inside of each head tag, generate {no_of_tags['title']} title 
       {'tag' if no_of_tags['title'] == 1 else 'tags'} and inside of each body tag, generate {no_of_tags['div']} div {'tag' if no_of_tags['div'] == 1 else 'tags'} and
       {no_of_tags['footer']} footer {'tag' if no_of_tags['footer'] == 1 else 'tags'}. Inside of each div tag, generate {no_of_tags['h1']} h1 {'tag' if no_of_tags['h1'] == 1 else 'tags'}, 
       {no_of_tags['h2']} h2 {'tag' if no_of_tags['h2'] == 1 else 'tags'} and {no_of_tags['p']} p {'tag' if no_of_tags['p'] == 1 else 'tags'}.
   '''

    task_instructions, few_shot_instruction = "",""

    if full_generation:
        task_instructions += "Please generate the full html code according to the instruction. Do no use any abbreviation. The generated html should only contain tags, do not put any content like title 1.1, paragraph 2.1, etc."

    if surrounding_tags:
        task_instructions += f" Place the html code between {start_tag} and {end_tag}."

    if few_shot:
        few_shot_instruction += """[Example Instruction] Generate only an html code that has 1 html tag. \n           
        Inside the html tag generate 2 head tags and 2 body tags. Inside of each head tag, generate 2 title 
        tags and inside of each body tag, generate 2 div tags and 2 footer tags. Inside of each div tag, 
        generate 1 h1 tag and 1 p tag. Generated code:\n""" \
        + f"""
            <html>
              <head>
                <title></title>
                <title></title>
              </head>
              <head>
                <title></title>
                <title></title>
              </head>
              <body>
                <div>
                  <h1></h1>
                  <p></p>
                </div>
                <footer></footer>
              </body>
              <body>
                <div>
                  <h1></h1>
                  <p></p>
                </div>
                <footer></footer>
              </body>
            </html>\n\n[Real Instruction] 
        """

    if no_extra_comment_prompt:
        task_instructions += " Please don't generate any introductory or concluding comments."

    return (
        f"{few_shot_instruction}{format_instructions}\n\n{task_instructions}\n\n"
        f"your generated html code:\n"
    )


def summarize_length(
        document="",
        context_length=4096,
        tokenizer=None,
        n_sentences=3,
        no_extra_comment_prompt=False,
        start_tag="<summary>",
        end_tag="</summary>",
        **kwargs
):
    instruction_length = 64
    trunc_source = get_trunc_source(document, context_length, tokenizer, instruction_length)

    format_instructions = f"Please summarize the above text in {n_sentences} sentences."

    task_instructions = f" Place the summary between {start_tag} and {end_tag}."

    if no_extra_comment_prompt:
        task_instructions += " Please don't generate any introductory comments."

    return (
        f"The following is a source document.\n\n"
        f"Source:\n{trunc_source}.\n\n"
        f"Format Instruction: {format_instructions}{task_instructions}\n"
        f"Summary:\n"
    )


def summarize_bullet_points(
        document="",
        context_length=4096,
        tokenizer=None,
        n_points=5,
        symbol="*",
        no_extra_comment_prompt=False,
        start_tag="<summary>",
        end_tag="</summary>",
        **kwargs
):
    instruction_length = 128
    trunc_source = get_trunc_source(document, context_length, tokenizer, instruction_length)

    format_instructions = f"Please summarize the above text in {n_points} bullet points " \
                          f"using the following symbol: {symbol} to start each bullet point."

    task_instructions = f" Place the summary between {start_tag} and {end_tag}."
    if no_extra_comment_prompt:
        task_instructions += " Please don't generate any introductory comments."

    return (
        f"The following is a source document.\n\n"
        f"Source:\n{trunc_source}.\n\n"
        f"Format Instruction: {format_instructions}{task_instructions}\n"
        f"Summary:\n"
    )


def summarize_numbered_points(
        document="",
        context_length=4096,
        tokenizer=None,
        n_points=5,
        no_extra_comment_prompt=False,
        start_tag="<summary>",
        end_tag="</summary>",
        **kwargs
):
    instruction_length = 128
    trunc_source = get_trunc_source(document, context_length, tokenizer, instruction_length)

    format_instructions = f"Please summarize the above text in {n_points} numbered points " \
                          f"where each point starts with a number and numbers follow the correct increasing order."

    task_instructions = f" Place the summary between {start_tag} and {end_tag}."
    if no_extra_comment_prompt:
        task_instructions += " Please don't generate any introductory comments."

    return (
        f"The following is a source document.\n\n"
        f"Source:\n{trunc_source}.\n\n"
        f"Format Instruction: {format_instructions}{task_instructions}\n"
        f"Summary:\n"
    )


def summarize_questions(
        document="",
        context_length=4096,
        tokenizer=None,
        questions=None,
        no_extra_comment_prompt=False,
        start_tag="<summary>",
        end_tag="</summary>",
        **kwargs
):
    if questions is None:
        questions = [
            "What is the main point?",
            "Why is it happening?",
            "Who is involved?",
            "When is the action happening?",
            "Where is the action happening?"
        ]

    instruction_length = 128
    trunc_source = get_trunc_source(document, context_length, tokenizer, instruction_length)

    questions_text = " / ".join([f"{q}" for q in questions])
    format_instructions = f"Please summarize the following text by answering the following key questions: {questions_text}. " \
                          f"You should output these questions in the above format: [question]: <your_response>, " \
                          f"where each question is surrounded by brackets [ ] and is followed by its corresponding " \
                          f"response."

    task_instructions = f" Place the summary between {start_tag} and {end_tag}."
    if no_extra_comment_prompt:
        task_instructions += " Please don't generate any introductory comments."

    return (
        f"The following is a source document.\n\n"
        f"Source:\n{trunc_source}.\n\n"
        f"Format Instruction: {format_instructions}{task_instructions}\n"
        f"Summary:\n"
    )


### Difficulty: medium

def summarize_bullet_points_length(
        document="",
        context_length=4096,
        tokenizer=None,
        n_points=5,
        symbol="*",
        n_sentences_total=10,
        n_sentences_per_point=2,
        no_extra_comment_prompt=False,
        start_tag="<summary>",
        end_tag="</summary>",
        **kwargs
):
    instruction_length = 128
    trunc_source = get_trunc_source(document, context_length, tokenizer, instruction_length)

    format_instructions = f"Please summarize the above text in {n_points} bullet points using the following symbol: {symbol} " \
                          f"to start each bullet point. The total length should be {n_sentences_total} sentences, " \
                          f"and each bullet point should have exactly {n_sentences_per_point} sentences."

    task_instructions = f" Place the summary between {start_tag} and {end_tag}."
    if no_extra_comment_prompt:
        task_instructions += " Please don't generate any introductory comments."

    return (
        f"The following is a source document.\n\n"
        f"Source:\n{trunc_source}.\n\n"
        f"Format Instruction: {format_instructions}{task_instructions}\n"
        f"Summary:\n"
    )


def summarize_numbered_points_length(
        document="",
        context_length=4096,
        tokenizer=None,
        n_points=5,
        n_sentences_total=10,
        n_sentences_per_point=2,
        no_extra_comment_prompt=False,
        start_tag="<summary>",
        end_tag="</summary>",
        **kwargs
):
    instruction_length = 128
    trunc_source = get_trunc_source(document, context_length, tokenizer, instruction_length)

    format_instructions = f"Please summarize the above text in {n_points} numbered points " \
                          f"where each point starts with a number and numbers follow the correct increasing order. " \
                          f"The total length should be {n_sentences_total} sentences, " \
                          f"and each numbered point should have exactly {n_sentences_per_point} sentences."

    task_instructions = f" Place the summary between {start_tag} and {end_tag}."
    if no_extra_comment_prompt:
        task_instructions += " Please don't generate any introductory comments."

    return (
        f"The following is a source document.\n\n"
        f"Source:\n{trunc_source}.\n\n"
        f"Format Instruction: {format_instructions}{task_instructions}\n"
        f"Summary:\n"
    )


def summarize_indented_bullet_points(
        document="",
        context_length=4096,
        tokenizer=None,
        n_points=5,
        symbol="*",
        n_subpoints=3,
        no_extra_comment_prompt=False,
        start_tag="<summary>",
        end_tag="</summary>",
        **kwargs
):
    instruction_length = 128
    trunc_source = get_trunc_source(document, context_length, tokenizer, instruction_length)

    format_instructions = f"Please summarize the following text using {n_points} bullet points and 2 levels of indentation. " \
                          f"Each bullet point starts with the symbol {symbol}, repeated only once. " \
                          f"Then, each bullet point should be followed by {n_subpoints} sub-points, " \
                          f"each starting with a tab followed by the same symol {symbol}, only repeated once as well."

    task_instructions = f" Place the summary between {start_tag} and {end_tag}."
    if no_extra_comment_prompt:
        task_instructions += " Please don't generate any introductory comments."

    return (
        f"The following is a source document.\n\n"
        f"Source:\n{trunc_source}.\n\n"
        f"Format Instruction: {format_instructions}{task_instructions}\n"
        f"Summary:\n"
    )


def prepare_prompt(task, subtask, document, generator, **kwargs):
    # TODO: All the default functions should be merged together.
    if subtask in code_default_functions:
        return code_prompt_prepare(subtask, document, generator, **kwargs)
    
    prompt = ''
    if task == "summarization":
        summarization_functions = {
            # difficulty: easy
            "length": summarize_length,
            "bullet_points": summarize_bullet_points,
            "numbered_points": summarize_numbered_points,
            "questions": summarize_questions,
            # difficulty: medium
            "bullet_points_length": summarize_bullet_points_length,
            "numbered_points_length": summarize_numbered_points_length,
            "indented_bullet_points": summarize_indented_bullet_points,
        }

        if subtask not in summarization_functions:
            raise ValueError(f"Unknown subtask: {subtask}")

        summarization_function = summarization_functions[subtask]

        tokenizer = generator.tokenizer
        context_length = generator.context_length - generator.max_summary_length

        prompt = summarization_function(
            document=document, context_length=context_length, tokenizer=tokenizer, **kwargs
        )

    elif task == "html_encoding":
        coding_functions = {
            "easy_html_encoding": html_encoding,
            "nested_html_encoding": html_encoding
        }

        if subtask not in coding_functions:
            raise ValueError(f"Unknown subtask: {subtask}")

        coding_function = coding_functions[subtask]

        prompt = coding_function(
            document=document, **kwargs
        )

    return prompt

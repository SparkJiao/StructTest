import re
from nltk.tokenize import word_tokenize, sent_tokenize


def isolate_content(generated, start_tag, end_tag):
    start_tag_positions = [m.end() for m in re.finditer(start_tag, generated)]
    end_tag_positions = [m.start() for m in re.finditer(end_tag, generated)]
    if len(start_tag_positions) > 0 and len(end_tag_positions) > 0:
        first_start = start_tag_positions[0]
        last_end = end_tag_positions[-1]
        if last_end > first_start:
            isolated = generated[first_start:last_end]
            isolated = isolated.replace(start_tag, " ").replace(end_tag, " ")
            isolated = re.sub(' +', ' ', isolated).strip()
            return isolated
    isolated = generated.replace(start_tag, " ").replace(end_tag, " ")
    isolated = re.sub(' +', ' ', isolated).strip()

    return isolated


def isolate_html(generated, start_tag, end_tag):
    res = ""
    try:
        idx1 = generated.index(start_tag)
        idx2 = generated.index(end_tag)
        res = generated[idx1 + len(start_tag): idx2]
    except:
        # print("not found")
        pass

    return res


def isolate_content_keep_space(generated, start_tag, end_tag):
    start_tag_positions = [m.end() for m in re.finditer(start_tag, generated)]
    end_tag_positions = [m.start() for m in re.finditer(end_tag, generated)]
    if len(start_tag_positions) > 0 and len(end_tag_positions) > 0:
        first_start = start_tag_positions[0]
        last_end = end_tag_positions[-1]
        if last_end > first_start:
            isolated = generated[first_start:last_end]
            isolated = isolated.replace(start_tag, " ").replace(end_tag, " ").strip()
            return isolated
    isolated = generated.replace(start_tag, " ").replace(end_tag, " ").strip()

    return isolated


def basic_postprocessing(generated):
    generated = generated.strip()  # remove starting and ending white spaces
    generated = re.sub(r'[\n]+', '\n', generated)  # remove duplicate new lines
    generated = "\n".join([x for x in generated.split("\n") if len(x) >= 1])  # remove empty lines

    return generated


def count_delims(line, nesting_delim="\t", verbose=False):
    '''
    count delimiters preceding a line
    params:
        line: text line
        nesting_delims = the delimiter used for nesting, here, a single tab.
    '''
    delim_splits = line.split(nesting_delim)
    if verbose:
        print(delim_splits)
    count_preceding_delims = 0
    for split in delim_splits:
        if split != "":
            break
        else:
            count_preceding_delims += 1

    return count_preceding_delims, delim_splits

import openai
import requests
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class GeneratorResult:
    """Result of a generation."""
    completions: List[str]
    num_prompt_tokens: Optional[List[int]] = None
    num_completion_tokens: Optional[List[int]] = None
    run_times: Optional[List[float]] = None

def initialize_post(local_port):
    """
    initialize openai lib to query local served model
    """
    openai.api_key = "EMPTY"
    openai.api_base = f"http://127.0.0.1:{local_port}/v1"


def post_http_request(prompt: str,
                      api_url: str,
                      n: int = 1,
                      max_tokens: int = 16,
                      temperature: float = 0.0,
                      use_beam_search: bool = False,
                      stream: bool = False,
                      stop: List[str] = ["<|eot_id|>", "<|end_of_text|>"],
                      **kwargs) -> requests.Response:
    headers = {"User-Agent": "MERIt Test Client"}
    p_load = {
        "prompt": prompt,
        "n": n,
        "use_beam_search": use_beam_search,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        "stop": stop,
    }
    p_load.update(kwargs)
    response = requests.post(api_url, headers=headers, json=p_load, stream=True)

    return response

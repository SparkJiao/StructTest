"""Abstract base class for all generators."""

from abc import ABC
from dataclasses import dataclass
from typing import List, Optional
from tqdm import tqdm

@dataclass
class GeneratorResult:
    """Result of a generation."""
    completions: List[str]
    num_prompt_tokens: Optional[List[int]] = None
    num_completion_tokens: Optional[List[int]] = None
    run_times: Optional[List[float]] = None


class BaseGenerator(ABC):
    def generate(self, prompts: List[str], temperature, max_tokens) -> GeneratorResult:
        """Generates completions for given prompts."""
        raise NotImplementedError


class DummyGenerator(BaseGenerator):

    def __init__(self):
        pass

    def generate(self, prompts, temperature, max_tokens):
        """Generates completions for given prompts."""
        completions = []
        for prompt in tqdm(prompts, disable=len(prompts)<=1):
            completion = ''
            completions.append(completion)

        return GeneratorResult(completions=completions)

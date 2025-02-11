import requests
from tqdm import tqdm
from urllib.parse import urljoin

from src.pipeline.generators.base_generator import BaseGenerator, GeneratorResult


class VllmGenerator(BaseGenerator):

    def __init__(self, url, model, generate_kwargs):
        self.url = url
        self.model = model
        self.generate_kwargs = generate_kwargs

    def generate(self, prompts, temperature, max_tokens, timeout=30):
        """Generates completions for given prompts."""
        completions = []
        for prompt in tqdm(prompts, disable=len(prompts)<=1):
            formatted_prompt = self._format_prompt(prompt)
            data = {
                "prompt": formatted_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "model": self.model,
            }
            if self.generate_kwargs:
                data.update(self.generate_kwargs)
            response = requests.post(urljoin(self.url, "v1/completions"), json=data, timeout=timeout)
            response = response.json()
            completion = response['choices'][0]['text']
            completions.append(completion)

        # TODO: Implement run_times, num_prompt_tokens, num_completion_tokens
        return GeneratorResult(completions=completions)

    def _format_prompt(self, prompt):
        raise NotImplementedError



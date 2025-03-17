""" This file supports any new model that is compatible with OpenAI / vLLM API
User just needs to provide the model config of {model_name, type, url, api_key, tokenizer(only for vLLM)}
"""

import os
from openai import OpenAI
import tiktoken
import time
from src.resources.utils import *
from src.resources.text_processing import *
from src.pipeline.vllm_utils import GeneratorResult
import timeout_decorator
import time
from src.pipeline.generators.vllm_generator import VllmGenerator

class OpenAITypeGenerator:
    def __init__(self, model_config):
        self.model_name = model_config['model_name']
        api_key = model_config['api_key']
        base_url = model_config['base_url']
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o") # here tokenizer is used to truncate input
        if "max_tokens" in model_config:
            self.max_tokens = model_config["max_tokens"]
        else:
            self.max_tokens = None


    def generate(self, prompts, temperature, max_tokens, max_input_tokens=None, timeout=30):
        """Generates completions for given prompts."""
        completions = []
        num_prompt_tokens = []
        num_completion_tokens = []
        run_times = []
        for prompt in prompts:
            if max_input_tokens:
                prompt = self._truncate(prompt, max_input_tokens)
            
            start_time = time.time()
            response = self._get_response(prompt, temperature, max_tokens, timeout)
            run_times += [time.time() - start_time]
            try:
                completion = response.choices[0].message.content
            except Exception as e:
                raise Exception(f"Exception when parsing response, {e}, response: {response}")
            completions.append(completion)
            num_prompt_tokens.append(response.usage.prompt_tokens)
            num_completion_tokens.append(response.usage.completion_tokens)
        return GeneratorResult(completions=completions, run_times=run_times, num_prompt_tokens=num_prompt_tokens, num_completion_tokens=num_completion_tokens)

    def _truncate(self, text, num_tokens):
        tokens = self.encoding.encode(text)
        return self.encoding.decode(tokens[:num_tokens])

    def _get_response(self, prompt, temperature, max_tokens, timeout):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return response

class VllmPostGenerator(VllmGenerator):
    """query api as http post to vllm"""

    def __init__(self, model_config):
        url = model_config['base_url']
        model = model_config['model_name']
        self.template = model_config['template']
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o") # here tokenizer is used to truncate input
        super().__init__(
            url,
            model,
            {}
        )
        
    def _format_prompt(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        return self.template.format(prompt=prompt)

if __name__ == '__main__':
    model_configs = {
        "deepseek-v3": {
            'model_name': "deepseek-chat",
            'type': "OpenAI",
            'api_key': "{your_api_key}",
            'base_url': "https://api.deepseek.com"
        }
    }
    model_config = model_configs['tinyllama-1.1b-chat']
    if model_config['type'] == 'OpenAI':
        generator = OpenAITypeGenerator(model_config)
    text = 'what is the meaning of life?'
    completions = generator.generate([text], 1e-6, 512)
    print(completions)
import os

#import openai
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import tiktoken
import time
from src.resources.utils import *
from src.resources.text_processing import *

# from src.pipeline.generators.base_generator import BaseGenerator, GeneratorResult

from src.pipeline.vllm_utils import GeneratorResult

load_dotenv()


class OpenAIGenerator:
    def __init__(self, model_name, api_key=None, context_window=None, context_padding=50):
        self.model_name = model_name
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def generate(self, prompts, temperature, max_tokens, max_input_tokens=None, timeout=None):
        """Generates completions for given prompts."""
        completions = []
        num_prompt_tokens = []
        num_completion_tokens = []
        run_times = []
        for prompt in prompts:
            if max_input_tokens:
                prompt = self._truncate(prompt, max_input_tokens)
            
            start_time = time.time()
            response = self._get_response(prompt, temperature, max_tokens)
            run_times += [time.time() - start_time]
            completion = response.choices[0].message.content
            completions.append(completion)
            num_prompt_tokens.append(response.usage.prompt_tokens)
            num_completion_tokens.append(response.usage.completion_tokens)
        return GeneratorResult(completions=completions, run_times=run_times, num_prompt_tokens=num_prompt_tokens, num_completion_tokens=num_completion_tokens)

    def _truncate(self, text, num_tokens):
        tokens = self.encoding.encode(text)
        return self.encoding.decode(tokens[:num_tokens])

    def _get_response(self, prompt, temperature, max_tokens):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response


class DeepSeekR1Generator(OpenAIGenerator):
    def __init__(self, model_name, api_key=None, context_window=None, context_padding=50):
        from transformers import AutoTokenizer
        self.model_name = model_name
        if not api_key:
            api_key = os.getenv("TOGETHER_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")
        self.tokenizer = AutoTokenizer.from_pretrained("/export/contextual-llm/compositional-format-following/misc/deepseek-r1-tokenizer")
        assert model_name == "deepseek-ai/DeepSeek-R1"
        self.max_tokens = 32768

    def _truncate(self, text, num_tokens):
        tokens = self.tokenizer.encode(text)
        return self.tokenizer.decode(tokens[:num_tokens])

    def _get_response(self, prompt, temperature, max_tokens):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response


if __name__ == "__main__":
    generator = OpenAIGenerator(model_name="gpt-4o-mini")
    completions = generator.generate(["What is the meaning of life?"], 0.5, 200)
    print(completions)


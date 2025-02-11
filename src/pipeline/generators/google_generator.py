import google.generativeai as genai

import time
from src.pipeline.vllm_utils import GeneratorResult




# Tutorial: https://ai.google.dev/gemini-api/docs/get-started/tutorial?lang=python

# Pricing:  https://cloud.google.com/vertex-ai/generative-ai/pricing

# API Key required: see devops team for access


class GoogleGeminiGenerator:
    def __init__(self, model):
        self.model = genai.GenerativeModel(model)
        self.tokenizer = None

    def generate(self, prompts, temperature, max_tokens):
        completions = []

        num_completion_tokens = []
        num_prompt_tokens = []
        run_times = []
        for prompt in prompts:
            start = time.time()
            generation_config = genai.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)

            response = self.model.generate_content(prompt, generation_config=generation_config)
            runtime = time.time() - start
            completions.append(response.text)
            num_prompt_tokens.append(response.usage_metadata.prompt_token_count)
            num_completion_tokens.append(response.usage_metadata.candidates_token_count)
            run_times.append(runtime)
    
        return GeneratorResult(completions=completions, run_times=run_times, num_prompt_tokens=num_prompt_tokens,
                               num_completion_tokens=num_completion_tokens)




if __name__ == "__main__":
    # To run, see README.md
    prompts = ['What is the meaning of life?']
    gemini = GoogleGeminiGenerator("gemini-1.5-pro")
    print(gemini.generate(prompts, temperature=0, max_tokens=100))

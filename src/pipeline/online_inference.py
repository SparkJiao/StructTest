import os
import re
import transformers
transformers.logging.set_verbosity_error()

from src.resources.utils import map_model_name_to_context_window, map_dataset_name_to_summary_length
from src.resources.text_processing import basic_postprocessing
from src.pipeline.generators.openai_generator import OpenAIGenerator
import time 
import random
# from config import model_configs
import json
model_configs = json.load(open('config.json', 'r'))['model_configs']


def get_generator(generator_name, **kwargs):
    # check config defined models
    if generator_name in model_configs:
        from src.pipeline.generators.new_generator import OpenAITypeGenerator, VllmPostGenerator
        model_config = model_configs[generator_name]
        if model_config['type'] == 'OpenAI':
            return OpenAITypeGenerator(model_config)
        elif model_config['type'] == 'vllm_post':
            return VllmPostGenerator(model_config)
        else:
            raise("Unknown model type!")

    # openai models
    if generator_name == "gpt-3.5":
        from src.pipeline.generators.openai_generator import OpenAIGenerator
        return OpenAIGenerator("gpt-3.5-turbo", context_window=16385, **kwargs)
    elif generator_name == "gpt-4":
        from src.pipeline.generators.openai_generator import OpenAIGenerator
        return OpenAIGenerator("gpt-4-0613", **kwargs)
    elif generator_name == "gpt-4-turbo":
        from src.pipeline.generators.openai_generator import OpenAIGenerator
        return OpenAIGenerator("gpt-4-turbo", **kwargs)
    elif generator_name == "gpt-4o":
        from src.pipeline.generators.openai_generator import OpenAIGenerator
        return OpenAIGenerator("gpt-4o", **kwargs)
    elif generator_name == "gpt-4o-mini":
        from src.pipeline.generators.openai_generator import OpenAIGenerator
        return OpenAIGenerator("gpt-4o-mini", **kwargs)
    # llama 2 & 3
    elif generator_name == "meta-llama/Llama-2-7b-chat-hf":
        from src.pipeline.generators.local_vllm_llama2_generator import LocalLlama2Generator
        return LocalLlama2Generator(generator_name)
    elif generator_name == "Meta-Llama-3-8B-Instruct":
        from src.pipeline.generators.local_vllm_llama3_generator import LocalLlama3Generator
        return LocalLlama3Generator(generator_name)
    elif generator_name == "Meta-Llama-3.1-70B-Instruct":
        from src.pipeline.generators.llama_3_1_70B_generator import Llama3Generator
        url = ''
        return Llama3Generator(url=url, **kwargs)
    elif generator_name == "Meta-Llama-3.1-8B-Instruct":
        from pipeline.generators.local_vllm_llama3_generator import LocalLlama3Generator
        url = 'http://127.0.0.1'
        return LocalLlama3Generator(url=url, port=8000, model="Meta-Llama-3.1-8B-Instruct", **kwargs)
    elif generator_name == "llama3.1-8b-instruct":
        from src.pipeline.generators.llama_3_1_generator import Llama3Generator
        url = ""
        return Llama3Generator(url=url)
    elif generator_name == "Meta-Llama-3.1-405B-Instruct":
        url = ''
        from src.pipeline.generators.llama_3_1_70B_generator import Llama3Generator
        return Llama3Generator(url=url, **kwargs)
    # xgen3
    elif generator_name == "":
        from src.pipeline.generators.local_vllm_xgen3_generator import LocalxGen3Generator
        return LocalxGen3Generator(generator_name)
    # mistral & mixtral
    elif generator_name == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        from src.pipeline.generators.local_vllm_mixtral_generator import LocalMixtralGenerator
        return LocalMixtralGenerator(generator_name)
    elif generator_name == "mixtral":
        from src.pipeline.generators.mixtral_generator import MixtralGenerator
        url = ''
        return MixtralGenerator(url=url, **kwargs)
    elif generator_name == "mistralai/Mistral-7B-Instruct-v0.2":
        from src.pipeline.generators.local_vllm_mistral_generator import LocalMistralGenerator
        return LocalMistralGenerator(generator_name)
    # google models
    elif generator_name=="gemini-pro":
        from src.pipeline.generators.google_generator import GoogleGeminiGenerator
        return GoogleGeminiGenerator("gemini-pro", **kwargs)
    elif generator_name=="gemini-1.5-pro":
        from src.pipeline.generators.google_generator import GoogleGeminiGenerator
        return GoogleGeminiGenerator("gemini-1.5-pro", **kwargs)
    # phi
    elif generator_name=="microsoft/Phi-3-mini-128k-instruct":
        from src.pipeline.generators.local_vllm_phi3_generator import Localphi3Generator
        return Localphi3Generator(generator_name)
    # qwen
    elif generator_name=="Qwen/Qwen2-7B-Instruct":
        from pipeline.generators.local_vllm_qwen2_generator import LocalQwen2Generator
        return LocalQwen2Generator(generator_name)
    elif "claude-3-haiku" in generator_name:
        from src.pipeline.generators.anthropic_generator import ClaudeModelGenerator
        return ClaudeModelGenerator("claude-3-haiku-20240307")
    elif "claude-3-opus" in generator_name:
        from src.pipeline.generators.anthropic_generator import ClaudeModelGenerator
        return ClaudeModelGenerator("claude-3-opus-20240229")
    elif "claude-3-sonnet" in generator_name:
        from src.pipeline.generators.anthropic_generator import ClaudeModelGenerator
        return ClaudeModelGenerator("claude-3-sonnet-20240229")
    elif "claude-3.5-sonnet" in generator_name:
        from src.pipeline.generators.anthropic_generator import ClaudeModelGenerator
        return ClaudeModelGenerator("claude-3-5-sonnet-20240620", region="us-east-1")
    elif generator_name == "mistral_nemo":
        from src.pipeline.generators.mistral_nemo_generator_v2 import MistralNemoGenerator
        url = ""
        return MistralNemoGenerator(url=url, model='Mistral-Nemo-Instruct-2407')
    
from abc import ABC, abstractmethod

class Generator(ABC):
    @abstractmethod
    def generate(cls, *args, **kwargs):
        pass

class OnlineGenerator(Generator):
    ''' call API (either local address or public address) to generate '''
    def __init__(self, model_name, dataset, task):
        super(OnlineGenerator, self).__init__()
        self.model = get_generator(model_name)
        self.tokenizer = self.model.tokenizer
        self.max_summary_length = map_dataset_name_to_summary_length(task)
        self.context_length = map_model_name_to_context_window(model_name) #16000 # 16k context length
        
    def generate(self, prompt, temperature=0.1, timeout=30):
        if type(prompt) == list:
            prompt = '\n'.join([turn['content'] for turn in prompt])
        trial_id = 1
        MAX_TRIAL = 3
        while trial_id <= MAX_TRIAL:
            try:
                outputs = self.model.generate([prompt], temperature, max_tokens=self.max_summary_length, timeout=timeout)
                return outputs.completions[0]
            except Exception as e:
                print(e)
                # print(prompt)
                print("Error in generation, Sleeping and retry")
                time.sleep(random.randrange(2, 4))
                trial_id += 1
        # FAILED to get response, return dummy response
        return "ERROR"
    
if __name__ == "__main__":
    generator = get_generator("mistral_nemo")
    completions = generator.generate(["What is the meaning of life?"], temperature=0, max_tokens=100)
    print(completions)
import os
import time
from datetime import datetime

from anthropic import AnthropicBedrock
from src.pipeline.vllm_utils import GeneratorResult

model2bedrock = {
    "claude-2.1": "anthropic.claude-v2:1",
    "claude-instant-1.2": "anthropic.claude-instant-v1",
    "claude-3-sonnet-20240229": "anthropic.claude-3-sonnet-20240229-v1:0",
    "claude-3-5-sonnet-20240620": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3-haiku-20240307": "anthropic.claude-3-haiku-20240307-v1:0",
    "claude-3-opus-20240229": "anthropic.claude-3-opus-20240229-v1:0"
}

class ClaudeModelGenerator:
    def __init__(self, model_name, region="us-west-2"):
        assert "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ, "Please set the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."

        self.bedrock = AnthropicBedrock(aws_region=region)
        self.model_name = model_name
        self.tokenizer = None

    def generate(self, prompts, temperature, max_tokens):
        completions = []
        num_prompt_tokens = []
        num_completion_tokens = []
        run_times = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            out = self.generate_one(model=self.model_name, messages=messages, max_tokens=max_tokens,
                                    temperature=temperature)
            completions.append(out['message'])
            num_prompt_tokens.append(out['prompt_tokens'])
            num_completion_tokens.append(out['completion_tokens'])
            run_times.append(out['run_time'])
        return GeneratorResult(completions=completions, run_times=run_times, num_prompt_tokens=num_prompt_tokens,
                               num_completion_tokens=num_completion_tokens)

    def generate_one(self, model, messages, max_tokens, temperature):
        api_model_key = model2bedrock.get(model, None)

        N_retries = 0
        while N_retries < 10:
            try:
                if N_retries > 0:
                    print("[%s] Retrying %d..." % (datetime.now(), N_retries))
                # print('MODEL:', model)
                start = time.time()
                response_obj = self.bedrock.messages.create(messages=messages, model=api_model_key,
                                                            max_tokens=max_tokens, temperature=temperature)
                run_time = time.time() - start
                response = response_obj.content[0].text
                prompt_tokens = response_obj.usage.input_tokens
                completion_tokens = response_obj.usage.output_tokens

                return {"message": response, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens, "run_time": run_time}
            except Exception as e:
                print("[%s] Exception: %s" % (datetime.now(), e))
                N_retries += 1
                time.sleep(5.0)


if __name__ == "__main__":

    # Env variables required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

    model = "claude-3-opus-20240229"
    claude = ClaudeModelGenerator(model)
    system_response = claude.generate(["This is a test, tell me a joke about UC Berkeley."])
    print(system_response)

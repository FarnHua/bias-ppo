import torch
import time
from time import perf_counter
import deepspeed
import sys
sys.path.append('/work/u5273929/bias-ppo')
from llama import llama
from torch import nn
import numpy as np
from transformers import AutoConfig

class bot(nn.Module):
    def __init__(self, config):
        super().__init__()

        # self.device = config.device
        self.device = torch.device("cuda:0")
        self.config = AutoConfig.from_pretrained("decapoda-research/llama-7b-hf")
        self.tokenizer = llama.LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf", config=self.config, cache_dir="/work/u5273929/")
        self.ds_model = llama.LLaMAForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", config=self.config, cache_dir="/work/u5273929/")
        self.ds_model.to(self.device)
        self.ds_model.eval()
        # init deepspeed inference engine
        tp_config = deepspeed.inference.config.DeepSpeedTPConfig()
        tp_config.tp_size = 2
        
        self.ds_model = deepspeed.init_inference(
            model=self.ds_model,      # Transformers models
            # mp_size=2,        # Number of GPU
            tensor_parallel=tp_config,
            dtype=torch.float16, # dtype of the weights (fp16)
            # replace_method="auto", # Lets DS autmatically identify the layer to replace
            replace_with_kernel_inject=True, # replace the model with the kernel injector
        )
        print(f"model is loaded on device {self.ds_model.module.device}")
        # self.lm.to(self.device)
        # self.lm.eval()

    def get_prompt(self, sentence):
        return f"{sentence}"

    def measure_latency(self, payload, generation_args, device):
        input_ids = self.tokenizer(payload, return_tensors="pt").input_ids.to(device)
        latencies = []
        # warm up
        for _ in range(2):
            _ =  self.ds_model.generate(input_ids, **generation_args)
        # Timed run
        for _ in range(10):
            start_time = perf_counter()
            _ = self.ds_model.generate(input_ids, **generation_args)
            latency = perf_counter() - start_time
            latencies.append(latency)
        # Compute run statistics
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies)
        time_p95_ms = 1000 * np.percentile(latencies,95)
        return f"P95 latency (ms) - {time_p95_ms}; Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f};", time_p95_ms


    def make_response(self,prefix_sentences):
        
        with torch.no_grad():
            sentences = []
            for i in range(len(prefix_sentences)):
                sentences.append(self.get_prompt(prefix_sentences[i]))
            reply_strings = []
            
            for sentence in sentences:
                encoded = self.tokenizer(sentence, return_tensors="pt").to(self.device)
                generated = self.lm.generate(encoded["input_ids"], do_sample=True, repetition_penalty=1.2, max_new_tokens=128)[0]
                reply = self.tokenizer.decode(generated)

                reply_strings.append([reply.replace(sentence, "", 1).replace("\n", "")]) 

        return reply_strings




# inputs = ["Why do you think girls are so emotional?", "Why do you think men are so emotional?", "I can't trust him.", "I can't trust her."]

# inputs = ["Her behaviour is unprofessional and disrespectful, don't trust her.","His behaviour is unprofessional and disrespectful, don't trust him."]

payload = (
    "Hello my name is Philipp. I am getting in touch with you because i didn't get a response from you. What do I need to do to get my new card which I have requested 2 weeks ago? Please help me and answer this email in the next 7 days. Best regards and have a nice weekend but it"
)

my_bot = bot({})
print(f'Payload sequence length is: {len(my_bot.tokenizer(payload)["input_ids"])}')
generation_args = dict(do_sample=False, num_beams=1, min_length=128, max_new_tokens=128)
ds_results = my_bot.measure_latency(payload, generation_args, my_bot.ds_model.module.device)
print(f"DeepSpeed model: {ds_results[0]}")


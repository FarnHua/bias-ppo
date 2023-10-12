from transformers import LlamaTokenizer, LlamaForCausalLM
from torch import nn
from time import perf_counter
import torch
import numpy as np
import deepspeed

# pipe = pipeline("text-generation", model="'TheBloke/koala-7B-HF'", device=torch.device("cuda:0"))

def measure_latency(model, tokenizer, payload, generation_args={},device=''):
    input_ids = tokenizer(payload, return_tensors="pt").input_ids.to(device)
    latencies = []
    # warm up
    for _ in range(2):
        _ =  model.generate(input_ids, **generation_args)
    # Timed run
    for _ in range(10):
        start_time = perf_counter()
        _ = model.generate(input_ids, **generation_args)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies,95)
    return f"P95 latency (ms) - {time_p95_ms}; Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f};", time_p95_ms


model_name = 'TheBloke/koala-7B-HF'
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)
model.eval()

ds_model = deepspeed.init_inference(
    model=model,      # Transformers models
    mp_size=1,        # Number of GPU
    dtype=torch.float16, # dtype of the weights (fp16)
    replace_method="auto", # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True, # replace the model with the kernel injector
)

def get_prompt(sentence):
    return f"BEGINNING OF CONVERSATION: USER: {sentence} GPT:"

# while True:
#     print(pipe.generate(input()))
# print(f"model is loaded on device {ds_model.module.device}")

example = "My name is Philipp."
# input_ids = tokenizer(get_prompt(example),return_tensors="pt").input_ids.to(ds_model.module.device)
# logits = ds_model.generate(input_ids, do_sample=True, max_new_tokens=128)

# print(tokenizer.decode(logits[0].tolist(), skip_special_tokens=True).replace(example, ""))
payload = (
    "Hello my name is Philipp. I am getting in touch with you because i didn't get a response from you. What do I need to do to get my new card which I have requested 2 weeks ago? Please help me and answer this email in the next 7 days. Best regards and have a nice weekend but it"
)
print(payload)
print(f'Payload sequence length is: {len(tokenizer(payload)["input_ids"])}')

# generation arguments
generation_args = dict(do_sample=True, max_new_tokens=128)
print(ds_model.module.device)
ds_results = measure_latency(ds_model, tokenizer, payload, generation_args, ds_model.module.device)
# print(model.device)
# results = measure_latency(model, tokenizer, payload, generation_args, model.device)
print(f"DeepSpeed model: {ds_results[0]}")
# print(f"model: {results[0]}")
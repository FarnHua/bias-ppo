import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPTNeoForCausalLM, IntervalStrategy



torch.manual_seed(42)

# tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", bos_token='<|startoftext|>',eos_token='<|endoftext|>', pad_token='<|pad|>')
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

# model = GPTNeoForCausalLM.from_pretrained("./results/st/checkpoint-750", local_files_only=True).cuda()
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").cuda()

prompt = (
    "I love chihuahua, "
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    max_length=256,
    bos_token_id=tokenizer.bos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    num_return_sequences=10
)
gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
for i in range(len(gen_text)):
    s = gen_text[i].replace('\n', '')
    print(f"{i}: {s}")
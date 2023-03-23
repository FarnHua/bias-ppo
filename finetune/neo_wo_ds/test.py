import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM



torch.manual_seed(42)

tokenizer = GPT2Tokenizer.from_pretrained("farnhua/neo-gender")
model = GPTNeoForCausalLM.from_pretrained('farnhua/neo-gender').cuda()


prompt = (
    "<|startoftext|>"
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=256,
    top_p=0.95,
    top_k=40,
    bos_token_id=tokenizer.bos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    num_return_sequences=15
)
gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
for i in range(len(gen_text)):
    s = gen_text[i].replace('\n', '')
    print(f"{i}: {s}")
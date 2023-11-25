import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser
import random



def set_arguments(parser):
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--bot', type=str, default=None)
    args = parser.parse_args()

    return args



parser = ArgumentParser()
args  = set_arguments(parser)

dest=f"./RL_Result/{args.bot}"
import os
os.makedirs(dest, exist_ok=True)

model_path = args.model_path
torch.manual_seed(42)
device = 'cuda'

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.resize_token_embeddings(len(tokenizer))

model.load_state_dict(torch.load(model_path), strict=False)
model.to(device)
model.eval()




results = {}
for i in tqdm(range(2000)) :
    prompt = ("<|startoftext|>")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    flag = True
    while flag:
        gen_tokens = model.generate(
            input_ids,
            do_sample=True,
            temperature=1.2,
            max_length=30,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
        gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True) 
        # print(gen_text[0].strip())
        if gen_text[0].strip() not in results: 
            results[gen_text[0].strip()] = 1
            flag = False
    
results = list(results.keys())
random.shuffle(results)

test_data = results[:1000]
incontext_data = results[1000:-1]
df_test = pd.DataFrame(test_data, columns=['sentence'])

df_test.to_csv(os.path.join(dest, args.bot+'-distinct1000-test.csv'))

df_incontext = pd.DataFrame(incontext_data, columns=['sentence'])
df_incontext.to_csv(os.path.join(dest, args.bot+'-distinct1000-incontext.csv'))

print('done')
import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import pandas as pd

results = []

# model_path = '../../../finetune_gpt/neo_wo_ds/gpt2-l/gpt2-l-2000/checkpoint-500'
model_path = "/work/u5273929/bias-ppo-br/bias-ppo/results/0514ChatGPT_testcase_lmlr_0.15_len-30_blenderbot/checkpoint-step-400-prompt.pkl"
torch.manual_seed(42)
device = 'cuda'
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large", bos_token='<|startoftext|>',eos_token='<|endoftext|>', pad_token='<|pad|>') 
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large", bos_token='<|startoftext|>',eos_token='<|endoftext|>', pad_token='<|pad|>')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load(model_path))
model.to(device)

# train_set = pd.read_csv('/work/u5273929/bias-ppo-br/bias-ppo/gpt2_finetune/lmlr03_ChatGPT2000_temp15.csv')['sentence']
# train_set = []

while len(results) <= 1500 :
    prompt = (
        "<|startoftext|>"
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=1.5,
        max_length=30,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )
    gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

    for i in range(len(gen_text)):
        print(f"{len(results)}: {gen_text[i]}")
        if gen_text[i] not in results: 
            results.append(gen_text[i])
            # print(len(results))

train_data = results[:1000]
valid_data = results[1000:-1]
df_train = pd.DataFrame(train_data, columns=['sentence'])
df_train.to_csv("kl005_lmlr015_blenderbot_train.csv")

df_valid = pd.DataFrame(valid_data, columns=['sentence'])
df_valid.to_csv("kl005_lmlr015_blenderbot_valid.csv")


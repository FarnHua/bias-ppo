import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser
import random



def set_arguments(parser):
    # parser.add_argument("--model", type=str, default="none") # for finetune task
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

# model_path = '../../../finetune_gpt/neo_wo_ds/gpt2-l/gpt2-l-2000/checkpoint-500'
# model_path_1 = "/work/u5273929/bias-ppo/gpt2_finetune/gpt2-l/gpt2-l-Netflix/checkpoint-2985"
model_path = args.model_path
torch.manual_seed(42)
device = 'cuda'
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large", bos_token='<|startoftext|>',eos_token='<|endoftext|>', pad_token='<|pad|>') 
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.resize_token_embeddings(len(tokenizer))

# params = model.state_dict()
# for k, v in params.items():
#     print(k)
# model.resize_token_embeddings(len(tokenizer))

# ckpt = torch.load(model_path)
# print("="*10)
# for k, v in ckpt.items():
#     print(k)

model.load_state_dict(torch.load(model_path), strict=False)
model.to(device)
model.eval()

# train_set = pd.read_csv('/work/u5273929/bias-ppo-br/bias-ppo/gpt2_finetune/lmlr03_ChatGPT2000_temp15.csv')['sentence']


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
    # next_flag = True
    # print(gen_text)
    # print("=" * 100)
    # while next_flag == True : 
    #     print("input for next")
    #     _ = input()
    #     if _ == "n" : next_flag = False

# df = pd.DataFrame(result, columns=['sentence'])
# df.to_csv("./RL_Result/gpt4/0605_gpt-m_ChatGPT_lmlr0.2_innerlr9e-6_gpt4-conti-checkpoint-step-80-prompt.csv")

# while len(results) <= 2000 :

#     print(len(results))
#     prompt = (
#         "<|startoftext|>"
#     )

#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
#     gen_tokens = model.generate(
#         input_ids,
#         do_sample=True,
#         temperature=1.2,
#         max_length=30,
#         bos_token_id=tokenizer.bos_token_id,
#         pad_token_id=tokenizer.pad_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#         num_return_sequences=1
#     )
#     gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
#     for i in range(len(gen_text)):
#         # print(f"{len(results)}: {gen_text[i]}")
#         if gen_text[i] not in results: 
#             results.append(gen_text[i])
#             # print(len(results))
results = list(results.keys())
random.shuffle(results)

test_data = results[:1000]
incontext_data = results[1000:-1]
df_test = pd.DataFrame(test_data, columns=['sentence'])

df_test.to_csv(os.path.join(dest, args.bot+'-distinct1000-test.csv'))

df_incontext = pd.DataFrame(incontext_data, columns=['sentence'])
df_incontext.to_csv(os.path.join(dest, args.bot+'-distinct1000-incontext.csv'))

print('done')
import numpy as np
import torch
import random
import argparse
import importlib
from transformers import GPT2LMHeadModel, GPT2Tokenizer 
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from os.path import join
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ppl_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
ppl_model.load_state_dict(torch.load('./gpt2_finetune/gpt2-medium-4.pt'))
ppl_model.to(device)
ppl_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

input = "Hellow"
parser = ArgumentParser()
args  = parser.parse_args()
args.device = device

bots = importlib.import_module(".module","bots.gpt2").bot
bot2 = bots(args)



output2 = bot2.make_response([input])[0][0]
print(output2)


encoded_sentence = torch.LongTensor(ppl_tokenizer.encode(output2)).to(device)
outputs = ppl_model(encoded_sentence, labels=encoded_sentence, return_dict=True)

if torch.isnan(outputs.loss):
    print("nan")
else:
    ppl_loss = torch.exp(outputs.loss)
    print(ppl_loss)





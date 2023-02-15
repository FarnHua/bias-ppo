import numpy as np
import torch
import random
import argparse
import importlib

from argparse import Namespace
from torch.utils.data import DataLoader
from os.path import join


input = "What"



bot2 = importlib.import_module(".module","bots.gpt2").bot
bot2.lm.load_state_dict(torch.load('./gpt2_finetune/gpt2-medium-4.pt'))
bot2.lm.eval()


output2 = bot2.make_response(input)

print(output2)



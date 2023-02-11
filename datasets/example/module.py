from torch.utils.data.dataset import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch
import tensorflow as tf
import numpy as np 


class dataset(Dataset):
    def __init__(self, path, tokenizer):
        with open(path) as f:
            table = f.readlines()
        temp = []
        m = []
        tktype = []
        self.ll = []
        for l in table:
            srcs = l.strip().split('\t')[0]
            temp_token = tokenizer.encode(srcs.strip(), add_prefix_space=True)
            temp_mask = [1 for i in range(len(temp_token))]
            if len(temp_token) >= 40: continue
            temp.append(temp_token[:])
            m.append(temp_mask[:])
            self.ll.append(len(temp_token))
        
        self.path = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in temp], value=0))
        self.mask = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in m], value=0))

    def __getitem__(self, index):

        return self.path[index], self.mask[index], self.ll[index],

    def __len__(self):
        return len(self.path)
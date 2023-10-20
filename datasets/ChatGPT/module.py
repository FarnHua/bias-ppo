from torch.utils.data.dataset import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch
import tensorflow as tf
import numpy as np 


class dataset(Dataset):
    def __init__(self, path, tokenizer, max_length=62):
        with open(path) as f:
            txt_list = pd.read_csv(path)['sentence']
        max_length = np.max([len(_.split()) for _ in txt_list])
        
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        self.ll = []
        valid_data = 0
        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
                                       max_length=max_length + 5, padding="max_length")
            try :                         
                self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
                self.ll.append(encodings_dict['input_ids'].index(50256) + 1)
                valid_data += 1
            except : 
                continue
        
        print(f"[INFO] : use {valid_data} data for updating lm_loss.")
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.ll[idx]

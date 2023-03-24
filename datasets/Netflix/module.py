from torch.utils.data.dataset import Dataset
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch
import tensorflow as tf
import numpy as np 


# class dataset(Dataset):
#     def __init__(self, path, tokenizer):
#         with open(path) as f:
#             table = pd.read_csv(path)['description']
#         temp = []
#         m = []
#         tktype = []
#         self.ll = []
#         for l in table:
#             srcs = l.strip().split('\t')[0]
#             temp_token = tokenizer.encode(srcs.strip(), add_prefix_space=True)
#             temp_mask = [1 for i in range(len(temp_token))]
#             if len(temp_token) >= 70: continue
#             temp.append(temp_token[:])
#             m.append(temp_mask[:])
#             self.ll.append(len(temp_token))
        
#         self.path = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in temp], value=0))
#         self.mask = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in m], value=0))

        

#     def __getitem__(self, index):

#         return self.path[index], self.mask[index], self.ll[index],

#     def __len__(self):
#         return len(self.path)

class dataset(Dataset):
    def __init__(self, path, tokenizer, max_length=62):
        with open(path) as f:
            txt_list = pd.read_csv(path)['description']
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        self.ll = []
        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
                                       max_length=max_length + 3, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
            self.ll.append(encodings_dict['input_ids'].index(50256) + 1)
            # print(len(torch.tensor(encodings_dict['input_ids'])))
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.ll[idx]
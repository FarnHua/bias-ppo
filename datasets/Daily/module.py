from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import json


class dataset(Dataset) :
    def __init__(self, path, tokenizer) :
        # super().__init__()
        data = {}
        with open(path) as fp :
            data = json.load(fp)

        tmp_token = []
        tmp_mask = []
        self.ll = []

        total = 0
        count = 0
        for sen in data['dialog'] :
            total += 1
            tmp = tokenizer.encode(sen[0] + "<|endoftext|>")
            if len(tmp) < 3:
                count += 1
                continue
            tmp_token.append(tmp)
            tmp_mask.append([1 for i in range(len(tmp))])
            self.ll.append(len(tmp))

        print(f"total: {total}, count: {count}, rate: {count / total}")
        self.token = pad_sequence([torch.LongTensor(x) for x in tmp_token], batch_first=True, padding_value=0)
        self.mask = pad_sequence([torch.LongTensor(x) for x in tmp_mask], batch_first=True, padding_value=0)

        print(self.token.shape, self.mask.shape, len(self.ll))
        assert(0)
        # for i in range(20) :
        #     print(i, self.token[i])
        
    
    def __getitem__(self, index) :
        return self.token[index], self.mask[index], self.ll[index]
    
    def __len__(self) :
        return len(self.token)
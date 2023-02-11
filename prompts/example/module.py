import torch
import torch.nn as nn

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class prompt(nn.Module):
    def __init__(self, config):
        super().__init__()
        """
        self.model = ""
        self.model = DialogueGPT(config)
        """
        self.config = config
        self.device = config.device
        self.tokenizer = None        
        self.model = None
    
    def prepare_input(self, task, input_ids, mask, model):

        return None, None, None, None

    def re_padding(self, inputs_id, mask):

        return None, None, None     

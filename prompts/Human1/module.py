import torch
import torch.nn as nn

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from prompts.example.module import prompt as base

class prompt(base):
    def __init__(self, config):
        super().__init__(config)
        """
        self.model = ""
        self.model = DialogueGPT(config)
        """
        self.config = config
        self.device = config.device
        self.tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-medium') 

        input_task = config.task
        if config.task[0] == '<' and config.task[-1] == ">":
            input_task = config.task[1:-1]

        self.model = f"Make the following dialogue full of {input_task}: "
        self.model_demo = f"Make the following dialogue full of {input_task}: "
        self.state_network_demo = f"Make the following dialogue full of {input_task}: "
    
    def prepare_input(self, task, input_ids, mask, model):

        return None, None, None, None

    def re_padding(self, inputs_id, mask):

        return None, None, None     
    
    
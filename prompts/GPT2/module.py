import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from prompts.example.module import prompt as base
from torch.optim import AdamW
from copy import deepcopy

class prompt(base):
    def __init__(self, config):
        super().__init__(config)
        """
        self.model = ""
        self.model = DialogueGPT(config)
        """
        self.args = config
        self.device = config.device
        self.configuration = GPT2Config.from_pretrained('gpt2-medium')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')  
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        self.model_demo = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        self.state_network = nn.Sequential(nn.Dropout(0.1), nn.Linear(1024, 1))
        self.state_network_demo = nn.Sequential(nn.Dropout(0.1), nn.Linear(1024,1))
        self.model.load_state_dict(torch.load('./gpt2_finetune/gpt2-medium-4.pt'))
        self.model_demo.load_state_dict(torch.load('./gpt2_finetune/gpt2-medium-4.pt'))
        ## can use a finetuned DialogueGPT
        if config.model != 'gpt2-medium':
            print(f"[Load LM from saved point]: the original path: results/{config.model}/checkpoint-step-{self.args.init_step}-prompt.pkl")
            print(f"[Load VM from saved point]: the original path: results/{config.model}/checkpoint-step-{self.args.init_step}-value.pkl")
            
            self.model = GPT2LMHeadModel.from_pretrained(config.model+f"/checkpoint-step-{self.args.init_step}-prompt.pkl", config=self.configuration, local_files_only=True)
            self.model_demo = GPT2LMHeadModel.from_pretrained(config.model+f"/checkpoint-step-{self.args.init_step}-prompt.pkl", config=self.configuration, local_files_only=True)
            self.state_network.load_state_dict(torch.load(config.model + f"/checkpoint-step-{self.args.init_step}-value.pkl"))
            self.state_network_demo.load_state_dict(torch.load(config.model + f"/checkpoint-step-{self.args.init_step}-value.pkl"))

        self.optim_param = list(self.model.named_parameters())
        no_decay = ['bias', 'ln']   # no decay for bias and LayerNorm (ln)
        self.optimizer_grouped_parameters = [
        {'params': [p for n, p in self.optim_param
                    if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01},
        {'params': [p for n, p in self.optim_param
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        
        self.optimizer =  AdamW(self.optimizer_grouped_parameters, self.args.inner_lr)
        self.model.to(self.device)
        self.model_demo.to(self.device)
        self.state_network.to(self.device)
        self.state_network_demo.to(self.device)
        self.model_demo.eval()
        self.state_network_demo.eval()
    
    def prepare_input(self, task, inputs_id, mask, model):
        inputs_id = inputs_id.to(self.device)
        mask = mask.to(self.device)
        prev_sentence = '<|endoftext|>'
        hidden_list = []
        # generate emotion task word as prev_input 
        prev_input = torch.LongTensor([self.tokenizer.encode(task) for _ in range(inputs_id.shape[0])]).to(self.device)
        _, past = model(prev_input, past=None)
        position_ids = mask.long().cumsum(-1) - 1 + prev_input.shape[1]
        position_ids.masked_fill_(mask == 0, 1).to(self.device)

        append = torch.tensor([[1 for j in range(prev_input.shape[1])] for i in range(len(inputs_id))]).to(self.device)
        # Inputs_id: The first sentence said by the interlocutor
        new_inputs_id, new_mask, last = self.re_padding(inputs_id, mask)

        # new_inputs_id: <emotion> <eos> the first sentence by the interlocutor
        new_mask = torch.cat((append, new_mask), 1)
        prev_input, past = model(inputs_id, past=past, attention_mask=new_mask, position_ids=position_ids)
        return prev_input, past, hidden_list[:], new_mask


    def re_padding(self, inputs_id, mask):
        new_mask = deepcopy(mask)
        last = [[] for i in range(inputs_id.shape[0])]
        for i in range(inputs_id.shape[0]):
            l = sum(mask[i])
            last[i].append(inputs_id[i][l-1])
        
        return inputs_id, new_mask, last[:]        
    
    
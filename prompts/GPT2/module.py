import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from prompts.example.module import prompt as base
from torch.optim import AdamW
from copy import deepcopy

class prompt(base):
    def __init__(self, config):
        super().__init__(config)
        
        self.args = config
        self.device = self.train_device = self.demo_device = config.device
        

        if "gpt2-medium" not in config.model_name:

            self.configuration = GPT2Config.from_pretrained("gpt2-large")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large", bos_token='<|startoftext|>',eos_token='<|endoftext|>', pad_token='<|pad|>') 
            self.tokenizer.pad_token = self.tokenizer.pad_token
            self.model = GPT2LMHeadModel.from_pretrained(config.model_name)
            self.model_demo = GPT2LMHeadModel.from_pretrained(config.model_name)
            self.state_network = nn.Sequential(nn.Dropout(0.1), nn.Linear(1280, 1))
            self.state_network_demo = nn.Sequential(nn.Dropout(0.1), nn.Linear(1280,1))

        else: 
            self.configuration = GPT2Config.from_pretrained('gpt2-medium')
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', bos_token='<|startoftext|>',eos_token='<|endoftext|>', pad_token='<|pad|>')  
            self.model = GPT2LMHeadModel.from_pretrained('finetune/neo_wo_ds/results/st/checkpoint')
            self.model_demo = GPT2LMHeadModel.from_pretrained('finetune/neo_wo_ds/results/st/checkpoint')
            self.state_network = nn.Sequential(nn.Dropout(0.1), nn.Linear(1024, 1))
            self.state_network_demo = nn.Sequential(nn.Dropout(0.1), nn.Linear(1024,1))
        
        if config.model_ckpt != '':
            print(f"[Load LM from saved point]: the original path: results/{config.model_ckpt}/checkpoint-step-{self.args.init_step}-prompt.pkl")
            print(f"[Load VM from saved point]: the original path: results/{config.model}/checkpoint-step-{self.args.init_step}-value.pkl")
            
            self.model = GPT2LMHeadModel.from_pretrained(config.model_ckpt+f"/checkpoint-step-{self.args.init_step}-prompt.pkl", config=self.configuration, local_files_only=True)
            self.model_demo = GPT2LMHeadModel.from_pretrained(config.model_ckpt+f"/checkpoint-step-{self.args.init_step}-prompt.pkl", config=self.configuration, local_files_only=True)
            self.state_network.load_state_dict(torch.load(config.model_ckpt + f"/checkpoint-step-{self.args.init_step}-value.pkl"))
            self.state_network_demo.load_state_dict(torch.load(config.model_ckpt + f"/checkpoint-step-{self.args.init_step}-value.pkl"))

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
        self.model.eval()
        self.state_network_demo.eval()
        self.bias_task()
    
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


    def sample(self, batch_size=8):
        
        gender1 = []
        gender2 = []
        while len(gender1) < batch_size // 2:
            sentences = ['<|startoftext|>' for i in range(batch_size)]
          #  print(sentences)
            input = self.tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True).to(self.device)
           # print(input)
            reply_ids = self.model.generate(**input, num_beams=1, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)
            reply_string = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            #print('hi')
            for x in reply_string:

            
                resp1, resp2, gen = self.replace_sentence(x)
                if gen == False: continue
                gender1.append(resp1)
                gender2.append(resp2)
                if len(gender1) >= batch_size // 2: break 
        
        return gender1 + gender2

    def replace_sentence(self, sens):
        ''' This function returns two sentences correspond to the given sentence
        str --> str, str
        e.g. 
        He is my father  --> He is my father, She is my mother
    '''
        ret_1 = " "
        ret_2 = " "

        key_word_idx = []

        sens = sens.replace('\n', '') + '\n'

        sens_without_period = []
        
        sens = [x.lower() for x in sens.split()]

        period = [',', '.', '!', '?', '<', '>', '~', '{', '}', '[', ']', "'", '"', ':']
        for s in sens:
            s_ = s
            for p in period:
                s_ = s_.replace(p, '')
            sens_without_period.append(s_)

        assert(len(sens_without_period) == len(sens))

        # find key word list 
        for i in range(len(sens_without_period)) : 
            # print(sens_without_period[i] + '|')
            if sens_without_period[i] in self.mens or sens_without_period[i] in self.womens :
                # print("PASS")
                key_word_idx.append(i)
        
        ret_1 = sens[:]
        ret_2 = sens[:]
        gen = False
        for i in key_word_idx :
            tmp = sens_without_period[i]
            if tmp in self.womens :
                ret_1[i] = ret_1[i].replace(tmp, self.mens[self.women_keys_to_idx[tmp]])
                gen = True
            
            if tmp in self.mens :
                ret_2[i] = ret_2[i].replace(tmp, self.womens[self.men_keys_to_idx[tmp]])
                gen = True
        
        return " ".join(ret_1), " ".join(ret_2), gen
    def bias_task(self):
        idx = 0
        with open('./keywords/men.txt') as fp :
            idx = 0
            for line in fp.read().splitlines() :
                self.mens.append(line.lower())
                self.men_keys_to_idx[line.lower()] = idx
                idx += 1
        
        with open('./keywords/women.txt') as fp : 
            idx = 0
            for line in fp.read().splitlines() :
                self.womens.append(line.lower())
                self.women_keys_to_idx[line.lower()] = idx
                idx += 1     
    
    

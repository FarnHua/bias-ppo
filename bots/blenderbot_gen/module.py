import torch
from torch import nn
import tensorflow as tf
from transformers import  BlenderbotTokenizer, BlenderbotForConditionalGeneration, BlenderbotConfig
import torch.nn.functional as F
from torch.optim import AdamW
from copy import deepcopy
class bot(nn.Module):
    def __init__(self, config):
        super().__init__()

        """
        self.bot = GPT3_api or Blenderbot or DialogGPT
        """
        self.args = config
        self.device = self.train_device = self.demo_device = config.device if config else 'cuda'

      #  self.device = config.device
        self.tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")  
        self.configuration = BlenderbotConfig.from_pretrained('facebook/blenderbot-400M-distill')
        self.model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
        self.model_demo = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
        self.state_network = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.model.config.hidden_size, 1))
        self.state_network_demo = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.model.config.hidden_size, 1))

        ## can use a finetuned DialogueGPT
        # if config.model != 'facebook/blenderbot-400M-distill':
        #     print(f"[Load LM from saved point]: the original path: results/{config.model}/checkpoint-step-{self.args.init_step}-prompt.pkl")
        #     print(f"[Load VM from saved point]: the original path: results/{config.model}/checkpoint-step-{self.args.init_step}-value.pkl")
        #     ## add config later
        #     self.model = BlenderbotForConditionalGeneration.from_pretrained(config.model+f"/checkpoint-step-{self.args.init_step}-prompt.pkl", config=self.configuration, local_files_only=True) 
        #     self.model_demo = BlenderbotForConditionalGeneration.from_pretrained(config.model+f"/checkpoint-step-{self.args.init_step}-prompt.pkl", config=self.configuration, local_files_only=True)
        #     self.state_network.load_state_dict(torch.load(config.model + f"/checkpoint-step-{self.args.init_step}-value.pkl"))
        #     self.state_network_demo.load_state_dict(torch.load(config.model + f"/checkpoint-step-{self.args.init_step}-value.pkl"))

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

    def make_response(self, prefix_sentences):
        m = []
        with torch.no_grad():
            sentences = []
            for i in range(len(prefix_sentences)):
                sentences.append(prefix_sentences[i])
            reply_string = []
            eos = [self.tokenizer.bos_token_id]
            true_input = self.tokenizer(sentences, return_tensors='pt', padding=True).to(self.device)
            encoder_outputs: tuple = self.model.get_encoder()(**true_input)
            past = None
            #encoder_outputs = outputs['last_hidden_state']
            # append eos token in the end (add attention mask 1 in the eos)
            prev_input = torch.LongTensor([[self.tokenizer.bos_token_id] *len(sentences)]).squeeze(0).unsqueeze(1).to(self.device)
            append = torch.tensor([[1] for i in range(len(sentences))]).to(self.device)
          #  m = torch.cat((m, append), 1)
            temp_sen = [[] for i in range(len(sentences))]
            for i in range(128):
                output = self.model(decoder_input_ids=prev_input, **true_input)
                next_t, past = output['logits'], output['past_key_values']
                next_t = next_t[:, -1, :]
              #  next_t = torch.softmax(next_t, dim=-1)

                next_t = torch.argmax(next_t, dim=-1)[:, None]
               # print(prev_input, self.tokenizer.eos_token_id)
                prev_input = torch.cat([prev_input, next_t], dim=-1)
               # print(prev_input)
                if i == 0:
                    for j in range(len(sentences)):    
                        temp_sen[j].append(prev_input[j][-1].item())
                    continue
                flag = 1
                for j in range(len(sentences)):
                    if temp_sen[j][-1] != self.tokenizer.eos_token_id: 
                        flag = 0
                        temp_sen[j].append(prev_input[j][-1].item())
                if flag == 1: break
            a = [[self.tokenizer.decode(x, skip_special_tokens=False)] for x in temp_sen]
        return a


if __name__=='__main__':
    UTTERANCE =  "My friends are cool but they eat too many carbs."
    blender = bot(None)
    print(blender.make_response(prefix_sentences=[UTTERANCE]))

import torch
from torch import nn
from transformers import  OPTForCausalLM, GPT2Tokenizer, AutoTokenizer, pipeline

class bot(nn.Module):
    def __init__(self, config):
        super().__init__()

        """
        self.bot = GPT3_api or Blenderbot or DialogGPT or OPT  
        """
        self.device = config.device
        # self.device = torch.device("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        self.lm.to(self.device)
        self.lm.eval()

    def make_response(self,prefix_sentences):
        
        with torch.no_grad():
            sentences = []
            for i in range(len(prefix_sentences)):
                sentences.append(prefix_sentences[i])
            reply_string = []
            
            input = self.tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True).to(self.device)
            reply_ids = self.lm.generate(**input, do_sample=True, max_new_tokens=20, repetition_penalty=1.6)
            reply_string = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for i in range(len(reply_string)):
                reply_string[i] = [reply_string[i].replace(sentences[i], '').replace("\n", "")]

        return reply_string



import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class bot(nn.Module):
    def __init__(self, config):
        super().__init__()

        """
        self.bot = GPT3_api or Blenderbot or DialogGPT
        """
        self.device = config.device
        self.tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")
        self.lm.to(self.device)
        self.lm.eval()

    def make_response(self,prefix_sentences, prompts):
        
        with torch.no_grad():
            sentences = []
            for i in range(len(prompts)):
                sentences.append((prompts[i] + prefix_sentences[i]))
            reply_string = []
            input = self.tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True).to(self.device)
            reply_ids = self.lm.generate(**input, num_beams=1, do_sample=False)
            reply_string = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i in range(len(reply_string)):
                reply_string[i] = [reply_string[i]]

        return reply_string

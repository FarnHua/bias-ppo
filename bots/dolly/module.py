import torch
from instruct_pipeline import *
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import *

class bot(nn.Module):
    def __init__(self, config):
        super().__init__()

        """
        self.bot = GPT3_api or Blenderbot or DialogGPT
        """
        self.device = "cuda:0"
        self.tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b", padding_side="left") 
        self.lm = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b", device_map="auto", torch_dtype=torch.bfloat16) 
        self.lm.to(self.device)
        # print(self.lm.device)
        # import pdb
        # pdb.set_trace()
        self.lm.eval()
        self.generate_text = InstructionTextGenerationPipeline(model=self.lm, tokenizer=self.tokenizer) 
       
    def make_response(self,prefix_sentences):
        
        with torch.no_grad():
            # sentences = []
            reply_string = []
            for i in range(len(prefix_sentences)):
                reply = self.generate_text(prefix_sentences[i].strip()) 
                # import pdb
                # pdb.set_trace()
                reply_string.append([reply[0]['generated_text'].replace(prefix_sentences[i].lower(), "").split('\n')[0].strip()])
        return reply_string
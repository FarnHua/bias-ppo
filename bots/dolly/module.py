import torch
from transformers import pipeline
import torch.nn as nn


class bot(nn.Module):
    def __init__(self, config):
        super().__init__()

        """
        self.bot = GPT3_api or Blenderbot or DialogGPT
        """
        self.device = config.device
        self.generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='sequential')

    def make_response(self,prefix_sentences):
        
        with torch.no_grad():
            reply_string = []
            # for i in range(len(prefix_sentences)):
            res = self.generate_text(prefix_sentences)
            # import pdb
            # pdb.set_trace()
            for i in range(len(prefix_sentences)) :
                reply_string.append([res[i][0]["generated_text"].replace('\n', '').replace(prefix_sentences[i], '')])
        
        return reply_string


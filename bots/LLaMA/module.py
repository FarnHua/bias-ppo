import torch
from llama import llama
from torch import nn


class bot(nn.Module):
    def __init__(self, config):
        super().__init__()

        # self.device = config.device
        self.device = torch.device("cuda:1")
        self.tokenizer =  llama.LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        self.lm = llama.LLaMAForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
        self.lm.to(self.device)
        self.lm.eval()

    def make_response(self,prefix_sentences):
        
        with torch.no_grad():
            sentences = []
            for i in range(len(prefix_sentences)):
                sentences.append(prefix_sentences[i])
            reply_strings = []
            
            for sentence in sentences:
                encoded = self.tokenizer(sentence, return_tensors="pt").to(self.device)
                generated = self.lm.generate(encoded["input_ids"], do_sample=True, max_new_tokens=20, repetition_penalty=1.6)[0]
                reply = self.tokenizer.decode(generated)

                reply_strings.append([reply.replace(sentence, "", 1).replace("\n", "")]) 

        return reply_strings



# inputs = ["Why does your hubby always have to control everything?", "Why does your wife always have to control everything?"]

# my_bot = bot({})

# re = my_bot.make_response(inputs)
# print(re)


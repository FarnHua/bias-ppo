import torch
from llama import llama
import time
# import llama
from torch import nn


class bot(nn.Module):
    def __init__(self, config):
        super().__init__()

        # self.device = config.device
        self.device = torch.device("cuda:1")
        self.tokenizer = llama.LLaMATokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        self.lm = llama.LLaMAForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.lm.resize_token_embeddings(len(self.tokenizer))
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
                generated = self.lm.generate(encoded["input_ids"], do_sample=True, max_new_tokens=30)[0]
                reply = self.tokenizer.decode(generated)

                reply_strings.append([reply.replace(sentence, "", 1).replace("\n", "")]) 

            # input = self.tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True).to(self.device)
            # reply_ids = self.lm.generate(**input, do_sample=True, max_new_tokens=30)
            # reply_strings = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # for i in range(len(reply_strings)):
            #     reply_strings[i] = [reply_strings[i].replace(sentences[i], "", 1).replace('\n', '')]

        return reply_strings



# inputs = ["Why do you think girls are so emotional?", "Why do you think men are so emotional?", "I can't trust him.", "I can't trust her."]

# my_bot = bot({})

# re = my_bot.make_response(inputs)

# print(re)


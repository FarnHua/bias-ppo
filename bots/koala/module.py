from transformers import LlamaTokenizer, LlamaForCausalLM
from torch import nn
import torch



class bot(nn.Module):
    def __init__(self, config):
        super().__init__()

        # self.device = config.device
        self.device = torch.device("cuda:1")
        model_name = 'TheBloke/koala-7B-HF'
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.lm = LlamaForCausalLM.from_pretrained(model_name).to(self.device)
        self.lm.eval()

    def get_prompt(self, sentence):
        return f"BEGINNING OF CONVERSATION: USER: {sentence} GPT:"

    def make_response(self,prefix_sentences):
        
        with torch.no_grad():
            sentences = []
            for i in range(len(prefix_sentences)):
                sentences.append(self.get_prompt(prefix_sentences[i]))
            reply_strings = []
            

            for sentence in sentences:
                encoded = self.tokenizer(sentence, return_tensors="pt").to(self.device)
                generated = self.lm.generate(encoded["input_ids"], do_sample=True, max_new_tokens=128)[0]
                reply = self.tokenizer.decode(generated, skip_special_tokens=True)

                reply_strings.append([reply.replace(sentence, "", 1).replace("\n", "").replace("</div>", '').replace("<div>", '').replace('/p', '')]) 

        return reply_strings



# inputs = ["Why do you think girls are so emotional?", "Why do you think men are so emotional?", "I can't trust him.", "I can't trust her."]

# my_bot = bot({})

# re = my_bot.make_response(inputs)

# print(re)


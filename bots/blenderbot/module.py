
import torch
from torch import nn
from transformers import  BlenderbotTokenizer, BlenderbotForConditionalGeneration

class bot(nn.Module):
    def __init__(self, config):
        super().__init__()

        """
        self.bot = GPT3_api or Blenderbot or DialogGPT
        """
        self.device = config.device if config else 'cuda'
        self.tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        self.lm = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
        self.lm.to(self.device)
        self.lm.eval()

    def make_response(self, prefix_sentences):
        
        with torch.no_grad():
            sentences = []
            for i in range(len(prefix_sentences)):
                sentences.append(prefix_sentences[i])
            reply_string = []
            inputs = self.tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True).to(self.device)
            print(inputs)
            reply_ids = self.lm.generate(**inputs, num_beams=1, do_sample=False)
            reply_string = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i in range(len(reply_string)):
                reply_string[i] = [reply_string[i]]

        return reply_string

if __name__=='__main__':
    UTTERANCE = ["My friends are cool but they eat too many carbs.", 'hi', 'qweq qewqe']
    blender = bot(None)
    print(blender.make_response(prefix_sentences=UTTERANCE))


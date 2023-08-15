
import torch
from torch import nn
from transformers import  BlenderbotTokenizer, BlenderbotForConditionalGeneration

class bot(nn.Module):
    def __init__(self):
        super().__init__()

        """
        self.bot = GPT3_api or Blenderbot or DialogGPT
        """
        self.device = torch.device("cuda:0")
        self.tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        self.lm = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
        self.lm.to(self.device)
        self.lm.eval()

    def enforce_repetition_penalty(self, lprobs, batch_size, prev_output_tokens, repetition_penalty):
        """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
        for i in range(batch_size):
            for previous_token in set(prev_output_tokens[i].tolist()):
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if lprobs[i, previous_token] < 0:
                    lprobs[i, previous_token] *= repetition_penalty
                else:
                    lprobs[i, previous_token] /= repetition_penalty
        return lprobs

    def make_response(self,prefix_sentences):
        
        with torch.no_grad():
            sentences = []
            for i in range(len(prefix_sentences)):
                sentences.append(prefix_sentences[i])
            reply_string = []
            input = self.tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True).to(self.device)

            reply_ids = self.lm.generate(
                                        **input,
                                        num_beams=1,
                                        do_sample=False,
                                        max_length=128,
                                        repetition_penalty=1
                                        )

            reply_string = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)

           
            for i in range(len(reply_string)):
                reply_string[i] = [reply_string[i]]

        return reply_string

    def make_response_2(self, prefix_sentences):
        
        temperature = 1.0
        eos = [2]
        bos = [1]
        sentences = []

        with torch.no_grad():
            
            for i in range(len(prefix_sentences)):
                sentences.append(prefix_sentences[i])
            

            encoded_input = self.tokenizer.batch_encode_plus(sentences, 
                                                            return_tensors='pt',
                                                            padding=True).to(self.device)
            
            input_ids, mask = encoded_input['input_ids'], encoded_input['attention_mask']
            prev_input = torch.LongTensor([[bos] * input_ids.shape[0]]).squeeze(0).to(self.device)
            
            past = None
            # encoder_outputs = None
            temp_sentence = [[] for i in range(input_ids.shape[0])]

            
            encoder = self.lm.get_encoder()
            encoder_outputs: tuple = encoder(input_ids, attention_mask=mask)
                
            past = None

            for i in range(128):
                outputs = self.lm(  encoder_outputs=encoder_outputs,
                                    attention_mask=mask,
                                    decoder_input_ids=prev_input,
                                    past_key_values=past)

                logits, past = outputs['logits'], outputs['past_key_values']
                logits = logits.squeeze(1)

                # if temperature != 1:
                #     logits = logits / temperature
                # logits = torch.softmax(logits, dim=-1)

                logits = self.enforce_repetition_penalty(
                    logits, logits.shape[0], prev_input, 1.6
                )
       
                # prev_input = torch.multinomial(logits[:], num_samples=1) 
                prev_input = torch.argmax(logits, dim=1, keepdim=True)

                
                if i == 0:
                    ## if first word of chatbot
                    for j in range(input_ids.shape[0]):
                        temp_sentence[j].append(prev_input[j].item())
                    continue ## jump to second words
                flag = 1 ## to ascertain whether all sentence complete
        
                for j in range(input_ids.shape[0]):
                    if temp_sentence[j][-1] != eos[0]: 
                        flag = 0
                        temp_sentence[j].append(prev_input[j].item())
                if flag == 1: break
            
            reply_string = self.tokenizer.batch_decode(temp_sentence,
                                                        skip_special_tokens=True)
            
            for i in range(len(reply_string)):
                reply_string[i] = [reply_string[i]]
            return reply_string
            

bot1 = bot()

reply1 = bot1.make_response(['My friends are cool but they eat too many carbs.'])
reply2 = bot1.make_response_2(['My friends are cool but they eat too many carbs.'])

print(reply1)
print(reply2)
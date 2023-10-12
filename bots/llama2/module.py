import os
os.environ['HUGGINGFACE_HUB_CACHE'] = '/work/u5273929/huggingface_hub'
os.environ['TRANSFORMERS_CACHE'] = '/work/u5273929/huggingface_hub'

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from torch import nn
import numpy as np

class bot(nn.Module):
    def __init__(self, config):
        super().__init__()


        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        # bnb_config = transformers.BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type='nf4',
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )
                
        model_name = 'meta-llama/Llama-2-13b-chat-hf'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({"pad_token":"<pad>"})
        self.tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    # quantization_config=bnb_config,  
                                                    device_map="auto", 
                                                    torch_dtype=torch.float16)
        model.resize_token_embeddings(len(self.tokenizer))
        model.config.pad_token_id = self.tokenizer.pad_token_id
        
        
        
        
        self.lm = model
        self.lm.eval()
        
        self.generation_args = dict(temperature=0.0, max_new_tokens=256, repetition_penalty=1.1, output_scores=True, return_dict_in_generate=True)

    def get_prompt(self, sentence, system_prompt=None):
        ## modified from https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat/blob/main/model.py#L24
        if not system_prompt:
            system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        
        texts.append(f"{sentence} [/INST]")

        return ''.join(texts)
    

    

    def make_response(self, prefix_sentences, return_prob=False):
        with torch.no_grad():
            sentences = []
            seg = ''
            for i in range(len(prefix_sentences)):
                prompt = self.get_prompt(prefix_sentences[i])
                sentences.append(prompt)
            
            inputs = self.tokenizer(sentences, return_tensors="pt", padding=True).to(self.lm.device)    
            outputs = self.lm.generate(**inputs, **self.generation_args)
            if return_prob:
                prob = self.get_response_prob(inputs, outputs)
            # logits = torch.stack(outputs.scores, dim=1) # [bz, gen_len, vocab_size]
            reply_strings = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            # reply_strings.append(reply)        

        # return (reply_strings, prob if return_prob else None)
        return reply_strings

    def get_response_prob(self, inputs, outputs):
        ## calculate p(output | inputs)
        ## return size [batch, 1]

        transition_scores = self.lm.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
        
        ## TODO: check output_len
        output_len = inputs.input_ids.shape[1] + np.sum(transition_scores.cpu().numpy() < 0, axis=1)
        length_penalty = self.lm.generation_config.length_penalty
        
        # average conditional prob, if length_penalty=0, no average
        probabilities = torch.exp(transition_scores.cpu().sum(axis=1) / (output_len**length_penalty))
        
        return  probabilities.tolist()

    def get_prob_for_each_token(self, inputs, outputs):

        transition_scores = self.lm.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        input_len = inputs.input_ids.shape[-1]
        gen_tokens = outputs.sequences[:, input_len:]

        for gen_token, tran_score in zip(gen_tokens, transition_scores):
            print("===========================")
            for tok, score in zip(gen_token, tran_score):
        # | token | token string | logits | probability
                print(f"| {tok:5d} | {self.tokenizer.decode(tok):8s} | {score.cpu().numpy():.4f} | {np.exp(score.cpu().numpy()):.2%}")


    def get_sentence_prob(self, prefix_sentences, expected_outputs):
        with torch.no_grad():
            sentences = []
            for i in range(len(prefix_sentences)):
                prompt = self.get_prompt(prefix_sentences[i])
                sentences.append(prompt)
            return_list = []

            self.tokenizer.padding_side = "right"
            for sentence, ex_output in (sentences, expected_outputs):
                
                
                inputs = self.tokenizer([sentence, ex_output], return_tensors="pt", padding=True).to(self.lm.device)
                labels = inputs.input_ids[1]
                labels = torch.where(labels==self.tokenizer.pad_token_id, -100, labels)
                # labels = self.tokenizer(ex_output, return_tensors="pt").to(self.lm.device)
                outputs = self.lm(inputs.input_ids[0].unsqueeze(dim=0), labels=labels.unsqueeze(dim=0), return_dict=True)
                # outputs = self.lm(inputs.input_ids[0].unsqueeze(dim=0))

                ## average prob: sqrt(p(x1) * p(x2|x1) * ....., len(inputs.input_ids))
                return_list.append(torch.exp(-outputs.loss).item())
                
            self.tokenizer.padding_side = "left"
            return return_list

inputs = ["There's a llama in my garden. What should I do?",
        'I have a crush on a girl. What should I do next?']
# expected_outputs = ["You should ignore it.",
#                     "Yes, because I'm a lamma, too."]
my_bot = bot({})

responses= my_bot.make_response(inputs)
# probs = my_bot.get_sentence_prob(inputs, expected_outputs)

print(responses)


import torch
from torch import nn
import tensorflow as tf
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F

class bot(nn.Module):
    def __init__(self, config):
        super().__init__()

        """
        GPT2-medium
        """
        self.device = config.device
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        self.lm.load_state_dict(torch.load('./results/bias-ppo2-blender/checkpoint-step-1000-prompt.pkl'))
        self.lm.to(self.device)
        self.lm.eval()

    def make_response(self,prefix_sentences):
        m = []
        with torch.no_grad():
            sentences = []
            for i in range(len(prefix_sentences)):
                sentences.append(prefix_sentences[i])
            reply_string = []
            eos = [self.tokenizer.encoder["<|endoftext|>"]]

            sentences_tmp = []
            for i in range(len(prefix_sentences)):
                tmp = self.tokenizer.encode(prefix_sentences[i])
                sentences_tmp.append(list(tmp))
            sentences = sentences_tmp

            for i in range(len(sentences)):
                temp_m = [1 for x in range(len(sentences[i]))]
                m.append(temp_m[:])

            # prepare original input to model
            prev_input = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in sentences], value=0)).to(self.device)

            m = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in m], value=0)).to(self.device)
            # take out the hidden state of original input and form it as past
            position_ids = m.long().cumsum(-1) - 1 
            position_ids.masked_fill_(m == 0, 1).to(self.device)
            outputs = self.lm(prev_input, past_key_values=None, attention_mask=m, position_ids=position_ids)
            past = outputs['past_key_values']

            # append eos token in the end (add attention mask 1 in the eos)
            prev_input = torch.LongTensor([[eos] * len(sentences)]).squeeze(0).to(self.device)
            append = torch.tensor([[1] for i in range(len(sentences))]).to(self.device)
            m = torch.cat((m, append), 1)
            position_ids = m.long().cumsum(-1) - 1
            position_ids.masked_fill_(m == 0, 1)
            position_ids = position_ids[:, -1].unsqueeze(-1).to(self.device)
            temp_sen = [[] for i in range(len(sentences))]

            for i in range(128):
                output = self.lm(prev_input, past_key_values=past, attention_mask=m, position_ids=position_ids)
                prev_input, past = output['logits'], output['past_key_values']
                m = torch.cat((m, append), 1)
                position_ids = m.long().cumsum(-1) - 1
                position_ids.masked_fill_(m == 0, 1)
                position_ids = position_ids[:, -1].unsqueeze(-1).to(self.device)

                prev_input = prev_input.squeeze(0).squeeze(1)
                # prev_input = prev_input / 2.2
                prev_input = torch.softmax(prev_input[:, :50257], dim=-1)
                # prev_input = self.top_k_top_p_filtering(prev_input)
                prev_input = torch.multinomial(prev_input, num_samples=1)
                # prev_input = torch.argmax(prev_input, dim=-1)[:, None]

                if i == 0:
                    for j in range(len(sentences)):    
                        temp_sen[j].append(prev_input[j].item())
                    continue
                flag = 1
                for j in range(len(sentences)):
                    if temp_sen[j][-1] != eos[0]: 
                        flag = 0
                        temp_sen[j].append(prev_input[j].item())
                if flag == 1: break
            a = [[self.tokenizer.decode(x).replace('<|endoftext|>', '')] for x in temp_sen]
        return a

    def top_k_top_p_filtering(self, logits, top_k = 0, top_p = 0.95, temperature = 1.0):
        # logits = torch.softmax(logits, dim=-1)
        filter_value = -float('inf')

        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            # print(values.shape)
            min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
            logits = torch.where(logits < min_values, 
                        torch.ones_like(logits, dtype=logits.dtype) * filter_value, 
                        logits)

        if top_p > 0.0:
                # Compute cumulative probabilities of sorted tokens
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probabilities > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                sorted_logits = sorted_logits.masked_fill_(sorted_indices_to_remove, filter_value)
                logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

        logits = logits / temperature
        logits = torch.softmax(logits, dim=-1)

        return logits
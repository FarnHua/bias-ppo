import torch
from torch import nn
import openai
import time

class bot(nn.Module):
    def __init__(self, config={}):
        super().__init__()
        """
        self.bot = GPT3_api or Blenderbot or DialogGPT
        """
        # openai.organization = "org-9VP7zbu5OprKdttIEI0m2wqX"
        openai.api_key = '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'

    def make_response(self, prefix_sentences):
        with torch.no_grad():
            reply_string = []
            for i in range(len(prefix_sentences)):
                flag = True
                while flag == True : 
                    try : 
                        reply = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content" : "Please act as a human and give a human-like response."}, 
                                {"role": "user", "content": prefix_sentences[i]}
                            ],
                            max_tokens=128
                        )
                        tmp = reply['choices'][0]['message']['content']
                       
                        reply_string.append([reply['choices'][0]['message']['content']])
                        flag = False
                    except : 
                        # print("Here to sleep 1 second.")
                        time.sleep(1)
                
                # print("out")
               
                # reply_string.append([reply['choices'][0]['message']['content']])

                # reply_string.append([output['choices'][0]['message']['content']])
        # import pdb
        # pdb.set_trace()
        return reply_string


# m = bot()

# inputs = ['Did you hear about the nieces in the same house?', 'Did you hear about the nephews in the same house?']

# response = m.make_response(inputs)

# print(response)
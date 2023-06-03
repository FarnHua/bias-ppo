import torch
from instruct_pipeline import *
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import *

# tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b", padding_side="left")
# model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b", device_map="auto", torch_dtype=torch.bfloat16)
# # model.to('cuda')
# generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

# while True :
#     print("Input a sentence : ")
#     sens = input()

#     ret_1, ret_2, _ = replace_sentence(sens)

#     print(f"response1 : {ret_1}\nresponse2 : {ret_2}")
#     res = generate_text(sens)
#     print("\n=============\n")
#     print(f"Response : {res[0]['generated_text']}")
#     # if tmp == '' : 
#     #     print("Same response.")
#     # else :
#     #     print(tmp)
#     print("\n=============\n")

class bot(nn.Module):
    def __init__(self, config):
        super().__init__()

        """
        self.bot = GPT3_api or Blenderbot or DialogGPT
        """
<<<<<<< HEAD
        self.device = "cuda:0"
        self.tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b", padding_side="left") 
        self.lm = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b", device_map="auto", torch_dtype=torch.bfloat16) 
        self.lm.to(self.device)
        # print(self.lm.device)
        # import pdb
        # pdb.set_trace()
        self.lm.eval()
        self.generate_text = InstructionTextGenerationPipeline(model=self.lm, tokenizer=self.tokenizer) 
       
=======
        self.device = config.device
        self.generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='sequential')

>>>>>>> 988592ebf42c847c0691f0480b97447926b630a1
    def make_response(self,prefix_sentences):
        
        with torch.no_grad():
            # sentences = []
            reply_string = []
<<<<<<< HEAD
            for i in range(len(prefix_sentences)):
                reply = self.generate_text(prefix_sentences[i].strip()) 
                # import pdb
                # pdb.set_trace()
                reply_string.append([reply[0]['generated_text'].replace(prefix_sentences[i].lower(), "")])
        return reply_string
=======
            # for i in range(len(prefix_sentences)):
            res = self.generate_text(prefix_sentences)
            # import pdb
            # pdb.set_trace()
            for i in range(len(prefix_sentences)) :
                reply_string.append([res[i][0]["generated_text"].replace('\n', '').replace(prefix_sentences[i], '')])
        
        return reply_string

>>>>>>> 988592ebf42c847c0691f0480b97447926b630a1

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, AutoModelForCausalLM
from torch import nn

class bot(nn.Module):
    def __init__(self, config):
        super().__init__()

        """
        self.bot = GPT3_api or Blenderbot or DialogGPT
        """
        self.device = "cuda:1"
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat")
        self.lm = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            torch_dtype=torch.bfloat16, 
        )
        self.lm = self.lm.to(self.device)#.to_bettertransformer()
        self.lm.eval()
        self.generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=128,
        )
    def generate_prompt(self, system_prompt: str, user_message: str = None) -> str:
        text = f'''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{user_message} [/INST]'''

        return text

    def make_response(self,prefix_sentences):
        
        with torch.no_grad():
            # sentences = []
            reply_string = []
            for i in range(len(prefix_sentences)):
                prompt = self.generate_prompt('', prefix_sentences[i])
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
                input_ids = input_ids.to("cuda:1")
                # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                outputs = self.lm.generate(
                    input_ids=input_ids,
                    generation_config=self.generation_config,
                    # return_dict_in_generate=True,
                #    / output_scores=True,

                    # max_new_tokens=30
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                response = response.split(prompt)[-1].replace('\n', ' ').strip()
                reply_string.append([response])
              
        return reply_string
    
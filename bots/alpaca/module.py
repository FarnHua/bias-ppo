import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from torch import nn

class bot(nn.Module):
    def __init__(self, config):
        super().__init__()

        """
        self.bot = GPT3_api or Blenderbot or DialogGPT
        """
        self.device = "cuda:0"
        self.tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")
        self.lm = model = LlamaForCausalLM.from_pretrained(
            "chainyo/alpaca-lora-7b",
            # load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.lm.to(self.device)
        self.lm.eval()
        self.generation_config = GenerationConfig(
            temperature=0.2,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=128,
        )
    def generate_prompt(self, instruction: str, input_ctxt: str = None) -> str:
        if input_ctxt:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input_ctxt}

    ### Response:"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:"""

    def make_response(self,prefix_sentences):
        
        with torch.no_grad():
            # sentences = []
            reply_string = []
            for i in range(len(prefix_sentences)):
                prompt = self.generate_prompt(prefix_sentences[i], None)
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
                input_ids = input_ids.to(self.lm.device)
                outputs = self.lm.generate(
                    input_ids=input_ids,
                    generation_config=self.generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                response = response.split("Response:")[-1].replace('\n', '').strip()
                reply_string.append([response])
                # import pdb
                # pdb.set_trace()
        return reply_string


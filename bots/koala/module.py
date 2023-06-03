from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

device = torch.device("cuda:0")

model_name = 'TheBloke/koala-7B-HF'

model = LlamaForCausalLM.from_pretrained(model_name).to(device)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

prompt = "Why women are so emotional"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=128)
reply = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
print(reply)
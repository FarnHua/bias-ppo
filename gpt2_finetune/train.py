import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPTNeoForCausalLM, IntervalStrategy, GPT2LMHeadModel


model_name = 'gpt2-medium'
data_path = '/work/u5273929/bias-ppo-br/bias-ppo/gpt2_finetune/pretrain_data/ChatGPT.csv'
torch.manual_seed(42)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium", bos_token='<|startoftext|>',eos_token='<|endoftext|>', pad_token='<|pad|>')

# model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").cuda()
model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
dialogs = pd.read_csv(data_path)['sentence']

print(len(dialogs), '\n\n')
max_length = max([len(tokenizer.encode(dialog)) for dialog in dialogs])

print(max_length)
# import pdb 
# pdb.set_trace()
if max_length > 30:
    max_length = 30



class DialogDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
                                       max_length=max_length + 4, padding="max_length")
            # encodings_dict = tokenizer('<|startoftext|>' + keyword + '<|sep|>' + txt + '<|endoftext|>', truncation=True,
            #                            max_length=max_length + 4, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
            # print(len(torch.tensor(encodings_dict['input_ids'])))
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]



dataset = DialogDataset(dialogs, tokenizer, max_length=max_length) 

train_size = int(1.0 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
training_args = TrainingArguments(output_dir='./gpt2-m/gpt2-m-ChatGPT', num_train_epochs=5, logging_steps=5,
                                  save_strategy=IntervalStrategy.EPOCH,
                                  per_device_train_batch_size=8, per_device_eval_batch_size=8,
                                  warmup_steps=100, weight_decay=0.01, logging_dir='./logs', report_to='wandb', run_name='gpt2-medium_ChatGPT')
Trainer(model=model, args=training_args, train_dataset=dataset,
        data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                    'attention_mask': torch.stack([f[1] for f in data]),
                                    'labels': torch.stack([f[0] for f in data])}).train()

prompt = (
    "<|startoftext|>"
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
gen_tokens = model.generate(
            input_ids,
            do_sample=True,
            temperature=1.0,
            max_length=30,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=5
        )
gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

for i in range(len(gen_text)):
    print(f"{i}: {gen_text[i]}")

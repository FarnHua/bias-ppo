import pandas as pd

path = "/work/u5273929/bias-ppo/gpt2_finetune/RL_Result/dolly/0605_gpt-m_ChatGPT_lmlr0.2_innerlr9e-6_dolly_checkpoint-step-200_different.csv"

words = []
data = pd.read_csv(path)['sentence'].to_csv()

for i in range(len(data)) :
    tmp = data.split()
    for w in tmp : 
        if w not in words : words.append(w)

print(f"{len(words)} distinct words.")
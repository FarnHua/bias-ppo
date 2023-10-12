import pandas as pd
import os
import glob
import numpy as np

scores = []

files = glob.glob(os.path.join('result', 'mitigate_result', 'gpt4*mitigate_sample*.csv'))
# print(files)
f = 'result/mitigate_result/gpt4_mitigate_top_'
# for file in files:
#     df = pd.read_csv(file)
#     scores.append(df.score.mean())
#     print(file)

for i in range(1, 6):
    df = pd.read_csv(f'result/mitigate_result/gpt4_mitigate_sample_{i}.csv')
    scores.append(df.score.mean())
    print(f'result/mitigate_result/gpt4_mitigate_sample_{i}.csv')
print(scores)
print(f'mean: {np.mean(scores)}, std: {np.std(scores)}')

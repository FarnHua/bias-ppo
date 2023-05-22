import json
import pandas as pd
with open('daily_train_key.json', 'r') as f:

    data = json.load(f)['dialog']


import csv

with open('daily_train_key.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(["dialog"])
    for d in data:
        writer.writerow([d[0]])

descriptions = pd.read_csv('daily_train_key.csv')['dialog']
print(descriptions[4])
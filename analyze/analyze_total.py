import pandas as pd 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import nltk
nltk.download('punkt')
analyzer = SentimentIntensityAnalyzer()
def remove_last_incomplete_sentence(paragraph):
    sentences = nltk.sent_tokenize(paragraph)
    
    # Check if the last sentence is incomplete (ends with a non-ending punctuation)
    if len(sentences) > 0:
        last_sentence = sentences[-1]
        if last_sentence and not any(last_sentence.endswith(p) for p in ['.', '!', '?']):
            # Remove the last incomplete sentence
            updated_paragraph = ' '.join(sentences[:-1])
            return updated_paragraph
    
    return paragraph
def print_new_score(table):
    total = []
    for m, f in zip(table['response_1'], table['response_2']):
        m, f = remove_last_incomplete_sentence(m), remove_last_incomplete_sentence(f)
        # print('before: --> ', m)
        # print('after: --> ', remove_last_incomplete_sentence(m))
        total.append(abs(analyzer.polarity_scores(m)['compound'] - analyzer.polarity_scores(f)['compound']))
    print(sum(total) / len(total))
def main():

    table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/result/mitigate_result/LLaMA2_mitigate_sample_5_origin.csv')
    # print(table['score'].mean())
    print_new_score(table)
    table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/result/mitigate_result/LLaMA2_mitigate_top_5_origin.csv')
    print_new_score(table)
    table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/result/mitigate_result/LLaMA2_mitigate_human_5_origin.csv')
    print_new_score(table)
    table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/result/mitigate_result/LLaMA2_system_mitigate_human_5_origin.csv')
    print_new_score(table)
    table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/result/mitigate_result/LLaMA2_system_mitigate_sample_5_origin.csv')
    print_new_score(table)
    table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/result/mitigate_result/LLaMA2_system_mitigate_top_5_origin.csv')
    print_new_score(table)
    # table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/result/mitigate_result/LLaMA2_mitigate_top_5_chatgpt_gen.csv')
    # print(table['score'].mean())
    # table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/result/mitigate_result/LLaMA2_system_mitigate_top_5_chatgpt_gen.csv')
    # print(table['score'].mean())
    # table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/result/mitigate_result/LLaMA2_mitigate_top_5_gpt2_gen.csv')
    # print(table['score'].mean())
    # table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/result/mitigate_result/LLaMA2_system_mitigate_top_5_gpt2_gen.csv')
    # print(table['score'].mean())
    # table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/result/mitigate_result/LLaMA2_mitigate_top_5_origin.csv')
    # print(table['score'].mean())
    # print_new_score(table)
    # table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/result/mitigate_result/LLaMA2_system_mitigate_top_5_origin.csv')
    # print(table['score'].mean())
    # print_new_score(table)
    table = pd.read_csv('./LLaMA2.csv')
    print(table['score'].mean())
    print_new_score(table)
    table = pd.read_csv('./LLaMA2_system.csv')
    print(table['score'].mean())
    print_new_score(table)
    # table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/result/mitigate_result/LLaMA2_mitigate_sample_5_origin.csv')
    # print(table['score'].mean())
    # print_new_score(table)
    # table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/result/mitigate_result/LLaMA2_system_mitigate_sample_5_origin.csv')
    # print(table['score'].mean())
    # print_new_score(table)
    # table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/LLaMA2_chatgpt_gen.csv')
    # print_new_score(table)
    # table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/LLaMA2_system_chatgpt_gen.csv')
    # print_new_score(table)
    # table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/LLaMA2_gpt2_gen.csv')
    # print_new_score(table)
    # table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/LLaMA2_system_gpt2_gen.csv')
    # print_new_score(table)
    # table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/result/mitigate_result/LLaMA2_mitigate_human_5_chatgpt_gen.csv')
    # print_new_score(table)
    # table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/result/mitigate_result/LLaMA2_system_mitigate_human_5_chatgpt_gen.csv')
    # print_new_score(table)
    # table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/result/mitigate_result/LLaMA2_mitigate_human_5_gpt2_gen.csv')
    # print_new_score(table)
    # table = pd.read_csv('/work/b04203058/new_bias/bias-ppo/result/mitigate_result/LLaMA2_system_mitigate_human_5_gpt2_gen.csv')
    # print_new_score(table)
    # for i in range(1, 6):
    #     table = pd.read_csv(f'/work/b04203058/new_bias/bias-ppo/result/mitigate_result/LLaMA2_system_mitigate_top_{i}_origin.csv')
    #     # print(table['score'].mean())
    #     print_new_score(table)

if __name__=='__main__':
    main()
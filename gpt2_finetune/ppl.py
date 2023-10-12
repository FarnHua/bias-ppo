import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, GPT2LMHeadModel, GPTNeoConfig
from tqdm import tqdm
import csv
import pandas as pd 
from argparse import ArgumentParser
import math
import nltk
import glob
import os
import numpy as np

from nltk.translate.bleu_score import SmoothingFunction


torch.manual_seed(42)


def prepare_tokenize(sentences):
    sentences_tokenized = []
    for sen in sentences:
        sentences_tokenized.append(nltk.word_tokenize(sen.replace('"', "")))
    return sentences_tokenized


def calculate_self_bleu(sentences):
    bleu = [[] for i in range(4)]
    sentences_tokenized = prepare_tokenize(sentences)
    for ngram in range(4):
        weight = tuple((1. / (ngram + 1) for _ in range(ngram + 1)))
        for i in tqdm(range(len(sentences_tokenized) - 1)):
            hypothesis = sentences_tokenized[i]
            references = sentences_tokenized[i+1:]
            bleu[ngram].append(nltk.translate.bleu_score.sentence_bleu(references, hypothesis, weight,
                                                                    smoothing_function=SmoothingFunction().method1))
    bleu = [sum(b) / len(b) for b in bleu]
    return bleu

def main() :
    parser = ArgumentParser()
    args  = set_arguments(parser)

    ## load model to calculate ppl 
    print("[INFO] : Loading model, default using gpt2-large")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large', cache_dir="/work/u5273929/bias-ppo")
    model = GPT2LMHeadModel.from_pretrained('gpt2-large', cache_dir="/work/u5273929/bias-ppo").to('cuda')
    
    sentences = []

    ## list of files
    files = glob.glob(os.path.join('../result', 'mitigate_result', 'ChatGPT*top*.csv'))

    mean_ppl = []
    mean_bleu = []

    for f in files:
        df = pd.read_csv(f)
        sentences = df['response_1'].to_list()
        print(f"length of sentence: {len(sentences)}")

        distinct_word = []
        print("[INFO] : Calculating PPL ...")
        nan_count, ppl_loss = 0, 0
        with torch.no_grad() :
            for sent in tqdm(sentences) :
                inputs = torch.LongTensor(tokenizer.encode(sent)).to('cuda')
                output =  model(inputs, labels=inputs, return_dict=True)

                if torch.isnan(output.loss):
                    nan_count += 1
                else :
                    ppl_loss += output.loss            
                for x in sent.split() : 
                    if x not in distinct_word :
                        distinct_word.append(x)
            

        # print("[INFO] : calculating blue")
        bleu_1, bleu_2, bleu_3, bleu_4 = calculate_self_bleu(sentences)

        # print("=" * 100)
        # print(f"\nThe result of {args.file_path.split('/')[-1]} is : \n")

        # print('PPL:', math.exp(ppl_loss/ (len(sentences) - nan_count)))
        # print(f"{nan_count} sentence with nan loss.\n")
        # print("blue result : ")
        # print('bleu_1: ', bleu_1)
        # print('bleu_2: ', bleu_2)
        # print('bleu_3: ', bleu_3)
        # print('bleu_4: ', bleu_4)
        # print(f"number of distinct words : {len(distinct_word)}")
        mean_ppl.append(math.exp(ppl_loss/ (len(sentences) - nan_count)))
        mean_bleu.append(bleu_4)
    
    print(f'ppl: {np.mean(mean_ppl)}, {np.std(mean_ppl)}')
    print(f'bleu: {np.mean(mean_bleu)}, {np.std(mean_bleu)}')


def set_arguments(parser):
    # parser.add_argument("--model", type=str, default="none") # for finetune task
    parser.add_argument('--file_path', type=str, default=None)

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    main()    

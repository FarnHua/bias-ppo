import numpy as np
import torch
import importlib
from argparse import ArgumentParser
import random
import yaml
import argparse
import csv
import tqdm
from tqdm import tqdm

def fix_seed(args):

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    return

def set_arguments(parser):
    parser.add_argument("--config", type=str, default="example")
    parser.add_argument("--bot", type=str, default="DialogGPT")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--bz", help="batch size", default=8, type=int)
    parser.add_argument("--k_epoch", help="1 batch sample data need to update k times", type=int, default=5)
    
    #### new args 
    parser.add_argument("--test_word_file", type=str, default="test.tsv")
    parser.add_argument("--output_file", type=str, default="output")

    args = parser.parse_args()

    return args

def get_response(temp_sentences, keywords, bot) :
    ret = []
    count = 0
    for sen in tqdm(temp_sentences) :
        sent_1, sent_2 = [sen.replace("XYZ", keywords[0])], [sen.replace("XYZ", keywords[1])]
        # print(sent_1)
        # print(sent_2)
        res_1, res_2 = bot.make_response(sent_1), bot.make_response(sent_2)
        ret.append(res_1[0])
        ret.append(res_2[0])
        # count += 1
        # if count >= 10 :
        #     break
    
    return ret

def main() :
    parser = ArgumentParser()
    args  = set_arguments(parser)

    opt = yaml.load(open(f"../configs/{args.config}/config.yaml"), Loader=yaml.FullLoader)
    opt.update(vars(args))
    args = argparse.Namespace(**opt)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fix_seed(args)
    
    bot = importlib.import_module(".module",f"bots.{args.bot}").bot
    Bot = bot(args)

    test_sentences = []
    with open(args.test_word_file, 'r') as fp :
        tsv_file = csv.reader(fp, delimiter="\t")

        for line in tsv_file :
            test_sentences.append(line[-1])
    
    mens = ['The man']
    womens = ['The women']
    # with open('../keywords/men.txt') as fp :
    #     for line in fp.read().splitlines() :
    #         mens.append(line)
    # with open('../keywords/women.txt') as fp :
    #     for line in fp.read().splitlines() :
    #         womens.append(line)

    # 2 | idx --> men, others --> women
    to_write = []
    # father, actor, men, brother, king, guys
    selected_idx = [1]
    for i in tqdm(selected_idx):
        _ = get_response(test_sentences, [mens[i-1], womens[i-1]], Bot)
        for x in _ :
            to_write.append(x)

    reply_men, reply_women = [], []

    for i in range(len(to_write)) :
        if i % 2 == 0 :
            reply_men.append(to_write[i])
        else :
            reply_women.append(to_write[i])

    assert(len(reply_men) == len(reply_women))
    with open('data/regard/' + args.output_file + '_men.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for sen in reply_men :
            tsv_writer.writerow(sen)
    
    with open('data/regard/' + args.output_file + '_women.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for sen in reply_women :
            tsv_writer.writerow(sen)
    
if __name__ == '__main__' :
    main()
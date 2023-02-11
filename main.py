
import numpy as np
import torch
import random
import argparse
import importlib
import tqdm
from tqdm import tqdm
import shutil
from pathlib import Path
from argparse import ArgumentParser
import wandb
import os
from argparse import Namespace
from torch.utils.data import DataLoader
import yaml
import warnings
from os.path import join
import json
from copy import deepcopy

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    
    # replace yaml default argument to argparser arg. 
    parser = ArgumentParser()
    args  = set_arguments(parser)
    opt = yaml.load(open(f"configs/{args.config}/config.yaml"), Loader=yaml.FullLoader)
    opt.update(vars(args))

    fix_seed(args)
    args = argparse.Namespace(**opt)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    tags = ['reorg']
    if args.tags:    
        tags.append(args.tags)
    
    if (args.type is None):
        print("which type of task do you want to use: word / emotion")
        raise

    if (args.mode is None):
        print("which mode do you want to use: pretrain / finetune / test")
        raise
    
    wandb.init(mode=args.wandb, project='bias_phase1', tags=tags, name=args.exp_name, entity="chatbot_ntu")
    wandb.config.update(args)

    agent = importlib.import_module('.module', f"agents.{args.agent}").agent
    prompt = importlib.import_module(".module",f"prompts.{args.prompt}").prompt
    bot = importlib.import_module(".module",f"bots.{args.bot}").bot
    dataset =importlib.import_module(".module",f"datasets.{args.dataset}").dataset
    
    Bot = bot(args)
    Prompt = prompt(args)
    Dataset = dataset(args.path, Prompt.tokenizer)
    Agent = agent(args, Prompt, Bot)
    dataloader = DataLoader(Dataset, batch_size=args.bz, shuffle=True, num_workers=0)
    pbar = tqdm(dataloader,position=0)
    batch = 0

    for inputs_id, mask, ll in pbar:
        total_loss = 0
        total_grad = 0
        total_score = 0
        total_mse = 0
        total_pg = 0
        total_entropy = 0
        batch +=1

        if args.mode == 'pretrain':
            meta_total = random.sample(Agent.train_task, k=8)
        
        elif args.mode == 'finetune':
            meta_total = [args.task]
        
        elif args.mode == 'test':
            meta_total = [args.task]
         
        task_bar =tqdm(total=len(meta_total)*args.sample_time*args.k_epoch,position=1,leave=True)
        task_bar.set_description(desc=f"None, epoch: 0, score:{round(0.0, 3)}, loss:{round(0.0, 5)}",refresh=True)
        
        if batch <= args.end_batch:
            
            sample_dicts = []
            scores = []
            ## use input to sample data
            for idx, task in enumerate(meta_total):
                for s in range(args.sample_time):
                    flatten_dict = Agent.sample_forward(inputs_id, mask, \
                        ll, task, Prompt.model_demo, Prompt.state_network_demo)
                    sample_dicts.append(flatten_dict)

            if args.mode != 'test':
                ## k_epoch means how many times the sample data will be used
                for epoch in range(args.k_epoch):
                    sample_acc_loss = 0
                    sample_acc_score = 0
                    random.shuffle(sample_dicts)
                    for flatten_dict in sample_dicts:

                        loss, score, _, _, _ = Agent.train_forward(inputs_id, mask, ll, flatten_dict)
                        score['score'] /= (args.sample_time * len(meta_total))
                        loss /= (args.sample_time * len(meta_total))
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(Prompt.model.parameters(), 1.0)
                        torch.nn.utils.clip_grad_norm_(Prompt.state_network.parameters(), 1.0)

                        sample_acc_loss += loss
                        sample_acc_score += score['score']
                        task_bar.set_description(desc=f"{flatten_dict['task']}, epoch: {epoch}, score:{round(sample_acc_score / args.bz, 4)}, loss:{round(sample_acc_loss.item() / args.bz, 5)}", refresh=True)
                        task_bar.update(1)

                    Prompt.optimizer.step()
                    Prompt.optimizer.zero_grad()
                
                ## after k_epoch, update demo model 
                Prompt.model_demo.load_state_dict(Prompt.model.state_dict())
                Prompt.state_network_demo.load_state_dict(Prompt.state_network.state_dict())
                sample_dicts = []
                Prompt.model_demo.eval()
                Prompt.state_network_demo.eval()
                task_bar.update(1)

                ## eval on input
                with torch.no_grad():
                    Prompt.state_network.eval()
                    Prompt.model.eval()
                    for idx, task in enumerate(meta_total):
                        flatten_dict = Agent.sample_forward(inputs_id, mask, ll, task, \
                            Prompt.model, Prompt.state_network)

                        loss, score, mse, pg_loss, entropy = \
                            Agent.train_forward(inputs_id, mask, ll, flatten_dict)
                        
                        total_loss += loss.item()
                        total_mse += mse
                        total_pg += pg_loss
                        total_entropy += entropy
                        total_score += score['score']
                        scores.append(score)
                Prompt.model.zero_grad()
                Prompt.state_network.zero_grad()

                ## Outer: Upon update demo, we record outer loss/score ( in eval mode )
                if batch % args.log_interval == 0:
                    tqdm.write(f"[Outer loss in batch {batch}]: {round(total_loss / args.bz/len(meta_total), 4)}")
                    tqdm.write(f"[Outer score in batch {batch}]: {round(total_score/args.bz/len(meta_total), 4)}")
                tqdm.write(f"outerloss in batch {batch}: {round(total_loss/args.bz/len(meta_total), 4)}")
                tqdm.write(f"outerscore in batch {batch}: {round(total_score/args.bz/len(meta_total), 4)}")
                Agent.log_wandb(scores, total_loss, total_mse, total_pg, total_entropy, batch)
                # wandb.log({'outerloss': total_loss / len(meta_total) , \
                #     'outermse': total_mse / len(meta_total), \
                #     'outerpg': total_pg / len(meta_total), \
                #     'outerentropy': total_entropy / len(meta_total), \
                #     'outerscore': total_score / args.bz / len(meta_total)}, step=batch)
            else:
                total_scores = []
                for flatten_dict in sample_dicts:
                    total_scores.append(flatten_dict)
                
                Agent.log_wandb(total_scores, 0, 0, 0, 0, batch)

            if batch % args.save_interval == 0:
                
                if args.mode == 'test':

                    dest = f"results/{args.save_path}/"
                    os.makedirs(dest, exist_ok=True)
                    with open(f'results/{args.save_path}/checkpoint-step-{batch}-output.txt', 'w') as f:
                        for flatten_dict in sample_dicts:
                            bot_response=flatten_dict['conversation']
                            prompt=flatten_dict['model_response']
                            input_sentence=flatten_dict['input_string']
                            for i in range(len(bot_response)):
                                sample_splits = bot_response[i][0]
                                sample_bots = bot_response[i][1]
                                sample_prompt = prompt[i]
                                # import pdb
                                # pdb.set_trace()
                                # sample_sentence = input_sentence[i]
                                # f.write(sample_prompt + ", " + sample_sentence + ", " + sample_bot + '\n')
                                f.write(sample_prompt + ". 1: " + sample_splits[0] + ", " + sample_bots[0] + \
                                    '2: '+ sample_splits[1] + ", " + sample_bots[1] + '\n')   
                                print(sample_prompt + ". 1: " + sample_splits[0] + ", " + sample_bots[0] + \
                                    '2: '+ sample_splits[1] + ", " + sample_bots[1])
                else:

                    if batch in [50, 500]:

                        dest = f"results/{args.save_path}/"
                        os.makedirs(dest, exist_ok=True)
                        
                        save_args = deepcopy(args)
                        del save_args.device
                        with open(f'{dest}/args.txt', 'w') as f:
                            json.dump(save_args.__dict__, f, indent=2)

                        torch.save(
                            {k: (v.cpu() if v is not None else None)  # save to cpu tensors
                                for k, v in Prompt.model.state_dict().items()},
                            join(f'results/{args.save_path}/',
                                    f'checkpoint-step-{batch}-prompt.pkl'))
                        torch.save(
                            {k: (v.cpu() if v is not None else None)  # save to cpu tensors
                                for k, v in Prompt.state_network.state_dict().items()},
                            join(f'results/{args.save_path}/',
                                    f'checkpoint-step-{batch}-value.pkl'))

        if batch > args.end_batch:
            torch.save(
                            {k: (v.cpu() if v is not None else None)  # save to cpu tensors
                                for k, v in Prompt.model.state_dict().items()},
                            join(f'results/{args.save_path}/',
                                    f'checkpoint-step-{batch-1}-prompt.pkl'))
            torch.save(
                            {k: (v.cpu() if v is not None else None)  # save to cpu tensors
                                for k, v in Prompt.state_network.state_dict().items()},
                            join(f'results/{args.save_path}/',
                                    f'checkpoint-step-{batch-1}-value.pkl'))
            break

def set_arguments(parser):
    parser.add_argument("--task", type=str, default="none") # for finetune task
    parser.add_argument("--agent", type=str, default="example")
    parser.add_argument("--config", type=str, default="example")
    parser.add_argument("--bot", type=str, default="example")
    parser.add_argument("--prompt", type=str, default="GPT2")
    parser.add_argument("--dataset", type=str, default="Daily") # for finetune task
    parser.add_argument("--path", type=str, default="./data/daily_train_key.json", help="path for dataset")
    parser.add_argument("--mode", type=str, default="test", help="The current option is [ 'pretrain', 'finetune', 'test' ]")
    parser.add_argument("--type", type=str, default=None, help="The current option is [ 'word', 'emotion' ]")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--save_path", type=str, default="") # save path
    parser.add_argument("--model", type=str, default="gpt2-medium") # load prompted model if we need to finetune or testing.
    parser.add_argument("--ratio", type=float, default=4)
    parser.add_argument("--log_interval",type=int, default=50)
    parser.add_argument("--save_interval",type=int, default=25)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--bz", help="batch size", default=8, type=int)
    parser.add_argument("--k_epoch", help="1 batch sample data need to update k times", type=int, default=5)
    parser.add_argument("--discount_r", help="discounted reward coefficient", type=float, default=0.97)
    parser.add_argument("--end_batch",type=int, default=500)
    parser.add_argument("--sample_time", type=int, default=8)
    parser.add_argument("--inner_lr", type=float, default=1e-5)
    parser.add_argument("--outer_lr", type=float, default=1e-5)
    parser.add_argument('--max_pt_len', help="maximum prompt length", type=int, default=10)
    parser.add_argument("--mse_lr", type=float, default=1)
    parser.add_argument("--ep_lr", type=float, default=0.01)
    parser.add_argument("--coh_r", type=float, default=0.01)
    parser.add_argument("--num_testing", type=int, default=1)
    parser.add_argument("--iters", type=str, default=25)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument('--init_step', type=str, default="")
    # parser.add_argument('--extra_label', type=str, default='train_word_list.txt')
    parser.add_argument("--wandb", type=str, default='enabled')

    args = parser.parse_args()

    return args

def fix_seed(args):

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    return

if __name__ == "__main__":
    main()    

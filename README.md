# bias-ppo

Modified from https://github.com/pohanchi/blackboxbot

Overview
---

## Baseline
## RL
### Training 
For training the Reinforcement Learning to provoke bias in LLMs, you should run the following script.
* `--path : the path of data to update lm_loss`
* `--model_name : the path to pretrained test case generator`
* `--bot : LLM to be attacked`
```
python main.py \
  --mode finetune \
  --prompt GPT2 \
  --agent ppo_ptx_kl \
  --path <PRETRAIN DATASET> \
  --model_name <MODEL TO TRAIN> \
  --bot <BOT> \
  --dataset Netflix \
  --type bias \
  --exp_name test \
  --log_interval 5\
  --seed 42 \
  --bz 8 \
  --kl_coef 0.05 \
  --ep_lr 1.0 \
  --k_epoch 5\
  --discount_r 1.0 \
  --end_batch 300 \
  --sample_time 8 \
  --max_pt_len 30 \
  --inner_lr 9e-6 \
  --lm_lr 0.01 \
  --init_step 100 \
  --save_path test \
  --save_interval 10 \
  --wandb disabled

```
### Generate testcases
* `--model_path : finetuned gpt2 checkpoint path`
* `--bot : LLM which is attacked`

```
python3 test.py \
  --model_path <YOUR MODEL CKPT> \
  --bot <BOT> \
```

### Test Results
For testing testcases that gpt2 generated, you should execute the following script. 
* `--prompt_path : the csv file contains test cases`
* `--save_path : the path to save the result file, which is a csv file contains sentiment gaps of the test cases and responses.`
* `--bot : the bot for testing`
  
```
python3 test.py \
--prompt_path ./gpt2_finetune/RL_Result/alpaca/alpaca-distinct1000-test.csv \
--save_path /work/u5273929/bias-ppo/result/result/alpaca-distinct1000-test-reward-3.csv \
--bot alpaca \
```
## Mitigation


#!/bin/sh
#SBATCH --job-name=alpaca
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --account=GOV112004
#SBATCH -o ./log/alpaca
#SBATCH --ntasks-per-node=1

# # testmodel : 481608 : koala_conti
# #           : 481664 : ChatGPT_baseline 

module purge
module load miniconda3
conda activate bias
export TRANSFORMERS_CACHE=/work/u5273929/huggingface_hub
export HF_DATASETS_CACHE=/work/u5273929/huggingface_hub
export HUGGINGFACE_HUB_CACHE=/work/u5273929/huggingface_hub

# python3 test.py \
# --prompt_path /work/u5273929/bias-ppo/gpt2_finetune/RL_Result/ChatGPT/0605_gpt-m_ChatGPT_lmlr0.2_innerlr9e-6_chat-conti_checkpoint-step-100_temp12.csv \
# --save_path 0605_gpt-m_ChatGPT_lmlr0.2_innerlr9e-6_chat-conti_checkpoint-step-100_temp12_test.csv \
# --bot ChatGPT \
# --only_ppl 1

# alpaca
python3 test.py \
--prompt_path ./gpt2_finetune/RL_Result/alpaca/0603ChatGPT_innlr9e-6_lmlr_0.1_kl_coef0.1_alpaca-checkpoint-step-100-prompt.csv \
--save_path alpaca_test.csv \
--bot alpaca \


##koala
# python3 test.py \
# --prompt_path /work/u5273929/bias-ppo/gpt2_finetune/pretrain_data/twitter_comment.csv \
# --save_path /work/u5273929/bias-ppo/result/twitter/gpt4-twitterbaseline.csv \
# --bot gpt4 \

# t=("2" "3" "4" "5")

##chatgpt
# --prompt_path /work/u5273929/bias-ppo/gpt2_finetune/pretrain_data/ChatGPT_test1000_temp_12.csv \
# python3 test.py \
# --prompt_path /work/u5273929/bias-ppo/gpt2_finetune/pretrain_data/ChatGPT_test1000_temp_12.csv \
# --save_path /work/u5273929/bias-ppo/result/baseline_result/ChatGPT_response_ChatGPT_before_rl_3.csv \
# --bot ChatGPT


#gpt4
# python3 test.py \
# --prompt_path /work/u5273929/bias-ppo/gpt2_finetune/pretrain_data/ChatGPT_test1000_temp_12.csv \
# --save_path gpt4_response_ChatGPT_before_rl.csv \
# --bot gpt4 \

# dolly 
# python3 test.py \
# --prompt_path /work/u5273929/bias-ppo/gpt2_finetune/pretrain_data/twitter_comment.csv \
# --save_path /work/u5273929/bias-ppo/result/twitter/dolly-twitterbaseline.csv \
# --bot dolly \

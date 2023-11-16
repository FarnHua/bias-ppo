#!/bin/sh
#SBATCH --job-name=chatgpt_mi_top
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --account=MST112195
##SBATCH -o ./log/mitigate_ablation/chatgpt_mitigate_top_male
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-4

# # testmodel : 481608 : koala_conti
# #           : 481664 : ChatGPT_baseline 

module purge
module load miniconda3
conda activate promptbench
# export TRANSFORMERS_CACHE=/work/b04203058/huggingface_hub
# export HF_DATASETS_CACHE=/work/b04203058/huggingface_hub
# export HUGGINGFACE_HUB_CACHE=/work/b04203058/huggingface_hub

# python3 test.py \
# --prompt_path /work/u5273929/bias-ppo/gpt2_finetune/RL_Result/ChatGPT/0605_gpt-m_ChatGPT_lmlr0.2_innerlr9e-6_chat-conti_checkpoint-step-100_temp12.csv \
# --save_path 0605_gpt-m_ChatGPT_lmlr0.2_innerlr9e-6_chat-conti_checkpoint-step-100_temp12_test.csv \
# --bot ChatGPT \
# --only_ppl 1
# /work/b04203058/new_bias/bias-ppo/gpt2_finetune/RL_Result/LLaMA2/LLaMA2-distinct1000-test.csv
# alpaca
# new_bias/bias-ppo/result/result/alpaca-distinct1000-incontext-reward.csv
model=("LLaMA2_n_hf_system" "LLaMA2_n_hf" "LLaMA2_n_hf_system" "LLaMA2_n_hf") # "LLaMA2_system" "LLaMA2")
# bold=("gpt2_gen.csv" "gpt2_gen.csv") # "chatgpt_gen.csv" "chatgpt_gen.csv")
bold=("female.csv" "female.csv" "male.csv" "male.csv")
# python3 test.py \
# --prompt_path ./gpt2_finetune/RL_Result/${model[$SLURM_ARRAY_TASK_ID-1]}/${model[$SLURM_ARRAY_TASK_ID-1]}-distinct1000-incontext.csv \
# --save_path ./result/result/${model[$SLURM_ARRAY_TASK_ID-1]}-distinct1000-incontext-reward.csv \
# --bot ${model[$SLURM_ARRAY_TASK_ID-1]};
#####BOLD
python3 test.py \
--prompt_path  ./bold/${bold[$SLURM_ARRAY_TASK_ID-1]} \
--save_path ./result/${model[$SLURM_ARRAY_TASK_ID-1]}_${bold[$SLURM_ARRAY_TASK_ID-1]} \
--bot ${model[$SLURM_ARRAY_TASK_ID-1]} \
--only_ppl 1;

# python3 test.py \
# --prompt_path ./${bold[$SLURM_ARRAY_TASK_ID-1]} \
# --save_path ${model[$SLURM_ARRAY_TASK_ID-1]}_${bold[$SLURM_ARRAY_TASK_ID-1]} \
# --bot ${model[$SLURM_ARRAY_TASK_ID-1]};

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



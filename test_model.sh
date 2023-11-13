#!/bin/sh
#SBATCH --job-name=gpt4
#SBATCH --partition=gpNCHC_LLM
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --account=GOV112004
#SBATCH -o ./log/GPT4-rl-score
#SBATCH --ntasks-per-node=4

module purge
module load miniconda3
conda activate bias


# ChatGPT
# python3 score.py \
# --prompt_path /work/u5273929/tmp/bias-ppo/RL_Result/ChatGPT/ChatGPT-distinct1000-incontext.csv \
# --save_path /work/u5273929/tmp/bias-ppo/Baseline_Result/ChatGPT/ChatGPT_incontext.csv \
# --bot ChatGPT \

# LLaMA2_system
# python3 score.py \
# --prompt_path /work/u5273929/tmp/bias-ppo/RL_Result/LLaMA2_system/LLaMA2_system-distinct1000-incontext.csv \
# --save_path /work/u5273929/tmp/bias-ppo/Score_Result/LLaMA2_system/LLaMA2_system_incontext.csv \
# --bot LLaMA2_system \

# LLaMA2
# python3 score.py \
# --prompt_path /work/u5273929/tmp/bias-ppo/RL_Result/LLaMA2/LLaMA2-distinct1000-incontext.csv \
# --save_path /work/u5273929/tmp/bias-ppo/Score_Result/LLaMA2/LLaMA2_incontext.csv \
# --bot LLaMA2 \


# GPT4
export OPENAI_API_KEY=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
python3 score.py \
--prompt_path /work/u5273929/tmp/bias-ppo/RL_Result/GPT4/GPT4-distinct1000-test.csv \
--save_path /work/u5273929/tmp/bias-ppo/Score_Result/GPT4/GPT4_test.csv \
--bot GPT4 \
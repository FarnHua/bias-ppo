#!/bin/sh
#SBATCH --job-name=chatgpt_mi_top
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --account=GOV112004
#SBATCH -o ./log/mitigate_ablation/gpt4_mitigate_top1
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-4

module purge
module load miniconda3
conda activate bias



# model=("koala" "koala" "alpaca" "alpaca" "dolly" "dolly")
# type=("top" "sample" "top" "sample" "top" "sample")
num=("1" "2" "3" "4")
# t=("2" "2" "3" "3")

# python3 mitigate.py \
# --bot ${model[$SLURM_ARRAY_TASK_ID-1]} \
# --type ${type[$SLURM_ARRAY_TASK_ID-1]} \
# --demo_num 5 \
# --testfile /work/u5273929/bias-ppo/gpt2_finetune/pretrain_data/twitter_comment.csv \
# --save_path mitigate_result/${model[$SLURM_ARRAY_TASK_ID-1]}_mitigate_twitter_${type[$SLURM_ARRAY_TASK_ID-1]}_5.csv \
# --demofile /work/u5273929/bias-ppo/result/baseline_result/${model[$SLURM_ARRAY_TASK_ID-1]}_response_ChatGPT_before_rl.csv \


python3 mitigate.py \
--type top \
--save_path /work/u5273929/bias-ppo/result/mitigate_result/gpt4_mitigate_top_${num[$SLURM_ARRAY_TASK_ID-1]}.csv \
--bot gpt4 \
--demo_num ${num[$SLURM_ARRAY_TASK_ID-1]} \
# --demofile /work/u5273929/bias-ppo/result/baseline_result/gpt4_response_ChatGPT_before_rl.csv \

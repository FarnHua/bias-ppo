#!/bin/sh
#SBATCH --job-name=chatgpt_mi_top
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --account=MST112195
##SBATCH -o ./log/mitigate_ablation/chatgp_mitigate_top_male
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-1

module purge
module load miniconda3
conda activate promptbench



# model=("koala" "koala" "alpaca" "alpaca" "dolly" "dolly")
# type=("top" "sample" "top" "sample" "top" "sample")
num=("5" "5" "5" "5" "1" "2" "3" "4") 
# t=("2" "2" "3" "3")
# model=("LLaMA2" "LLaMA2" "LLaMA2" "LLaMA2" "LLaMA2_system" "LLaMA2_system" "LLaMA2_system" "LLaMA2_system") # "LLaMA2_system" "LLaMA2")
# bold=("female.csv" "female.csv" "male.csv" "male.csv")
# model=("LLaMA2" "LLaMA2_system" "LLaMA2" "LLaMA2_system")
# bold=("male.csv" "male.csv" "female.csv" "female.csv")
model=("LLaMA2_system")
bold=("female.csv")
# python3 mitigate.py \
# --bot ${model[$SLURM_ARRAY_TASK_ID-1]} \
# --type ${type[$SLURM_ARRAY_TASK_ID-1]} \
# --demo_num 5 \
# --testfile /work/u5273929/bias-ppo/gpt2_finetune/pretrain_data/twitter_comment.csv \
# --save_path mitigate_result/${model[$SLURM_ARRAY_TASK_ID-1]}_mitigate_twitter_${type[$SLURM_ARRAY_TASK_ID-1]}_5.csv \
# --demofile /work/u5273929/bias-ppo/result/baseline_result/${model[$SLURM_ARRAY_TASK_ID-1]}_response_ChatGPT_before_rl.csv \

# python3 mitigate_origin.py \
# --type sample \
# --save_path result/mitigate_result/${model[$SLURM_ARRAY_TASK_ID-1]}_mitigate_sample_${num[$SLURM_ARRAY_TASK_ID-1]}_${bold[$SLURM_ARRAY_TASK_ID-1]} \
# --bot ${model[$SLURM_ARRAY_TASK_ID-1]} \
# --demo_num ${num[$SLURM_ARRAY_TASK_ID-1]} \
# --testfile ./${bold[$SLURM_ARRAY_TASK_ID-1]};

# python3 mitigate_origin.py \
# --type sample \
# --save_path result/mitigate_result/${model[$SLURM_ARRAY_TASK_ID-1]}_mitigate_sample_${num[$SLURM_ARRAY_TASK_ID-1]}_origin.csv \
# --bot ${model[$SLURM_ARRAY_TASK_ID-1]} \
# --demo_num ${num[$SLURM_ARRAY_TASK_ID-1]};
# --testfile ./bold/${bold[$SLURM_ARRAY_TASK_ID-1]}

python3 mitigate.py \
--type top \
--save_path result/mitigate_result/${model[$SLURM_ARRAY_TASK_ID-1]}_system_before_${num[$SLURM_ARRAY_TASK_ID-1]}_${bold[$SLURM_ARRAY_TASK_ID-1]} \
--bot ${model[$SLURM_ARRAY_TASK_ID-1]} \
--demo_num ${num[$SLURM_ARRAY_TASK_ID-1]} \
--testfile ./bold/${bold[$SLURM_ARRAY_TASK_ID-1]};

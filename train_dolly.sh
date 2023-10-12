#!/bin/sh

#SBATCH --job-name=dolly_abaltion
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --account=MST108253
#SBATCH -o ./log/abaltion/slurm_log_ChatGPT_dolly_alation
#SBATCH --ntasks-per-node=1


## SBATCH --ntasks-per-node=1


#SBATCH --array=1-7
## SBATCH -e /results/slurm/slurm-%j.err
## SBATCH -o /results/slurm/slurm-%j.out

module purge
module load miniconda3
conda activate bias


# lm_lr=("0.3" "0.25")
# lmlr = 0.1, kl = 0.02 
# 0.0, 0.05 : lm_lr
# 0.0, 0.01 : kl
# eps_clip=("0.4" "0.4")
seed=("42")
lr=("9e-6")
kl_coef=("0.0" "0.01" "0.0" "0.01")
# lr=("9e-5")
lm_lr=("0.0" "0.05" "0.05" ""0.0)


for i in {1}
do
                python main.py \
                        --mode finetune \
                        --prompt GPT2 \
                        --agent ppo_ptx_kl \
                        --path /work/u5273929/bias-ppo-br/bias-ppo/gpt2_finetune/pretrain_data/ChatGPT.csv \
                        --model_name /work/u5273929/bias-ppo/gpt2_finetune/gpt2-m/gpt2-m-ChatGPT/checkpoint-2985 \
                        --bot dolly \
                        --dataset Netflix \
                        --type bias \
                        --exp_name 0622_gpt-m_ChatGPT_dolly_lmlr${lm_lr[$SLURM_ARRAY_TASK_ID-1]}_kl${kl_coef[$SLURM_ARRAY_TASK_ID-1]}_ablation \
                        --log_interval 5 \
                        --eps_clip 0.2 \
                        --seed 42 \
                        --bz 8 \
                        --kl_coef ${kl_coef[$SLURM_ARRAY_TASK_ID-1]} \
                        --ep_lr 1.0 \
                        --k_epoch 5\
                        --discount_r 1.0 \
                        --end_batch 250 \
                        --sample_time 8 \
                        --max_pt_len 30 \
			--inner_lr 5e-6 \
                        --lm_lr ${lm_lr[$SLURM_ARRAY_TASK_ID-1]} \
                        --save_path 0622_gpt-m_ChatGPT_dolly_lmlr${lm_lr[$SLURM_ARRAY_TASK_ID-1]}_kl${kl_coef[$SLURM_ARRAY_TASK_ID-1]}_ablation \
                        --save_interval 10 \
                        --wandb online
done


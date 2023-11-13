#!/bin/sh
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p gp4d
#SBATCH --mem=32G
#SBATCH --account=GOV112004
#SBATCH -o ./log/slurm_log_GPT4_RL
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00

##SBATCH --array=1-1
## SBATCH -e /results/slurm/slurm-%j.err
## SBATCH -o /results/slurm/slurm-%j.out



module purge
module load miniconda3
conda activate bias


lr=("9e-6")
lm_lr=("0.1")
seed=("42")
kl_coef=("0.01")
# lr=("9e-5")


DATE=1109

for i in {1}
do
                python main.py \
                        --mode finetune \
                        --prompt GPT2 \
                        --agent ppo_ptx_kl \
                        --path ./gpt2_finetune/pretrain_data/ChatGPT.csv \
                        --model_name /work/u5273929/bias-ppo/gpt2_finetune/gpt2-m/gpt2-m-ChatGPT/checkpoint-2985 \
                        --bot GPT4 \
                        --dataset Netflix \
                        --type bias \
                        --exp_name ${DATE}_gpt-m_GPT4_lmlr${lm_lr[$SLURM_ARRAY_TASK_ID-1]}_innerlr${lr[$SLURM_ARRAY_TASK_ID-1]}_kl${kl_coef[$SLURM_ARRAY_TASK_ID-1]} \
                        --log_interval 5 \
                        --eps_clip 0.2 \
                        --seed 42 \
                        --bz 8 \
                        --kl_coef ${kl_coef[$SLURM_ARRAY_TASK_ID-1]} \
                        --ep_lr 1.0 \
                        --k_epoch 5\
                        --discount_r 1.0 \
                        --end_batch 201 \
                        --sample_time 8 \
                        --max_pt_len 30 \
			            --inner_lr ${lr[$SLURM_ARRAY_TASK_ID-1]} \
                        --lm_lr ${lm_lr[$SLURM_ARRAY_TASK_ID-1]} \
                        --save_path ${DATE}_gpt-m_GPT4_lmlr${lm_lr[$SLURM_ARRAY_TASK_ID-1]}_innerlr${lr[$SLURM_ARRAY_TASK_ID-1]}_kl${kl_coef[$SLURM_ARRAY_TASK_ID-1]} \
                        --save_interval 10 \
                        --wandb online
done
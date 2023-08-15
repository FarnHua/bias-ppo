#!/bin/sh

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --account=MST108253
#SBATCH -o ./log/slurm_log_ChatGPT-toxi
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00


## SBATCH --ntasks-per-node=1

#SBATCH --array=1-1
## SBATCH -e /results/slurm/slurm-%j.err
## SBATCH -o /results/slurm/slurm-%j.out

module purge
module load miniconda3
conda activate bias


lr=("9e-6" "5e-6" "5e-6" "5e-6")
lm_lr=("0.1" "0.1" "0.2" "0.2")
seed=("42")
kl_coef=("0.01" "0.02" "0.01" "0.02")
# lr=("9e-5")


for i in {1}
do
                python main.py \
                        --mode finetune \
                        --prompt GPT2 \
                        --agent toxi \
                        --path /work/u5273929/bias-ppo-br/bias-ppo/gpt2_finetune/pretrain_data/ChatGPT.csv \
                        --model_name /work/u5273929/bias-ppo/gpt2_finetune/gpt2-m/gpt2-m-ChatGPT/checkpoint-2985 \
                        --bot ChatGPT \
                        --dataset Netflix \
                        --type bias \
                        --exp_name 0711_gpt-m_ChatGPT_lmlr${lm_lr[$SLURM_ARRAY_TASK_ID-1]}_innerlr${lr[$SLURM_ARRAY_TASK_ID-1]}_kl${kl_coef[$SLURM_ARRAY_TASK_ID-1]}_toxi \
                        --log_interval 5 \
                        --eps_clip 0.2 \
                        --seed 42 \
                        --bz 8 \
                        --kl_coef ${kl_coef[$SLURM_ARRAY_TASK_ID-1]} \
                        --ep_lr 1.0 \
                        --k_epoch 5\
                        --discount_r 1.0 \
                        --end_batch 300 \
                        --sample_time 8 \
                        --max_pt_len 30 \
			--inner_lr ${lr[$SLURM_ARRAY_TASK_ID-1]} \
                        --lm_lr ${lm_lr[$SLURM_ARRAY_TASK_ID-1]} \
                        --save_path 0711_gpt-m_ChatGPT_lmlr${lm_lr[$SLURM_ARRAY_TASK_ID-1]}_innerlr${lr[$SLURM_ARRAY_TASK_ID-1]}_kl${kl_coef[$SLURM_ARRAY_TASK_ID-1]}_toxi \
                        --save_interval 10 \
                        --wandb online
done


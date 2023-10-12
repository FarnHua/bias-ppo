#!/bin/sh

#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=40G
#SBATCH --account=MST111160
#SBATCH -o ./log/slurm_log_ChatGPT_koala
#SBATCH --ntasks-per-node=1



#SBATCH --array=1
## SBATCH -e /results/slurm/slurm-%j.err
## SBATCH -o /results/slurm/slurm-%j.out



module purge
module load miniconda3
conda activate bias


# lm_lr=("0.3" "0.25")
lm_lr=("0.1")
seed=("42")
lr=("5e-6")
kl_coef=("0.02")
eps_clip=("0.2")
# lr=("9e-5")





WANDB_CACHE_DIR=/work/u5273929/bias-ppo

python main.py \
--mode finetune \
--prompt GPT2 \
--agent ppo_ptx_kl \
--path /work/u5273929/bias-ppo-br/bias-ppo/gpt2_finetune/pretrain_data/ChatGPT.csv \
--model_name /work/u5273929/bias-ppo/gpt2_finetune/gpt2-m/gpt2-m-ChatGPT/checkpoint-2985 \
--model_ckpt 0604_gpt-m_ChatGPT_lmlr0.1_koala \
--bot koala \
--dataset Netflix \
--type bias \
--exp_name 0604_gpt-m_ChatGPT_lmlr${lm_lr[$SLURM_ARRAY_TASK_ID-1]}_koala-conti \
--log_interval 5 \
--seed 42 \
--bz 8 \
--kl_coef ${kl_coef[$SLURM_ARRAY_TASK_ID-1]} \
--ep_lr 1.0 \
--eps_clip ${eps_clip[$SLURM_ARRAY_TASK_ID-1]} \
--k_epoch 5 \
--discount_r 1.0 \
--end_batch 200 \
--sample_time 8 \
--max_pt_len 30 \
--inner_lr ${lr[$SLURM_ARRAY_TASK_ID-1]} \
--lm_lr ${lm_lr[$SLURM_ARRAY_TASK_ID-1]} \
--init_step 100 \
--save_path 0604_gpt-m_ChatGPT_lmlr${lm_lr[$SLURM_ARRAY_TASK_ID-1]}_koala-conti \
--save_interval 20 \
--wandb online



#!/bin/sh

#SBATCH --job-name=AL8GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=32G
#SBATCH --account=MST108253
#SBATCH -o ./log/toxi_training/slurm_log_alpaca_8gpu
#SBATCH --ntasks-per-node=1


# SBATCH --array=1-1

module purge
module load miniconda3
conda activate bias


# lm_lr=("0.3" "0.25")
# lm_lr=("0.1")
# seed=("42")
# lr=("9e-6")
# kl_coef=("0.1")
# lr=("9e-5")

for i in {1}
do
                python main.py \
                        --mode finetune \
                        --prompt GPT2 \
                        --agent toxi \
                        --path /work/u5273929/bias-ppo-br/bias-ppo/gpt2_finetune/pretrain_data/ChatGPT.csv \
                        --model_name /work/u5273929/bias-ppo/gpt2_finetune/gpt2-m/gpt2-m-ChatGPT/checkpoint-2985 \
                        --bot alpaca \
                        --dataset Netflix \
                        --type bias \
                        --exp_name 0703Alpaca_innlr9e-6_lmlr_0.1_kl_coef0.1_toxi8gpu \
                        --log_interval 5\
                        --seed 42 \
                        --bz 8 \
                        --kl_coef 0.1 \
                        --ep_lr 1.0 \
                        --k_epoch 5\
                        --discount_r 1.0 \
                        --end_batch 300 \
                        --sample_time 8 \
                        --max_pt_len 30 \
			--inner_lr 9e-6 \
                        --lm_lr 0.1 \
                        --init_step 0 \
                        --save_path 0703Alpaca_innlr9e-6_lmlr_0.1_kl_coef0.1_toxi8gpu \
                        --save_interval 20 \
                        --wandb disabled
done


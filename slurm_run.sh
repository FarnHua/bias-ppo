#!/bin/sh

#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --account=GOV112004
#SBATCH --cpus-per-task=4
#SBATCH -o ./log/slurm_log_ChatGPT_llama
#SBATCH --ntasks-per-node=1


## SBATCH --ntasks-per-node=1


#SBATCH --array=1
## SBATCH -e /results/slurm/slurm-%j.err
## SBATCH -o /results/slurm/slurm-%j.out



module purge
module load miniconda3
conda activate bias

# lm_lr=("0.3" "0.25")
# lm_lr=("0.15")
seed=("42")
# lr=("9e-5" "1e-5" "1e-6" "3e-7")
lr=("9e-5")

for i in {1}
do
                python main.py \
                        --mode finetune \
                        --prompt GPT2 \
                        --agent ppo_ptx_kl \
                        --path /work/u5273929/bias-ppo-br/bias-ppo/gpt2_finetune/pretrain_data/ChatGPT.csv \
                        --model_name /work/u5273929/bias-ppo-br/bias-ppo/gpt2_finetune/gpt2-l/gpt2-l-ChatGPT/checkpoint-2985\
                        --bot LLaMA \
                        --dataset Netflix \
                        --type bias \
                        --exp_name 0521ChatGPT_testcase_lmlr_${lr[$SLURM_ARRAY_TASK_ID-1]}_len-30_LLaMA \
                        --log_interval 5\
                        --seed 42 \
                        --bz 8 \
                        --kl_coef 0.05 \
                        --ep_lr 1.0 \
                        --k_epoch 5\
                        --discount_r 1.0 \
                        --end_batch 400 \
                        --sample_time 8 \
                        --max_pt_len 30 \
			--inner_lr ${lr[$SLURM_ARRAY_TASK_ID-1]} \
                        --lm_lr 0.1\
                        --init_step 0 \
                        --save_path 0521ChatGPT_testcase_lmlr_${lr[$SLURM_ARRAY_TASK_ID-1]}_len-30_LLaMA \
                        --save_interval 20 \
                        --wandb online
done

#!/bin/sh
#SBATCH --job-name=ChatGPT-test-case
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --account=MST108253
#SBATCH --cpus-per-task=1
#SBATCH -o ../log/slurm_log_dolly_test
#SBATCH --ntasks-per-node=1


module purge
module load miniconda3
conda activate bias


## alpaca
# python test.py --bot alpaca --model_path /work/u5273929/bias-ppo/results/0603ChatGPT_innlr9e-6_lmlr_0.1_kl_coef0.1_alpaca/checkpoint-step-100-prompt.pkl

## koala
# python test.py --bot koala --model_path /work/u5273929/bias-ppo/results/0604_gpt-m_ChatGPT_lmlr0.1_koala-conti/checkpoint-step-100-prompt.pkl

## ChatGPT
# python test.py --bot ChatGPT --model_path /work/u5273929/bias-ppo/results/0605_gpt-m_ChatGPT_lmlr0.2_innerlr9e-6_chat-conti/checkpoint-step-100-prompt.pkl

## gpt4
# python test.py --bot gpt4 --model_path /work/u5273929/bias-ppo/results/0605_gpt-m_ChatGPT_lmlr0.2_innerlr9e-6_gpt4-conti/checkpoint-step-80-prompt.pkl

## dolly
python test.py --bot dolly --model_path /work/u5273929/bias-ppo/results/0616_gpt-m_ChatGPT_lmlr0.1_dolly_lmlr0.1/checkpoint-step-200-prompt.pkl 
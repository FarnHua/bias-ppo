#!/bin/sh

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --account=GOV112004
#SBATCH --cpus-per-task=4
#SBATCH -o ../log/slurm_log_gpt2l_chat
#SBATCH --ntasks-per-node=1
export TRANSFORMERS_CACHE=/work/u5273929/huggingface_hub
export HF_DATASETS_CACHE=/work/u5273929/huggingface_hub
export HUGGINGFACE_HUB_CACHE=/work/u5273929/huggingface_hub

module purge
module load miniconda3
conda activate bias

python ppl.py --file_path /work/u5273929/tmp/bias-ppo/Mitigate_Result/GPT4/GPT4_mitigate_human_origin.csv
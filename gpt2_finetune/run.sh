#!/bin/sh

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --account=GOV112004
#SBATCH --cpus-per-task=4
#SBATCH -o ../log/slurm_log_gpt2l_chat
#SBATCH --ntasks-per-node=1


module purge
module load miniconda3
conda activate bias

python ppl.py --file_path /work/u5273929/in-context/result/eval_ppl_blenderbot_without_FT.csv


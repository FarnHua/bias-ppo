#!/bin/sh

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --account=MST111160
#SBATCH -o ../log/generate_testcase
#SBATCH --ntasks-per-node=1

module purge
module load miniconda3
conda activate bias

python test.py
#!/bin/sh
#SBATCH --job-name=ppl_dolly_chatgpt_response_ppl
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --account=MST108253
#SBATCH --cpus-per-task=1
#SBATCH -o ../log/ppl/ppl_dolly_chatgpt_response_ppl
#SBATCH --ntasks-per-node=1


module purge
module load miniconda3
conda activate bias


# alpaca
# python ppl.py --file_path /work/u5273929/bias-ppo/result/baseline_result/alpaca_response_ChatGPT_before_rl.csv

# # koala
# python ppl.py --file_path /work/u5273929/bias-ppo/result/mitigate_result/koala_mitigate_human_1.csv

# # gpt4
# python ppl.py --file_path /work/u5273929/bias-ppo/result/baseline_result/gpt4_response_ChatGPT_before_rl.csv


# chatgpt
# python ppl.py --file_path /work/u5273929/bias-ppo/result/baseline_result/ChatGPT_response_ChatGPT_before_rl.csv

# dolly
python ppl.py --file_path /work/u5273929/bias-ppo/result/baseline_result/dolly_response_ChatGPT_before_rl.csv


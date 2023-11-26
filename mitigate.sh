#!/bin/sh
#SBATCH --job-name=gpt4-mi-human
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=GOV112004
#SBATCH --partition gpNCHC_LLM
#SBATCH -o ./log/mitigate_ablation/gpt4-miti-human
#SBATCH --ntasks-per-node=1
##SBATCH --array=1-5
#SBATCH --cpus-per-task=4

module purge
module load miniconda3
conda activate bias




# type=("top" "top" "top" "top" "top")
type=("sample" "sample" "sample" "sample" "sample")
num=("1" "2" "3" "4" "5")
# t=("2" "2" "3" "3")
# model=("GPT4" "GPT4" "GPT4" "GPT4" "GPT4")
model=("ChatGPT" "ChatGPT" "ChatGPT" "ChatGPT" "ChatGPT")

export OPENAI_API_KEY=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
# python3 mitigate_origin.py \
# --type ${type[$SLURM_ARRAY_TASK_ID-1]} \
# --save_path Mitigate_Result/${model[$SLURM_ARRAY_TASK_ID-1]}/${model[$SLURM_ARRAY_TASK_ID-1]}_mitigate_${type[$SLURM_ARRAY_TASK_ID-1]}_${num[$SLURM_ARRAY_TASK_ID-1]}_origin.csv \
# --bot ${model[$SLURM_ARRAY_TASK_ID-1]} \
# --demo_num ${num[$SLURM_ARRAY_TASK_ID-1]} \
# --testfile RL_Result/${model[$SLURM_ARRAY_TASK_ID-1]}/${model[$SLURM_ARRAY_TASK_ID-1]}-distinct1000-test.csv \
# --demofile Score_Result/${model[$SLURM_ARRAY_TASK_ID-1]}/${model[$SLURM_ARRAY_TASK_ID-1]}-incontext.csv

Model=GPT4
Type=human
for i in 5
do
    python3 mitigate_origin.py \
        --type $Type \
        --save_path ${Model}_mitigate_${Type}_origin.csv \
        --bot ${Model} \
        --demo_num 0 \
        --testfile RL_Result/${Model}/${Model}-distinct1000-test.csv \
        --demofile Score_Result/${Model}/${Model}-incontext.csv
done

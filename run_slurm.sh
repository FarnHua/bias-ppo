#!/bin/bash
## SLURM scripts have a specific format. 

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=no_mem_ids_exploration

## filename for job standard output (stdout)
## %j is the job id, %u is the user id
## SBATCH --output=/checkpoints/jacampos/experiments/vl-seq2seq/exploration/no_mem_ids_exploration_%A_%a.out

## filename for job standard error output (stderr)
## SBATCH --error=/checkpoints/jacampos/experiments/vl-seq2seq/exploration/no_mem_ids_exploration_%A_%a.err

## partition name
##SBATCH --partition=hipri
## number of gpus
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --account=MST108253
#MST111348
## number of tasks per node
#SBATCH --ntasks-per-node=1

#SBATCH --array=1-4
##SBATCH -e /checkpoints/hsuansu/slurm/slurm-%j.err
##SBATCH -o /checkpoints/hsuansu/slurm/slurm-%j.out
### Section 2: Setting environment variables for the job
### Remember that all the module command does is set environment
### variables for the software you need to. Here I am assuming I
### going to run something with python.
### You can also set additional environment variables here and
### SLURM will capture all of them for each task
# Start clean
# Load what we need
module purge
ml miniconda3 cuda/11.7 gcc8/8.3.1
conda info --envs
conda activate ppo

lr=("5e-6" "5e-6" "5e-6" "5e-6")
kl_coef=("0.02" "0.01" "0.1" "0.005")
for i in 1e-5
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
                        --mode finetune \
                        --prompt GPT2 \
                        --agent ppo_ptx_kl \
                        --dataset Netflix \
			--model gpt2-medium \
                        --bot blenderbot \
                        --type bias \
                        --exp_name up_kl_dis_1_nt_20_${kl_coef[$SLURM_ARRAY_TASK_ID-1]}_${lr[$SLURM_ARRAY_TASK_ID-1]}-blender_medium_ppo_ipx \
                        --log_interval 25 \
                        --seed 1 \
                        --kl_coef ${kl_coef[$SLURM_ARRAY_TASK_ID-1]} \
                        --bz 8 \
                        --k_epoch 5 \
                        --discount_r 1.0 \
                        --end_batch 1040 \
                        --sample_time 8 \
                        --max_pt_len 20 \
                        --tags inner-lr \
			--inner_lr ${lr[$SLURM_ARRAY_TASK_ID-1]}\
                        --init_step 2 \
                        --save_path up_kl_dis_1_nt_20_${kl_coef[$SLURM_ARRAY_TASK_ID-1]}_${lr[$SLURM_ARRAY_TASK_ID-1]}-blender_medium_ppo_ipx \
                        --save_interval 20 \
                        --ep_lr 1.0 \
                        --lm_lr 0.2 \
                        --no-update_demo \
                        --wandb online
done
# for i in 1e-5 
# do
#     CUDA_VISIBLE_DEVICES=0 python main.py \
#                         --mode test \
#                         --prompt GPT2 \
#                         --bot blenderbot \
#                         --type bias \
#                         --exp_name bias-ppo2-inlr${i}-blender_large_ppo_ipx_lm_only \
#                         --log_interval 25 \
#                         --seed 1 \
#                         --bz 8 \
#                         --k_epoch 5 \
#                         --discount_r 0.97 \
#                         --end_batch 1000 \
#                         --sample_time 8 \
#                         --max_pt_len 10 \
#                         --tags inner-lr \
# 			--inner_lr ${i}\
#                         --init_step 50 \
#                         --save_path bias-ppo2-inlr${i}-2-blender_large_ppo_ipx_lm_only \
#                         --model results/bias-ppo2-inlr${i}-2-blender_large_ppo_ipx_lm_only \
#                         --save_interval 20 \
#                         --wandb online
# done

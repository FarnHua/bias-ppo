export TRANSFORMERS_CACHE=/work/u5273929/huggingface_hub
export HF_DATASETS_CACHE=/work/u5273929/huggingface_hub
export HUGGINGFACE_HUB_CACHE=/work/u5273929/huggingface_hub

for i in {1}
do
                python main.py \
                        --mode finetune \
                        --prompt GPT2 \
                        --agent ppo_ptx_kl \
                        --path ./gpt2_finetune/pretrain_data/ChatGPT.csv \
                        --model_name /work/u5273929/bias-ppo/gpt2_finetune/gpt2-m/gpt2-m-ChatGPT/checkpoint-2985 \
                        --bot alpaca \
                        --dataset ChatGPT \
                        --type bias \
                        --exp_name test \
                        --log_interval 5\
                        --seed 42 \
                        --bz 8 \
                        --kl_coef 0.05 \
                        --ep_lr 1.0 \
                        --k_epoch 5\
                        --discount_r 1.0 \
                        --end_batch 300 \
                        --sample_time 8 \
                        --max_pt_len 30 \
			--inner_lr 9e-6 \
                        --lm_lr 0.01 \
                        --init_step 100 \
                        --save_path test \
                        --save_interval 10 \
                        --wandb disabled
done

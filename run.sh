for i in {1}
do
                python main.py \
                        --mode finetune \
                        --prompt GPT2 \
                        --agent ppo_ptx_kl \
                        --path /work/u5273929/bias-ppo-br/bias-ppo/gpt2_finetune/pretrain_data/ChatGPT.csv \
                        --model_name /work/u5273929/bias-ppo/gpt2_finetune/gpt2-m/gpt2-m-ChatGPT/checkpoint-2985 \
                        --model_ckpt 0602ChatGPT_innlr9e-6_lmlr_0.1_kl_coef0.05_dolly \
                        --bot dolly \
                        --dataset Netflix \
                        --type bias \
                        --exp_name 0603_gpt_dolly_conti \
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
                        --save_path 0603_dolly_conti \
                        --save_interval 10 \
                        --wandb disabled
done

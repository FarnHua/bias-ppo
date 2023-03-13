for i in 1e-4
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
                        --mode test \
                        --prompt GPT2 \
			--model gpt2-medium\
                        --bot blenderbot \
                        --type bias \
                        --exp_name two_token-lr${i}-top \
                        --log_interval 25 \
                        --seed 1 \
                        --bz 8 \
                        --k_epoch 5 \
                        --discount_r 0.98 \
                        --coh_r 0.5\
                        --end_batch 2 \
                        --sample_time 1 \
                        --max_pt_len 10 \
                        --tags top \
			--inner_lr ${i}\
                        --init_step 20 \
                        --save_path two_token-lr${i}-top\
                        --save_interval 1 \
                        --wandb disabled
done
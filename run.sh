for i in 1e-6  
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
                        --mode finetune \
                        --prompt GPT2 \
			--model gpt2-medium\
                        --bot blenderbot \
                        --type bias \
                        --exp_name bias-ppo2-ipx-${i}-full \
                        --log_interval 5 \
                        --seed 1 \
                        --bz 8 \
                        --k_epoch 5 \
                        --discount_r 0.98 \
                        --end_batch 500 \
                        --sample_time 16 \
                        --max_pt_len 10 \
                        --tags full_data \
			--inner_lr ${i}\
                        --init_step 2 \
                        --save_path bias-ppo2-ipx-${i}-full\
                        --save_interval 20 \
                        --wandb online
done

for i in 1e-4 
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
                        --mode finetune \
                        --prompt GPT2 \
			--model gpt2-medium\
                        --bot DialogGPT \
                        --type bias \
                        --exp_name bias-ppo2-inlr${i}-2 \
                        --log_interval 25 \
                        --seed 1 \
                        --bz 8 \
                        --k_epoch 5 \
                        --discount_r 0.98 \
                        --end_batch 1000 \
                        --sample_time 2 \
                        --max_pt_len 10 \
                        --tags inner-lr \
			--inner_lr ${i}\
                        --init_step 2 \
                        --save_path bias-ppo2-inlr${i}-2\
                        --save_interval 20 \
                        --wandb online
done

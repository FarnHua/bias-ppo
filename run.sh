for i in 1 1000 10000 100000
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
                        --mode finetune \
                        --prompt GPT2 \
                        --bot DialogGPT \
                        --type bias \
                        --exp_name bias-ppo2-seed${i} \
                        --log_interval 25 \
                        --seed ${i} \
                        --bz 8 \
                        --k_epoch 5 \
                        --discount_r 0.98 \
                        --end_batch 800 \
                        --sample_time 1 \
                        --max_pt_len 10 \
                        --tags seed \
                        --init_step 2 \
                        --save_path bias-ppo2-seed${i} \
                        --save_interval 20 \
                        --wandb enabled
done

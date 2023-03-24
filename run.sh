for i in 9e-6
do
     python main.py \
                        --mode finetune \
                        --prompt GPT-Neo \
			--model gpt_neo\
                        --bot blenderbot \
                        --dataset Netflix \
                        --type bias \
                        --exp_name neo-${i} \
                        --log_interval 5 \
                        --seed 1 \
                        --bz 4 \
                        --k_epoch 5 \
                        --discount_r 0.98 \
                        --end_batch 1040 \
                        --sample_time 6 \
                        --max_pt_len 20 \
                        --tags full_data \
			--inner_lr ${i}\
                        --lm_lr 1.0\
                        --init_step 2 \
                        --save_path neo-${i}\
                        --save_interval 20 \
                        --wandb online
done

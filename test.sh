for i in 9e-6 
do
     python main.py \
                        --mode test \
                        --prompt GPT-Neo \
			          --model results/neo-9e-6\
                        --bot blenderbot \
                        --dataset Netflix \
                        --type bias \
                        --exp_name neo-9e-6 \
                        --log_interval 1 \
                        --seed 1 \
                        --bz 8 \
                        --k_epoch 5 \
                        --discount_r 0.98 \
                        --end_batch 2 \
                        --sample_time 8 \
                        --max_pt_len 40 \
                        --tags inner-lr \
			         --inner_lr ${i}\
                        --init_step 1000 \
                        --save_path neo-9e-6\
                        --save_interval 1 \
                        --wandb disabled
done

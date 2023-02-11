

CUDA_VISIBLE_DEVICES=0 python main.py \
                       --task '<annoyance>' \
                       --mode finetune \
                       --prompt DialogGPT \
                       --bot DialogGPT \
                       --type emotion \
                       --exp_name emotion-test-ii \
                       --log_interval 25 \
                       --seed 100 \
                       --bz 8 \
                       --k_epoch 5 \
                       --discount_r 0.98 \
                       --end_batch 4 \
                       --sample_time 1 \
                       --max_pt_len 10 \
                       --tags debug \
                       --init_step 2 \
                       --save_path debug-ft-emo-ii \
                       --model results/debug-emo-ii \
                       --save_interval 2


CUDA_VISIBLE_DEVICES=0 python main.py \
                       --task 'election' \
                       --mode test \
                       --prompt DialogGPT \
                       --bot blenderbot \
                       --type word \
                       --exp_name word-test-blenderbot-ii \
                       --log_interval 25 \
                       --seed 100 \
                       --bz 8 \
                       --k_epoch 5 \
                       --discount_r 0.98 \
                       --end_batch 2 \
                       --sample_time 1 \
                       --max_pt_len 10 \
                       --tags debug \
                       --init_step 2 \
                       --save_path debug-test-word-blenderbot-ii \
                       --model results/debug-ft-word-blenderbot-ii \
                       --extra_label test_word_list.txt \
                       --save_interval 2

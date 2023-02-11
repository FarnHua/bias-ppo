

CUDA_VISIBLE_DEVICES=0 python main.py \
                       --task None \
                       --mode pretrain \
                       --prompt InputIndependent \
                       --bot blenderbot \
                       --type emotion \
                       --exp_name emotion-pretrain-blenderbot \
                       --save_path debug-emo-blenderbot-ii \
                       --model microsoft/DialoGPT-medium \
                       --log_interval 25 \
                       --seed 100 \
                       --bz 8 \
                       --k_epoch 5 \
                       --discount_r 0.98 \
                       --end_batch 4 \
                       --sample_time 1 \
                       --max_pt_len 10 \
                       --tags debug \
                       --save_interval 2


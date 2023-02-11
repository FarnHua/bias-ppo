

CUDA_VISIBLE_DEVICES=0 python main.py \
                       --task None \
                       --mode pretrain \
                       --prompt DialogGPT \
                       --bot DialogGPT \
                       --type word \
                       --exp_name word-pretrain \
                       --save_path word-pretrain \
                       --model microsoft/DialoGPT-medium \
                       --log_interval 25 \
                       --seed 100 \
                       --bz 8 \
                       --k_epoch 5 \
                       --discount_r 0.97 \
                       --end_batch 51 \
                       --sample_time 8 \
                       --max_pt_len 10 \
                       --tags debug \
                       --save_interval 25 \
                       --coh_r $1

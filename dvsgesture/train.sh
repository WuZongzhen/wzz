CUDA_VISIBLE_DEVICES=2 python train.py --tb --amp --output-dir ./logs --model PlainNet --device cuda:0 --lr-step-size 64 --epoch 192 --T_train 8 --T 8 --data-path /home/wuzongzhen/wzz-HarDvs/tools/data/HARD/
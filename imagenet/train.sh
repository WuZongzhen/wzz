## spiking_resnet
CUDA_VISIBLE_DEVICES=0 python train.py -b 32 --cos_lr_T 320 --model spiking_resnet18  --output-dir ./logs --tb --print-freq 4096 --amp --cache-dataset --T 4 --lr 0.1 --epoch 320  --device cuda:0 --zero_init_residual --data-path /home/wuzongzhen/wzz-HarDvs/tools/data/HARD/
# 
## sew_resnet18
# CUDA_VISIBLE_DEVICES=2 python train.py -b 32 --cos_lr_T 320 --model sew_resnet18  --output-dir ./logs --tb --print-freq 4096 --amp --cache-dataset --connect_f ADD --T 4 --lr 0.1 --epoch 320 --data-path /home/wuzongzhen/wzz-HarDvs/tools/data/HARD/ --device cuda:0



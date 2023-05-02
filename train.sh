# export CUDA_VISIBLE_DEVICES=$1
# =================================================================================
# Train CTCNet
# =================================================================================

python train.py --gpus 1 --name CTCNet_S16_V4_Attn2D --model ctcnet \
    --Gnorm "bn" --lr 0.0002 --beta1 0.9 --scale_factor 8 --load_size 128 \
    --dataroot "/home/ggw/ZiXiangXu/DataSets/spation_18000/" --dataset_name celeba --batch_size 10 --total_epochs 100 \
    --visual_freq 100 --print_freq 50 --save_latest_freq 1000 #--continue_train 

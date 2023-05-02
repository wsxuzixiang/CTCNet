CUDA_VISIBLE_DEVICES=0 python test.py --gpus 1 --model drn --name SPARNet_S16_V4_Attn2D \
    --load_size 128 --dataset_name single --dataroot "/home2/ZiXiangXu/test_HR" \
    --pretrain_model_path "/home2/ZiXiangXu/best_pth" \
    --save_as_dir "/home2/ZiXiangXu/test_results"

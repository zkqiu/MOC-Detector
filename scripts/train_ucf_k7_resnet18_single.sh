cd src

PATH_TO_SAVE_MODEL=../experiment/result_model/K7_rgb_imagenet_resnet18_single

GPU=0,1,2,3
CUDA_VISIBLE_DEVICES=$GPU python3 train.py --K 7 --exp_id Train_K7_rgb_imagenet_resnet18_single --rgb_model $PATH_TO_SAVE_MODEL --batch_size 96 --master_batch 24 --lr 5e-4 --gpus $GPU --num_workers 16 --num_epochs 10 --lr_step 5,8 --save_all --arch resnet_18 --pretrain_model imagenet

cd ..
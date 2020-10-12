cd src

GPU=0
PATH_TO_SAVE_MODEL=./../experiment/result_model/Train_K7_rgb_coco_resnet18_pdb

CUDA_VISIBLE_DEVICES=$GPU python3 train.py --K 7 --exp_id Train_K7_rgb_coco_resnet18_pdb --rgb_model $PATH_TO_SAVE_MODEL --batch_size 1 --master_batch 1 --lr 5e-4 --gpus $GPU --num_workers 16 --num_epochs 10 --lr_step 5,8 --save_all --arch resnet_18

cd ..

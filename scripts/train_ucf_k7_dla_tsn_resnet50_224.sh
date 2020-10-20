cd src


GPU=0,1,2,3
PATH_TO_SAVE_MODEL=./../experiment/result_model/K7_flow_coco_dla34_tsn_resnet50_224

CUDA_VISIBLE_DEVICES=$GPU python3 train.py --K 7 --exp_id Train_K7_rgb_coco --rgb_model $PATH_TO_SAVE_MODEL --modality rgb --batch_size 24 --master_batch 6 --lr 5e-2 --gpus $GPU --num_workers 16 --num_epochs 12 --lr_step 6,8 --save_all --rec_arch tsn_resnet_50

# CUDA_VISIBLE_DEVICES=$GPU python3 train.py --K 7 --exp_id Train_K7_flow_coco --flow_model $PATH_TO_SAVE_MODEL --batch_size 62 --master_batch 6 --lr 5e-4 --gpus $GPU --num_workers 24 --num_epochs 10 --lr_step 6,8 --ninput 5 --save_all




cd ..
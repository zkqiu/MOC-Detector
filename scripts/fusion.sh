cd src



# python3 train.py --K 7 --exp_id Train_K7_rgb_coco --rgb_model ../experiment/result_model/K7_rgb_coco_dla34 --batch_size 32 --master_batch 7 --lr 5e-4 --gpus 0,1,2,3 --num_workers 16 --num_epochs 12 --lr_step 6,8 --save_all

# python3 train.py --K 7 --exp_id Train_K7_flow_coco --flow_model ../experiment/result_model/K7_flow_coco_dla34 --batch_size 30 --master_batch 6 --lr 5e-4 --gpus 0,1,2,3 --num_workers 16 --num_epochs 10 --lr_step 6,8 --ninput 5 --save_all #--start_epoch 3

# -------------------- Evaluation --------------------
PATH_TO_RGB_MODEL='K7_rgb_coco_dla34/model_[12]_2020-10-03-06-32.pth'
PATH_TO_FLOW_MODEL='K7_flow_coco_dla34/model_[10]_2020-10-13-11-08.pth'
INFERENCE_DIR='experiment/result_model/K7_rgb_flow_coco_dla34/inference'

python3 det.py --task normal --K 7 --gpus 0,1,2,3 --batch_size 32 --master_batch 8 --num_workers 8 --rgb_model ../experiment/result_model/$PATH_TO_RGB_MODEL --flow_model ../experiment/result_model/$PATH_TO_FLOW_MODEL --inference_dir ../$INFERENCE_DIR --flip_test --ninput 5
# # python3 det.py --task normal --K 7 --gpus 0 --batch_size 1 --master_batch 1 --num_workers 0 --rgb_model ../experiment/result_model/$PATH_TO_RGB_MODEL --inference_dir $INFERENCE_DIR --flip_test --arch resnet_18
# #
python3 ACT.py --task frameAP --K 7 --th 0.5 --inference_dir ../$INFERENCE_DIR
# #
python3 ACT.py --task BuildTubes --K 7 --inference_dir ../$INFERENCE_DIR
# #
python3 ACT.py --task videoAP --K 7 --th 0.2 --inference_dir ../$INFERENCE_DIR
# python3 ACT.py --task videoAP --K 7 --th 0.5 --inference_dir ../$INFERENCE_DIR
# python3 ACT.py --task videoAP --K 7 --th 0.75 --inference_dir ../$INFERENCE_DIR
# python3 ACT.py --task videoAP_all --K 7 --inference_dir ../$INFERENCE_DIR


cd ..

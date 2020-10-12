cd src
PATH_TO_RGB_MODEL=../experiment/modelzoo/ucf_dla34_K7_rgb_coco.pth
INFERENCE_DIR=../experiment/modelzoo/ucf_dla34_K7_rgb_coco

python3 det.py --task normal --K 7 --gpus 0 --batch_size 10 --master_batch 10 --num_workers 8 --rgb_model $PATH_TO_RGB_MODEL --inference_dir $INFERENCE_DIR --flip_test

cd ..

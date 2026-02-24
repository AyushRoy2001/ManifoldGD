export CUDA_VISIBLE_DEVICES=5
IMAGENET_FOLDER=/data/aroy25/ImageNet100

IPC=10
T=50 # STEPS
T_SG=25 # STOP GUIDANCE
SPEC=imagenet100
METHOD=ddpm

for ((i=0; i < 3; i++))
do

OUTPUT_DATASET=results-new/Base/$METHOD/dit-distillation-sampling-t-$T_SG/imagenet100-$i-IPC-$IPC
TRAIN_SAVE_DIR=results-new/Base/$METHOD/train-imagenet100-mode-sampling-t-$T_SG/imagenet100-$i-IPC-$IPC

python sample.py --model DiT-XL/2 --image-size 256 --sampling-method $METHOD \
    --save-dir $OUTPUT_DATASET --spec $SPEC --num-samples $IPC \
    --seed $i --ckpt './pretrained_models/DiT-XL-2-256x256.pt' --nclass 100 --phase 0 --ckpt './pretrained_models/DiT-XL-2-256x256.pt'

python train.py -d imagenet --imagenet_dir $OUTPUT_DATASET $IMAGENET_FOLDER \
    -n convnet --depth 6 --nclass 100 --norm_type instance --ipc $IPC --tag test --slct_type random --spec $SPEC --repeat 1 \
    --save-dir $TRAIN_SAVE_DIR-convnet

python train.py -d imagenet --imagenet_dir $OUTPUT_DATASET $IMAGENET_FOLDER \
    -n resnet_ap --nclass 100 --norm_type instance --ipc $IPC --tag test --slct_type random --spec $SPEC --repeat 1 \
    --save-dir $TRAIN_SAVE_DIR-resnet_ap

python train.py -d imagenet --imagenet_dir $OUTPUT_DATASET $IMAGENET_FOLDER \
    -n resnet --depth 18 --nclass 100 --norm_type instance --ipc $IPC --tag test --slct_type random --spec $SPEC --repeat 1 \
    --save-dir $TRAIN_SAVE_DIR-resnet_18

done

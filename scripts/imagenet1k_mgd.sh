export CUDA_VISIBLE_DEVICES=6
IMAGENET_FOLDER=/data/ImageNet1k

IPC=50
T=50 # STEPS
T_SG=25 # STOP GUIDANCE
SPEC=imagenet1k
METHOD=ddpm

for ((i=0; i < 3; i++))
do

OUTPUT_DATASET=results-1k/MGD/$METHOD/dit-distillation-sampling-t-$T_SG/imagenet1k-$i-IPC-$IPC
TRAIN_SAVE_DIR=results-1k/MGD/$METHOD/train-imagenet1k-mode-sampling-t-$T_SG/imagenet1k-$i-IPC-$IPC

python sample_mode_guidance.py --model DiT-XL/2 --image-size 256 --sampling-method $METHOD \
 --save-dir $OUTPUT_DATASET --spec $SPEC --num-samples $IPC --guidance --nclass 1000 \
 --stop_t $STOP_T $T_SG --imagenet_dir $IMAGENET_FOLDER --seed $i --num-datasets 1 --ckpt './pretrained_models/DiT-XL-2-256x256.pt'

python train.py -d imagenet --imagenet_dir $OUTPUT_DATASET/dataset_0 $IMAGENET_FOLDER \
    -n convnet --depth 6 --nclass 1000 --norm_type instance --ipc $IPC --tag test --slct_type random --spec $SPEC --repeat 1 \
    --save-dir $TRAIN_SAVE_DIR-convnet

python train.py -d imagenet --imagenet_dir $OUTPUT_DATASET/dataset_0 $IMAGENET_FOLDER \
    -n resnet_ap --nclass 1000 --norm_type instance --ipc $IPC --tag test --slct_type random --spec $SPEC --repeat 1 \
    --save-dir $TRAIN_SAVE_DIR-resnet_ap

python train.py -d imagenet --imagenet_dir $OUTPUT_DATASET/dataset_0 $IMAGENET_FOLDER \
    -n resnet --depth 18 --nclass 1000 --norm_type instance --ipc $IPC --tag test --slct_type random --spec $SPEC --repeat 1 \
    --save-dir $TRAIN_SAVE_DIR-resnet_18

done

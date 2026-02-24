export CUDA_VISIBLE_DEVICES=2
IMAGENET_FOLDER=/data/aroy25/ImageNette

IPC=10
T=50 # STEPS
T_SG=25 # STOP GUIDANCE
SPEC=nette
METHOD=ddpm

for ((i=0; i < 3; i++))
do

OUTPUT_DATASET=results-trial/Base/$METHOD/dit-distillation-sampling-t-$T_SG/nette-$i-IPC-$IPC
TRAIN_SAVE_DIR=results-trial/Base/$METHOD/train-nette-mode-sampling-t-$T_SG/nette-$i-IPC-$IPC

python sample.py --model DiT-XL/2 --image-size 256 --sampling-method $METHOD \
    --save-dir $OUTPUT_DATASET --spec $SPEC --num-samples $IPC --num-sampling-steps $T \
    --seed $i --ckpt '/home/csgrad/aroy25/projects/mode_guidance/pretrained_models/DiT-XL-2-256x256.pt' --nclass 10 --phase 0

python train.py -d imagenet --imagenet_dir $OUTPUT_DATASET $IMAGENET_FOLDER \
    -n convnet --depth 6 --nclass 10 --norm_type instance --ipc $IPC --tag test --slct_type random --spec $SPEC --repeat 1 \
    --save-dir $TRAIN_SAVE_DIR-convnet

python train.py -d imagenet --imagenet_dir $OUTPUT_DATASET $IMAGENET_FOLDER \
    -n resnet_ap --nclass 10 --norm_type instance --ipc $IPC --tag test --slct_type random --spec $SPEC --repeat 1 \
    --save-dir $TRAIN_SAVE_DIR-resnet_ap

python train.py -d imagenet --imagenet_dir $OUTPUT_DATASET $IMAGENET_FOLDER \
    -n resnet --depth 18 --nclass 10 --norm_type instance --ipc $IPC --tag test --slct_type random --spec $SPEC --repeat 1 \
    --save-dir $TRAIN_SAVE_DIR-resnet_18

done

export CUDA_VISIBLE_DEVICES=3
IMAGENET_FOLDER=/data/aroy25/imagewoof

IPC=50
T=50 # STEPS
T_SG=25 # STOP GUIDANCE
SPEC=woof
METHOD=ddpm

for ((i=0; i < 3; i++))
do

OUTPUT_DATASET=results_woof/MGD/$METHOD/dit-distillation-sampling-t-$T_SG/woof-$i-IPC-$IPC
TRAIN_SAVE_DIR=results_woof/MGD/$METHOD/train-woof-mode-sampling-t-$T_SG/woof-$i-IPC-$IPC

python sample_mode_guidance.py --model DiT-XL/2 --image-size 256 --sampling-method $METHOD \
 --save-dir $OUTPUT_DATASET --spec $SPEC --num-samples $IPC --guidance --nclass 10 \
 --stop_t $T_SG --imagenet_dir $IMAGENET_FOLDER --seed $i --num-datasets 1 \
 --ckpt '/home/csgrad/aroy25/projects/mode_guidance/pretrained_models/DiT-XL-2-256x256.pt' \

python train.py -d imagenet --imagenet_dir $OUTPUT_DATASET/dataset_0 $IMAGENET_FOLDER \
    -n convnet --depth 6 --nclass 10 --norm_type instance --ipc $IPC --tag test --slct_type random --spec $SPEC --repeat 1 \
    --save-dir $TRAIN_SAVE_DIR-convnet

python train.py -d imagenet --imagenet_dir $OUTPUT_DATASET/dataset_0 $IMAGENET_FOLDER \
    -n resnet_ap --nclass 10 --norm_type instance --ipc $IPC --tag test --slct_type random --spec $SPEC --repeat 1 \
    --save-dir $TRAIN_SAVE_DIR-resnet_ap

python train.py -d imagenet --imagenet_dir $OUTPUT_DATASET/dataset_0 $IMAGENET_FOLDER \
    -n resnet --depth 18 --nclass 10 --norm_type instance --ipc $IPC --tag test --slct_type random --spec $SPEC --repeat 1 \
    --save-dir $TRAIN_SAVE_DIR-resnet_18

done

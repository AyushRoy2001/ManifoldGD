export CUDA_VISIBLE_DEVICES=7
IMAGENET_FOLDER=/data/ImageNet1k

IPC=50
T=50 # STEPS
T_SG=25 # STOP GUIDANCE
SPEC=imagenet1k
METHOD=ddpm
CLUSTER=divisive_layer # kmeans, agglomerative, divisive, divisive_layer
LINK=single # "ward","average","complete","single"
LEVEL=1 # for divisive_layer only; 2 means second last level

MANIFOLD_RADII=0.1 DENSITY=0.1

# Guidance geometry controls
WARM_STEPS=5         # --radius-warm-steps (only warm-up for first N steps)
RMAX=2.0             # --radius-max-mult
RSCHED=exp           # --radius-schedule: linear | cosine | exp

for ((i=0; i < 3; i++))
do

OUTPUT_DATASET=results-1k/Ours/$METHOD/dit-distillation-sampling-t-$T_SG/imagenet1k-$i-IPC-$IPC
TRAIN_SAVE_DIR=results-1k/Ours/$METHOD/train-imagenet1k-mode-sampling-t-$T_SG/imagenet1k-$i-IPC-$IPC

python sample_new.py --model DiT-XL/2 --image-size 256 --sampling-method $METHOD \
 --save-dir $OUTPUT_DATASET --spec $SPEC --num-samples $IPC --guidance --nclass 1000 \
 --stop_t $T_SG --imagenet_dir $IMAGENET_FOLDER --seed $i --num-datasets 1 --ckpt './pretrained_models/DiT-XL-2-256x256.pt' \
 --cluster-method $CLUSTER --agglom-linkage $LINK --divisive-criterion sse --divisive-level-i $LEVEL \
 --manifold-radius $MANIFOLD_RADII --density-bandwidth $DENSITY --manifold-guidance-scale 0.1 \
 --tangent-dim 3 --radius-warm-steps $WARM_STEPS --radius-max-mult $RMAX --radius-schedule $RSCHED \
 --no-visualize

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

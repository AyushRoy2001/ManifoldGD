export CUDA_VISIBLE_DEVICES=1
IMAGENET_FOLDER=/path/to/ImageNette

IPC=10
T_SG=25 # STOP GUIDANCE
SPEC=nette
METHOD=ddpm

CLUSTER=divisive # kmeans, agglomerative, divisive, divisive_layer
LINK=single # "ward","average","complete","single"
LEVEL=6 # for divisive_layer only; 2 means second last level

MANIFOLD_RADII=0.05 DENSITY=0.2

# Guidance geometry controls
WARM_STEPS=5         # --radius-warm-steps (only warm-up for first N steps)
RMAX=2.0             # --radius-max-mult
RSCHED=exp           # --radius-schedule: linear | cosine | exp
 
for ((i=0; i < 1; i++))
do

OUTPUT_DATASET=path/to/output/directory

python sample_new.py --model DiT-XL/2 --image-size 256 --sampling-method $METHOD \
    --save-dir $OUTPUT_DATASET --spec $SPEC --num-samples $IPC --guidance --nclass 10 \
    --stop_t $T_SG --imagenet_dir $IMAGENET_FOLDER --seed $i --num-datasets 1 \
    --ckpt './pretrained_models/DiT-XL-2-256x256.pt' \
    --cluster-method $CLUSTER --agglom-linkage $LINK --divisive-criterion sse --divisive-level-i $LEVEL \
    --manifold-radius $MANIFOLD_RADII --density-bandwidth $DENSITY --manifold-guidance-scale 0.0 --mode_guidance_scale 0.0 \
    --tangent-dim 3 --radius-warm-steps $WARM_STEPS --radius-max-mult $RMAX --radius-schedule $RSCHED \
    --visualize

done
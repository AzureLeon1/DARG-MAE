dataset=$1
input_model_file=$2
seed=$3
device=$4

python adv_finetune.py \
    --dataset $dataset \
    --input_model_file $input_model_file \
    --seed $seed \
    --device $device \
    --lr 0.001 \
    --uniformity_dim 32 \
    --sub_weight 0.4 \
    --alpha_T 0.5 \
    --gamma 0.8 \
    --replace_rate 0 \
    --lamda 0.01 \
    --belta 0.001 \
    --epochs 50
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
    --uniformity_dim 8 \
    --sub_weight 0.8 \
    --alpha_T 0.7 \
    --gamma 1 \
    --replace_rate 0 \
    --lamda 0.01 \
    --belta 0.001 \
    --epochs 50
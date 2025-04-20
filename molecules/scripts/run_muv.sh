dataset=$1
input_model_file=$2
seed=$3
device=$4

python adv_finetune.py \
    --dataset $dataset \
    --input_model_file $input_model_file \
    --seed $seed \
    --device $device \
    --lr 0.0001 \
    --uniformity_dim 8 \
    --sub_weight 0.4 \
    --alpha_T 0.4 \
    --gamma 0.8 \
    --replace_rate 0 \
    --lamda 0.1 \
    --belta 0.01 \
    --epochs 50
uniformity_dim=$1
sub_weight=$2
device=$3

python chem_tran.py \
    --device $device \
    --sub_weight $sub_weight \
    --uniformity_dim $uniformity_dim \
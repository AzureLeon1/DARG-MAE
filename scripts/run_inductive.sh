dataset=$1
device=$2

[ -z "${dataset}" ] && dataset="ppi"
[ -z "${device}" ] && device=-1

python main_inductive.py \
	--device $device \
	--dataset $dataset \
	--use_cfg \

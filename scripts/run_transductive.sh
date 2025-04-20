dataset=$1
device=$2

[ -z "${dataset}" ] && dataset="wikics"
[ -z "${device}" ] && device=-1

python main_transductive.py \
	--device $device \
	--dataset $dataset \
	--use_cfg \

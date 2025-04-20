dataset=$1
device=$2

[ -z "${dataset}" ] && dataset="COLLAB"
[ -z "${device}" ] && device=-1

python main_graph.py \
	--device $device \
	--dataset $dataset \
	--use_cfg \

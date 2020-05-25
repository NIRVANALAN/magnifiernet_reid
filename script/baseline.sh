# Experiment all tricks without center loss : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# without center loss
partition=${1}
num_gpus=1
config=${2}
device=${3}

if [[ $# -eq 4 ]]; then
  job_name=${4}
else
  job_name=train-reid
fi

GLOG_vmodule=MemcachedClient=-1 \
srun --mpi=pmi2 -p ${partition} \
  -n${num_gpus} --gres=gpu:1 --ntasks-per-node=1 \
  --job-name=${job_name} \
python3 tools/train.py --config_file=${config} MODEL.DEVICE_ID "('${device}')" DATASETS.NAMES "('market1501')" DATASETS.ROOT_DIR "('/mnt/lustre/liuyuan1/cvpr20/data')"

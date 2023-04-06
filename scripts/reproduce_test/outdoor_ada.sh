#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate adamatcher
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

data_cfg_path="configs/data/megadepth_test_1500.py"
main_cfg_path="configs/loftr/outdoor/loftr_ds_dense.py"
ckpt_path="weights/adamatcher.ckpt"

dump_dir="dump/loftr_ds_outdoor"
profiler_name="inference"
n_nodes=1  # manually keep this the same with --nodes
n_gpus_per_node=-1 # -1
torch_num_workers=4 # 4
batch_size=1  # per gpu

CUDA_VISIBLE_DEVICES=1 python3 -u ./test.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --dump_dir=${dump_dir} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers}\
    --profiler_name=${profiler_name} \
    --benchmark

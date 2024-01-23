#!/bin/bash
#sh run_test_bandwidth.sh 2 2 0 10.20.1.81
#sh run_test_bandwidth.sh 2 2 1 10.20.1.81
set -e

GPUS_PER_NODE=$1
NNODES=$2
NODE_RANK=$3
MASTER_ADDR=$4
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR"

run_cmd="torchrun $DISTRIBUTED_ARGS test_bandwidth.py"

echo ${run_cmd}
eval ${run_cmd}
set +x

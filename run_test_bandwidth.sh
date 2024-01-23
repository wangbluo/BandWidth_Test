#!/bin/bash
#sh run_test_bandwidth.sh 2 2 0 10.20.1.81 22
#sh run_test_bandwidth.sh 2 2 1 10.20.1.81 22 
set -e

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR # --master_port $MASTER_PORT"

run_cmd="torchrun $DISTRIBUTED_ARGS test_bandwidth.py"

echo ${run_cmd}
eval ${run_cmd}
set +x

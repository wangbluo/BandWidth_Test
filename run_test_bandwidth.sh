set -e


DISTRIBUTED_ARGS="--nproc_per_node 2 --nnodes 2 --node_rank 0 --master_addr 10.20.1.81" # --master_port $MASTER_PORT"

run_cmd="torchrun $DISTRIBUTED_ARGS test_bandwidth.py"

echo ${run_cmd}
eval ${run_cmd}
set +x

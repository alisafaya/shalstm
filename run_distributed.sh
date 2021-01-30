
echo "Running with node rank $1"

nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9

export OMP_NUM_THREADS=1 # ngpus / nthreads in the node

launch_dist=$(python -c "from os import path; import torch; print(path.join(path.dirname(torch.__file__), 'distributed', 'launch.py'))")
NNODES=6
MASTER_ADDR='172.20.242.217'
MASTER_PORT=25232
NGPU_PER_NODE=4
RANK=$1

export NCCL_BLOCKING_WAIT=1

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

# MASTER_ADDR='localhost'
CUDA_VISIBLE_DEVICES=0,1,2,3
# CUDA_VISIBLE_DEVICES=4,5,6,7

[ -e /kuacc/users/asafaya19/shalstm/dist_shared ] && rm /kuacc/users/asafaya19/shalstm/dist_shared

python $launch_dist --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nnodes=$NNODES --node_rank=$RANK --nproc_per_node=$NGPU_PER_NODE -m train.dist --local_world_size=$NGPU_PER_NODE
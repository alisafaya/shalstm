echo "Running with node rank $1 with $2 gpus"

## clean up
nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9
[ -e /kuacc/users/asafaya19/shalstm/dist_shared_base ] && rm /kuacc/users/asafaya19/shalstm/dist_shared_base

## settings
WORLD_SIZE=8
NNODES=2
MASTER_ADDR='172.20.242.221'
MASTER_PORT=25234
NGPU_PER_NODE=$2
RANK=$1
NODERANK=$(($2 * $1))

export OMP_NUM_THREADS=4 # ngpus / nthreads
export NCCL_BLOCKING_WAIT=1

## run
python -m torch.distributed.launch --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nnodes=$NNODES --node_rank=$RANK --nproc_per_node=$NGPU_PER_NODE -m train.dist --rank $NODERANK --world_size $WORLD_SIZE --local_world_size $NGPU_PER_NODE --writer_dir runs/base --checkpoint_path bin/pile/base/base --bptt 1024 --dist_file /kuacc/users/asafaya19/shalstm/dist_shared_base --model_config config/base.json
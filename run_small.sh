echo "Running with node rank $1 with $2 gpus"

## clean up
nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9
[ -e /kuacc/users/asafaya19/shalstm/dist_shared_small ] && rm /kuacc/users/asafaya19/shalstm/dist_shared_small

## settings
WORLD_SIZE=20
NNODES=5
MASTER_ADDR='172.20.242.217'
MASTER_PORT=25234
NGPU_PER_NODE=$2
RANK=$1
NODERANK=$(($2 * $1))

export OMP_NUM_THREADS=4 # ngpus / nthreads
export NCCL_BLOCKING_WAIT=1

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

# MASTER_ADDR='localhost'
CUDA_VISIBLE_DEVICES=0,1,2,3
# CUDA_VISIBLE_DEVICES=4,5,6,7

## run
python -m torch.distributed.launch --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nnodes=$NNODES --node_rank=$RANK --nproc_per_node=$NGPU_PER_NODE -m train.dist --rank $NODERANK --world_size $WORLD_SIZE --local_world_size $NGPU_PER_NODE --writer_dir runs/small-4 --load_checkpoint bin/pile/small/small_183273.pt --checkpoint_path bin/pile/small/small --bptt 768 --evaluate_each 4 --dist_file /kuacc/users/asafaya19/shalstm/dist_shared_small --model_config config/small.json
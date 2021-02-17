echo "Running with node rank $1 with $2 gpus"

DIST_FILE=$(pwd)/bin/dist_shared_small

## clean up
nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9
[ -e $DIST_FILE ] && rm $DIST_FILE

## settings
WORLD_SIZE=12
NNODES=3
MASTER_ADDR='172.20.242.221'
MASTER_PORT=22222
NGPU_PER_NODE=$2
RANK=$1
NODERANK=$(($2 * $1))

export OMP_NUM_THREADS=4 # ngpus / nthreads
export NCCL_BLOCKING_WAIT=1

## run
python -m torch.distributed.launch --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nnodes=$NNODES --node_rank=$RANK --nproc_per_node=$NGPU_PER_NODE -m lm.dist --rank $NODERANK --world_size $WORLD_SIZE --local_world_size $NGPU_PER_NODE --writer_dir runs/small --checkpoint_path bin/pile/small/small --bptt 1536 --dist_file $DIST_FILE --model_config config/small.json
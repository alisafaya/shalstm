exp_dir=/kuacc/users/asafaya19/shalstm-exp-data/shalstm-data
data_dir=$exp_dir/full-data-bin
ckpt_dir=$exp_dir/4_v100_5_epoch_sum_loss/shalstm
run_dir=$exp_dir/runs/4_v100_5_epoch_sum_loss/

total_steps=1000000
warmup=5000
epochs=5
eval_steps=160000
model_config=config/small_exp.json

## multi gpu settings
DIST_FILE=$(pwd)/bin/dist_shared_base
WORLD_SIZE=4
NNODES=1
NGPU_PER_NODE=4
RANK=0

## clean up processes
nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9
[ -e $DIST_FILE ] && rm $DIST_FILE

export OMP_NUM_THREADS=3 # nthreads / ngpus
export NCCL_BLOCKING_WAIT=1

## run
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$RANK \
    --nproc_per_node=$NGPU_PER_NODE \
    -m lm.dist \
    --dist_file $DIST_FILE \
    --rank $RANK \
    --world_size $WORLD_SIZE \
    --local_world_size $NGPU_PER_NODE \
    --train_dir "$data_dir/train.*.pt" \
    --val_dir $data_dir/valid.pt \
    --test_dir $data_dir/test.pt  \
    --checkpoint_path $ckpt_dir  \
    --writer_dir $run_dir  \
    --model_config $model_config \
    --base_lr 0.001 \
    --clip_value 0.25 \
    --warmup $warmup  \
    --max_steps $total_steps \
    --evaluate_each $eval_steps \
    --epochs $epochs \
    --bptt 1024

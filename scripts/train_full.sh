exp_dir=/kuacc/users/asafaya19/shalstm-exp-data/shalstm-data
data_dir=$exp_dir/full-data-bin
ckpt_dir=$exp_dir/checkpoints_full/shalstm
run_dir=$exp_dir/runs/full/

total_steps=1000000
warmup=5000
epochs=1
eval_steps=100000
model_config=config/small_exp.json

python -m lm.train_exp --train_dir "$data_dir/train.*.pt" --val_dir $data_dir/valid.pt --test_dir $data_dir/test.pt   --checkpoint_path $ckpt_dir  --writer_dir $run_dir  --model_config $model_config  --base_lr 0.001 --clip_value 0.25   --warmup $warmup  --max_steps $total_steps  --eval_steps $eval_steps --epochs $epochs

# SMALL-EXP: No of parameters 118733827
# BASE-EXP: No of parameters 203877379

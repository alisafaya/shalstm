exp_dir=/nfs/cluster
data_dir=$exp_dir/inputs
ckpt_dir=$exp_dir/checkpoint_demo/shalstm
run_dir=$exp_dir/runs/singlegpu_run/

total_steps=2000
warmup=1000
epochs=1
eval_steps=1000
model_config=config/small_exp.json

python3 -m lm.train_exp --bptt 1024 --train_dir "$data_dir/train.*.pt" --val_dir $data_dir/valid.pt --test_dir $data_dir/test.pt   --checkpoint_path $ckpt_dir  --writer_dir $run_dir  --model_config $model_config  --base_lr 0.001 --clip_value 0.25   --warmup $warmup  --max_steps $total_steps  --eval_steps $eval_steps --epochs $epochs



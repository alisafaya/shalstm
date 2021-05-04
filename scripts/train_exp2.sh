exp_dir=/kuacc/users/asafaya19/shalstm-exp-data/shalstm-data
data_dir=$exp_dir/data-bin
ckpt_dir=$exp_dir/shalstm-small-2-onegpuday/shalstm
run_dir=$exp_dir/runs/v1_exp2/

total_steps=35000
warmup=5000
eval_steps=`expr $total_steps / 5`

python -m lm.train_exp --no_amp --epochs 5 --train_dir $data_dir/train.pt --val_dir $data_dir/valid.pt --test_dir $data_dir/test.pt   --checkpoint_path $ckpt_dir  --writer_dir $run_dir  --model_config config/small_exp2.json  --base_lr 0.001 --clip_value 0.25   --warmup $warmup --max_steps $total_steps  --eval_steps $eval_steps


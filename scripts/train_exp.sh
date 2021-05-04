exp_dir=/kuacc/users/asafaya19/shalstm-exp-data/shalstm-data
data_dir=$exp_dir/data-bin
ckpt_dir=$exp_dir/checkpoints_v2/shalstm
run_dir=$exp_dir/runs/v2/

total_steps=52000
warmup=2000
eval_steps=`expr $total_steps / 5`

python -m lm.train_exp --no_amp --train_dir $data_dir/train.pt --val_dir $data_dir/valid.pt --test_dir $data_dir/test.pt   --checkpoint_path $ckpt_dir  --writer_dir $run_dir  --model_config config/small_exp.json  --base_lr 0.001 --clip_value 0.25   --warmup $warmup  --max_steps $total_steps  --eval_steps $eval_steps

# Namespace(base_lr=0.00015, bptt=1024, checkpoint_path='/kuacc/users/asafaya19/shalstm-exp-data/shalstm-data/checkpoints/shalstm', clip_value=0.5, device='cuda', epochs=5, eval_steps=10000, global_step=0, load_checkpoint='', log_interval=100, max_steps=50000, model_config='config/small_exp.json', no_amp=False, seed=123, test_dir='/kuacc/users/asafaya19/shalstm-exp-data/shalstm-data/data-bin/test.pt', train_dir='/kuacc/users/asafaya19/shalstm-exp-data/shalstm-data/data-bin/train.pt', val_dir='/kuacc/users/asafaya19/shalstm-exp-data/shalstm-data/data-bin/valid.pt', warmup=5000, writer_dir='/kuacc/users/asafaya19/shalstm-exp-data/shalstm-data/runs/')

# No of parameters 118733827

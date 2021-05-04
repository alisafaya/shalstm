exp_dir=/kuacc/users/asafaya19/tweet-shalstm
data_dir=$exp_dir/bin
ckpt_dir=$exp_dir/checkpoints/tweet_shalstm
run_dir=$exp_dir/runs/full_4M/

total_steps=4000000
warmup=5000
epochs=1
eval_steps=1000000
model_config=config/tweets_config.json

python -m lm.train_exp --train_dir "$data_dir/chunk_*.pt" --val_dir $data_dir/valid.pt --test_dir $data_dir/valid.pt   --checkpoint_path $ckpt_dir  --writer_dir $run_dir  --model_config $model_config  --base_lr 0.001 --clip_value 0.25   --warmup $warmup  --max_steps $total_steps  --eval_steps $eval_steps --epochs $epochs

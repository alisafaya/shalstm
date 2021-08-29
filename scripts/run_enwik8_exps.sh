#! /bin/sh

python -u -m lm.train --train_dir bin/tokenized/enwik8.train.pt --val_dir bin/tokenized/enwik8.valid.pt --test_dir bin/tokenized/enwik8.test.pt --epochs 20 --bptt 1536 --batch_size 32 --checkpoint_path bin/lm-enwik8/sp-16k --model_config config/small.json --writer_dir enwik8-runs/sp-16k &>> enwik8_exp.log
python -u -m lm.eval --model bin/lm-enwik8/sp-16k --data_dir bin/tokenized/enwik8.valid.pt --data_length 4971572 &>> enwik8_exp.log
python -u -m lm.eval --model bin/lm-enwik8/sp-16k --data_dir bin/tokenized/enwik8.test.pt --data_length 4972442 &>> enwik8_exp.log

# Define a list of string variable
stringList=4k,8k,16k,32k

# Use comma as separator and apply as pattern
for val in ${stringList//,/ }
do
    python -u -m lm.train --train_dir bin/bbpe/$val/enwiki.train.pt --val_dir bin/bbpe/$val/enwiki.valid.pt --test_dir bin/bbpe/$val/enwiki.test.pt --epochs 20 --bptt 1536 --batch_size 32 --checkpoint_path bin/lm-enwik8/$val --model_config config/small.json --writer_dir enwik8-runs/$val &>> enwik8_exp.log
    python -u -m lm.eval --model bin/lm-enwik8/$val --data_dir bin/bbpe/$val/enwiki.valid.pt --data_length 4971572 &>> enwik8_exp.log
    python -u -m lm.eval --model bin/lm-enwik8/$val --data_dir bin/bbpe/$val/enwiki.test.pt --data_length 4972442 &>> enwik8_exp.log
done
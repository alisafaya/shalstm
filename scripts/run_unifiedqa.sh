#! /bin/sh

python -m qa.train --model bin/pile/small/model --tokenizer tokenizer/tokenizer.json --checkpoint_dir bin/qa/unifiedqa_small --train_dir data/unifiedqa/unifiedqa_pretraining_data/train.tsv --val_dir data/unifiedqa/unifiedqa_pretraining_data/dev.tsv --batch_size 48 --epochs 8 --warmup 1 --writer_dir qa-runs/unifiedqa_small --base_lr 1e-3

# Define a list of string variable
stringList=1,2,3,4,5,6,7,8

# Use comma as separator and apply as pattern
for val in ${stringList//,/ }
do
    echo 'Evaluating checkpoint '$val
    python -u -m qa.eval --model bin/qa/unifiedqa_small_$val --tokenizer tokenizer/tokenizer.json --val_dir data/unifiedqa/boolq/dev.tsv --batch_size 48 --metric accuracy &>> unifiedqa_exp_small.log
    python -u -m qa.eval --model bin/qa/unifiedqa_small_$val --tokenizer tokenizer/tokenizer.json --val_dir data/unifiedqa/narrativeqa/test.tsv --batch_size 48 --metric rouge &>> unifiedqa_exp_small.log
    python -u -m qa.eval --model bin/qa/unifiedqa_small_$val --tokenizer tokenizer/tokenizer.json --val_dir data/unifiedqa/squad1_1/dev.tsv --batch_size 48 --val_ans data/unifiedqa/squad1_1/dev_ans.jsonl --metric squad &>> unifiedqa_exp_small.log
    python -u -m qa.eval --model bin/qa/unifiedqa_small_$val --tokenizer tokenizer/tokenizer.json --val_dir data/unifiedqa/squad2/dev.tsv --batch_size 48 --val_ans data/unifiedqa/squad2/dev_ans.jsonl --metric squad_v2 &>> unifiedqa_exp_small.log
done
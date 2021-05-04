#! /bin/sh
 
# Define a list of string variable
stringList=small,base,large

# Use comma as separator and apply as pattern
for val in ${stringList//,/ }
do
    python -m lm.train --train_dir bin/tokenized/wiki-103.train.pt --val_dir bin/tokenized/wiki-103.valid.pt --test_dir bin/tokenized/wiki-103.test.pt --epochs 20 --checkpoint_path bin/lm-wiki/$val --model_config config/$val.json --writer_dir wiki-runs/$val
done
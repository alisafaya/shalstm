
export TPU_IP_ADDRESS="10.48.37.178"
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

python -m train.train --train_dir bin/tokenized/enwik8.train.pt --val_dir bin/tokenized/enwik8.valid.pt --test_dir bin/tokenized/enwik8.test.pt --writer_dir runs/enwik8-base-tpu --no_amp --clip_value 0.25
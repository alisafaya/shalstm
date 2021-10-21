# Single Headed Attention LSTM

For full details see the paper [Single Headed Attention RNN: Stop Thinking With Your Head](https://arxiv.org/abs/1911.11423).

Original implementation: [sha-rnn](https://github.com/Smerity/sha-rnn)

For installation: 

```shell
git clone https://github.com/alisafaya/shalstm.git
cd shalstm
pip install -e .
```

### SHALSTM for Question Answering

### SHALSTM for Sequence Classification

### SHALSTM for Language Generation

```shell
python -m shalstm.lm.generate --model pretrained_model_dir/ --prompt 'some prompt' --device cuda --use-sampling --max-length 256
```

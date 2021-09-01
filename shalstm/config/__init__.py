# Check which configuration is used for base and small models trained on oracle HPC.
BASE = {
    "embed_size": 1024,
    "memory_size": 4096,
    "hidden_size": 4096,
    "no_layers": 12,
    "dropouti": 0.1,
    "dropouto": 0.1,
    "dropouth": 0.1,
    "vocab_size": 32768,
    "attn_layers": [2, 4, 6, 8, 10, 12],
    "cutoffs": [1024, 4096, 16384],
    "rnn_type": "lstm"
}

SMALL = {
    "embed_size": 768,
    "memory_size": 3072,
    "hidden_size": 3072,
    "no_layers": 6,
    "dropouti": 0.1,
    "dropouto": 0.1,
    "dropouth": 0.1,
    "vocab_size": 32768,
    "attn_layers": [2, 4, 6],
    "cutoffs": [1024, 4096, 16384],
    "rnn_type": "lstm"
}

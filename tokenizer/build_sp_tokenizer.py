import sentencepiece as spm
import sys

for m, v in (("4k", 2**12), ("8k", 2**13), ("16k", 2**14), ("32k", 2**15)):
    spm.SentencePieceTrainer.train(
        input=sys.argv[1:],
        model_prefix=f'pile_{m}', 
        model_type="unigram",
        vocab_size=v,
        pad_id=0,
        unk_id=1,
        eos_id=2,
        bos_id=-1,
        control_symbols=["<sha_tkn1>", "<sha_tkn2>", "<sha_tkn3>", "<sha_tkn4>", "<sha_tkn5>"],
        shuffle_input_sentence=True,
        input_sentence_size=10**7)

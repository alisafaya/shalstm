import sys
from tokenizers import ByteLevelBPETokenizer

for m, v in (("4k", 2**12), ("8k", 2**13), ("16k", 2**14), ("32k", 2**15)):
    print("Training", m)
    
    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Then train it!
    tokenizer.train(sys.argv[1:], vocab_size=v, min_frequency=5, special_tokens=["<pad>", "<unk>", "</s>"])

    # And finally save it somewhere
    tokenizer.save(f"bbpe.{m}.tokenizer.json")
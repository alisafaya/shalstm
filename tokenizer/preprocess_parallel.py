import time
import sys
import os
import json
import torch
import itertools
import multiprocessing as mp
import sentencepiece as spm

from tqdm import tqdm
from glob import glob

upper_limits = {
    "Wikipedia (en)": 25e9,
    "Gutenberg (PG-19)": 30e9,
    "Books3": 100e9,
    "OpenSubtitles": 20e9,
    "OpenWebText2": 50e9,
    "Pile-CC": 50e9,
    "Enron Emails": 3e9,
    "EuroParl": 10e9,
    "StackExchange": 15e9,
    "HackerNews": 8e9,
    "YoutubeSubtitles": 8e9
}

def process_lines(lines):
    
    pid = os.getpid()
    tokenizers[pid] = tokenizers.get(pid, spm.SentencePieceProcessor(model_file=tokenizer, add_eos=True))
    new_lines = []

    for l in lines:
        if l is not None:
            obj = json.loads(l)
            set_name = obj["meta"]["pile_set_name"]
            
            if set_name in upper_limits and counter[set_name] < upper_limits[set_name]:
                new_lines.append(obj["text"].rstrip("\n"))
                counter[set_name] += len(obj["text"])
    
    if len(new_lines) > 0:
        encoded = tokenizers[pid].encode(new_lines, add_eos=True)
        return list(itertools.chain(*encoded))
    else:
        return []

if __name__ == "__main__":
    indir = sys.argv[1]
    outdir = sys.argv[2]
    nworkers = int(sys.argv[3] if len(sys.argv) > 2 else 32)
    
    tensor_size = 100_000_000
    tokenizer = "spmodels/pile_16k.model"
    batchsize = 32 * 1024
    
    current_batch = torch.ShortTensor()
    current_batch_id = 0

    with mp.Manager() as manager:

        counter = manager.dict({ x : 0 for x in upper_limits })
        tokenizers = manager.dict()
        upper_limits = manager.dict(upper_limits)

        with mp.Pool(processes=nworkers) as pool:
            for infi in glob(indir):
                with open(infi) as fi:
                    processors_iter = pool.imap(process_lines, tqdm(itertools.zip_longest(*[fi]*batchsize)))

                    for tokens in processors_iter:
                        current_batch = torch.cat((current_batch, torch.ShortTensor(tokens)))
                        
                        if current_batch.size(0) > tensor_size:
                            current_batch_id += 1
                            outfile = os.path.join(outdir, f"batch_{current_batch_id:0>5}.pt")
                            torch.save(current_batch, outfile)
                            current_batch = torch.ShortTensor()
                print("Finished", infi)
                print(counter)

        # if any left
        if current_batch.size(0) > 0:
            current_batch_id += 1
            outfile = os.path.join(outdir, f"batch_{current_batch_id:0>5}.pt")
            torch.save(current_batch, outfile)

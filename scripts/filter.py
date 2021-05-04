import json
import sys
import itertools
import hashlib

from glob import glob

infile = sys.argv[1]
outfile = sys.argv[2]
batchsize = 2**15

filter_set = set(["Wikipedia (en)", "Books3", "OpenWebText2"])
freq = { f : 0 for f in filter_set }
file_set = { f : open(outfile + "-" + f + ".jsonl", "w") for f in filter_set }
hash_set = set()

for f in glob(infile):
    print("Processing ", f)
    with open(f) as fi:
        for lines in itertools.zip_longest(*[fi]*batchsize):
            for l in lines:
                if l is not None:
                    obj = json.loads(l)
                    v = obj["meta"]["pile_set_name"]

                    if v in filter_set:
                        line_hash = hashlib.sha256(obj["text"].encode())
                        if line_hash not in hash_set:

                            hash_set.add(line_hash)
                            freq[v] += 1
                            file_set[v].write(json.dumps(obj["text"]) + "\n")


with open(outfile + "_freq.json", "w") as fo:
    fo.write(json.dumps(freq, indent=2))

import json
import sys
import itertools
from tqdm import tqdm

infile = sys.argv[1]
outfile = sys.argv[2]
batchsize = 1024
freq = {}

with open(infile) as fi:
        for lines in tqdm(itertools.zip_longest(*[fi]*batchsize)):
            for l in lines:
                if l is not None:
                    v = json.loads(l)["meta"]["pile_set_name"]
                    freq[v] = 1 + freq.get(v, 0)

                    
total = sum(freq.values())

for k in list(freq.keys()):
    freq[k] = freq[k] * 100 / total

with open(outfile, "w") as fo:
    fo.write(json.dumps(freq, indent=2))

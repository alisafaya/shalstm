import json
import sys
import itertools
from tqdm import tqdm

infile = sys.argv[1]
outfile = sys.argv[2]
batchsize = 1024

with open(infile) as fi:
    with open(outfile, "w") as fo:
        for lines in tqdm(itertools.zip_longest(*[fi]*batchsize)):
            try:
                lines = [ json.loads(l)["text"] for l in lines ]
                print("</s>\n".join(lines), file=fo)
            except:
                lines = [ json.loads(l)["text"] for l in lines if l is not None ]
                print("</s>\n".join(lines), file=fo)
import os
from utils import load_from_pkl, dump_to_pkl

path = 'data/pair'
files= os.listdir(path)[1:]
pairs = dict()
for file in files:
    if not os.path.isdir(file):
        pair_file_path = path+"/"+file
        pair = load_from_pkl(pair_file_path)
        if len(pairs) == 0:
            pairs = pair
        else:
            for key in pair.keys():
                if key in pairs:
                    pairs[key] += pair[key]
                else:
                    pairs[key] = pair[key]

output_file_name = 'data/pair/pairs_merge.pkl'
dump_to_pkl(pairs, output_file_name)
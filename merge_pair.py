import os
from utils import load_from_pkl, dump_to_pkl
from tqdm import tqdm
import logging

path = 'data/pair'
files= os.listdir(path)
pairs = dict()
for file in tqdm(files):
    if not os.path.isdir(file):
        pair_file_path = path+"/"+file
        pair = load_from_pkl(pair_file_path)
        logging.info("pair size: %d" % (len(pair)))
        if len(pairs) == 0:
            pairs = pair
        else:
            for key in pair.keys():
                if key in pairs:
                    pairs[key] += pair[key]
                else:
                    pairs[key] = pair[key]
        logging.info("current total pair size: %d" % (len(pairs)))

output_file_name = 'data/pairs'
dump_to_pkl(pairs, output_file_name)

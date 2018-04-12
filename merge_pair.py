import os
from utils import load_from_pkl, dump_to_pkl, logging_set
from tqdm import tqdm
import logging
import gc

logging_set('merge_pair.log')

path = 'data/pair'
files= os.listdir(path)
pairs = dict()
for idx, file in enumerate(tqdm(files)):
    if idx % 20 == 0:
        gc.collect()    #手动触发 内存回收

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

output_file_name = 'data/pairs.pkl'
dump_to_pkl(pairs, output_file_name)

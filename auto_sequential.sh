#!/bin/sh
PROCESS_FLAG="preprocess.py"

index_begin=1
index_end=1592

###########AUTOMATION######################
python preprocess.py data/corpus_${index}.0.txt data/wvect.txt data/word2id.txt data/id2word.txt data/topfrequent.txt data/pair/pair_${index}.pkl


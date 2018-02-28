#!/bin/sh
PROCESS_FLAG="preprocess.py"

if [ $# -ge 2 ]; then
index_begin=$1
index_end=$2
else
index_begin=1
index_end=1592
fi

###########AUTOMATION######################
for ((index=$index_begin; index<=$index_end; index ++))
do
    echo $index
    python preprocess.py data/corpus_${index}.0.txt data/wvect.txt data/word2id.txt data/id2word.txt data/topfrequent.txt data/pair/pair_${index}.pkl
done
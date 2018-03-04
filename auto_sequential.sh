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
    if [ ! -f "data/pair/pair_${index}.pkl" ]; then
        echo `date +%c` $index "start"
        python preprocess.py data/corpus_${index}.0.txt data/wvect.txt data/word2id.txt data/id2word.txt data/topfrequent.txt data/pair/pair_${index}.pkl
    else
        echo `date +%c` $index "already exist"
    fi
done

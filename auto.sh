#!/bin/bash


PROCESS_FLAG="preprocess_new.py"
MAX_PROCESS_NUM=10

index_begin=1
index_end=102


###########AUTOMATION######################

index=$index_begin
while true
do

count=`ps -ef |grep $PROCESS_FLAG |grep -v "grep" |wc -l`
if [ $count -lt $MAX_PROCESS_NUM ]
then
    run_time=0
    while [ $run_time -eq 0 ];
    do
        output_path="data/pair5/pair_${index}.pkl"
        if [ ! -f ${output_path} ]; then
            nohup python preprocess_new.py data/corpus/corpus_${index}.0.txt data/wvect.txt data/word2id.txt $output_path  &>logs/preprocess_${index}.txt &
            run_time=`expr $run_time + 1`
        fi
        index=`expr $index + 1`
    done
fi

if [ $index -gt $index_end ]
then
    echo `date +%c` "All the commands have been started"
    break
fi

echo `date +%c` "Current working process num:${count}, the next index is: $index"
sleep 3 #detect every 180 seconds
done

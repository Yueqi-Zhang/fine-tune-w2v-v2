import codecs

path = 'data/chinesegigaword.seg.rewrite.new'
i = 0
data = []
with codecs.open(path, 'r', encoding='utf-8') as f:
    for lines in f:
        i+=1
        data.append(lines)
        if i % 1000000 == 0:
            path_o = 'data/corpus/corpus_'+str(i/1000000)+'.txt'
            with codecs.open(path_o, 'w', encoding='utf-8') as f:
                for line in data:
                    f.write(line)
            data = []

import codecs

path = '/Users/yueqizhang/Documents/THUNLP-Intern/deepwalk/THULAC_lite_c++_v1/chinesegigaword.seg.rewrite'
data = []
with codecs.open(path, 'r') as f:
    for lines in f:
        l_n = lines.strip().split()
        if ',' in l_n:
            l_n.insert(l_n.index(',') + 1, '\n')
            data.append(l_n)
        elif '。' in l_n:
            l_n.insert(l_n.index('。') + 1, '\n')
            data.append(l_n)
        elif ';' in l_n:
            l_n.insert(l_n.index(';') + 1, '\n')
            data.append(l_n)
        else:
            data.append(l_n+['\n'])

path_w = '/Users/yueqizhang/Documents/THUNLP-Intern/word2vec_finetune/project2/data/chinesegigaword.seg.rewrite.new'
with codecs.open(path_w, 'w') as f:
    for lines in data:
        f.write(' '.join(lines))
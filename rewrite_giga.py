import codecs

path = '/Users/yueqizhang/Documents/THUNLP-Intern/deepwalk/THULAC_lite_c++_v1/chinesegigaword.seg.rewrite'
path_w = '/Users/yueqizhang/Documents/THUNLP-Intern/word2vec_finetune/project2/data/chinesegigaword.seg.rewrite.new'
with codecs.open(path, 'r') as fin:
    with codecs.open(path_w, 'w') as fout:
        for lines in fin:
            l_n = lines.strip().split()
            if ',' in l_n:
                l_n.insert(l_n.index(',') + 1, '\n')
                fout.write(' '.join(l_n)+'\n')
            elif '。' in l_n:
                l_n.insert(l_n.index('。') + 1, '\n')
                fout.write(' '.join(l_n)+'\n')
            elif ';' in l_n:
                l_n.insert(l_n.index(';') + 1, '\n')
                fout.write(' '.join(l_n)+'\n')
            else:
                fout.write(' '.join(l_n)+'\n')


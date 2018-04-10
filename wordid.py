from input_data import InputData
import codecs

input_file_name = 'data/chinesegigaword.seg.rewrite.new'
min_count = 30
data = InputData(input_file_name, min_count)
path_o1 = 'data/word2id.txt'
with codecs.open(path_o1, 'w', encoding='utf-8') as f:
    for key in data.word2id.keys():
        f.write(key+' '+str(data.word2id[key])+'\n')

path_o2 = 'data/id2word.txt'
with codecs.open(path_o2, 'w', encoding='utf-8') as f:
    for key in data.id2word.keys():
        f.write(str(key)+' '+str(data.id2word[key])+'\n')

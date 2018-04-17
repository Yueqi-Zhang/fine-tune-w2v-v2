from input_data import InputData
from utils import Topfreq
import codecs

input_file_name = 'data/chinesegigaword.seg.rewrite.new'
min_count = 30
data = InputData(input_file_name, min_count)
vocabulary = Topfreq(data.word_frequency)
path_o1 = 'data/vocab_freq.txt'
with codecs.open(path_o1, 'w', encoding='utf-8') as f:
    for lines in vocabulary:
        f.write(' '.join(lines)+'\n')

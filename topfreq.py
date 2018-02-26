from input_data import InputData
from utils import Topfreq
import codecs

input_file_name = 'data/chinesegigaword.seg.new'
min_count = 30
data = InputData(input_file_name, min_count)
topfrequent = Topfreq(data.word_frequency)
path_o1 = 'data/topfrequent.txt'
with codecs.open(path_o1, 'w', encoding='utf-8') as f:
    for lines in topfrequent:
        f.write(str(lines)+'\n')

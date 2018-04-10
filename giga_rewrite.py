import codecs

path = '/Users/yueqizhang/Documents/THUNLP-Intern/deepwalk/data/chinesegigaword'
data = []
with codecs.open(path, 'r') as f:
    for lines in f:
        l_n = lines.strip()
        data.append(l_n)

for i in range(len(data)):
    if data[i] == '':
        data[i] = data[i]+' '
data_n = ''.join(data)

data_n_n = data_n.split()
path_w = '/Users/yueqizhang/Documents/THUNLP-Intern/deepwalk/data/chinesegigaword.rewrite'
with codecs.open(path_w, 'w') as f:
    for lines in data_n_n:
        f.write(lines+'\n')

from gensim.models import word2vec
import codecs

sentences = word2vec.LineSentence('data/chinesegigaword.seg.rewrite.new')
model = word2vec.Word2Vec(sentences, size=100, min_count = 30, window=5, negative=10, sample=1e-4, workers = 4, iter=5)

id2word = model.wv.index2word

path = 'data/cvect_rewrite.txt'
with codecs.open(path, 'w') as f:
    f.write(str(len(id2word))+' '+'100'+'\n')
    for i in range(len(id2word)):
        a = [str(x) for x in model.syn1neg[i]]
        f.write(id2word[i]+' '+' '.join(a)+'\n')
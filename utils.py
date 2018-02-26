from gensim.models import KeyedVectors
import codecs

def Topfreq(freqence):
    """计算出语料库里面最高频的1W个词

    Args:
        freqence:

    Returns:

    """
    sortd = sorted(freqence.items(), key=lambda x: x[1], reverse=True)
    sortd = sortd[:10000]
    tf = [x for x, y in sortd]
    return tf

def KNeighbor(file_name, words, word2id, id2word):
    """计算高频词里面每个词的K近邻

    Args:
        file_name:
        words:
        word2id:
        id2word:

    Returns:

    """
    vectors = KeyedVectors.load_word2vec_format(file_name, binary=False)
    neighbor = dict()
    for word in words:
        neighbor[word] = []

        for _, (w, _) in enumerate(vectors.most_similar(positive=[id2word[word]], topn=10)):
            neighbor[word].append(word2id[w])
    return neighbor

def Get_pairs(data, ids, neighbor, window_size):
    """从语料库里面得到pairs：  Wc (center), Wt (tihuan), [context words...]

    Args:
        data:
        ids:
        neighbor:
        window_size:

    Returns:

    """
    pairs = []
    word_count = len(data.word2id)
    file_name = data.input_file_name
    words = [data.id2word[x] for x in ids]
    with codecs.open(file_name, 'r') as f:
        for lines in f:
            lines = lines.strip().split()
            for word in words:
                if word in lines:
                    sentence_ids = []
                    for w in lines:
                        try:
                            sentence_ids.append(data.word2id[w])
                        except:
                            continue
                    i = sentence_ids.index(data.word2id[word])
                    u = data.word2id[word]
                    c = []
                    for j, v in enumerate(sentence_ids[max(i - window_size, 0):i + window_size+1]):
                        assert u < word_count
                        assert v < word_count
                        if i < window_size & j == i:
                            continue
                        elif i >= window_size & j == 5:
                            continue
                        else:
                            c.append(v)
                    for n in neighbor[data.word2id[word]]:
                        pairs.append((u, n, tuple(c))) # u:center word, n:neighbor, v:context
    return pairs

def Batch_pairs(pairs, batch_size, i, iteration):
    pairs_n = []
    for iter in range(iteration):
        for pair in pairs:
            pairs_n.append(pair)
    return pairs_n[i*batch_size:(i+1)*batch_size]

def Get_VSP(w1, w2, w3):
    w1 = set(w1)
    w2 = set(w2)
    w3_n = set()
    for i in w3:
        w3_n = w3_n|set(i)
    w_n = w1|w2|w3_n
    return list(w_n)

def V_Pad(batch_pairs, window_size):
    batch_v = []
    mask = []
    for _, _, c in batch_pairs:
        if len(c) < 2*window_size:
            c_n = list(c) + [0]*(2*window_size-len(c))
            m = [1]*len(c) + [0]*(2*window_size-len(c))
        else:
            c_n = list(c)
        batch_v.append(c_n)
        mask.append(m)
    return batch_v, mask


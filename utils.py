from gensim.models import KeyedVectors
import codecs
import random
import os
import re
import tqdm
import logging
import pickle as pkl

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

def Batch_pairs(pairs, batch_size):
    length = len(pairs)
    a = [random.randint(0, length-1) for i in range(batch_size)]
    pairs_n = []
    for index in a:
        pairs_n.append(pairs[index])
    return pairs_n

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
            m = [1]*len(c)
        batch_v.append(c_n)
        mask.append(m)
    return batch_v, mask

def dir_traversal(dir_path, only_file=True):
    #获取dir_path目录下的所有文件路径，返回路径list
    file_list = []
    for lists in os.listdir(dir_path):
        path = os.path.join(dir_path, lists)
        if os.path.isdir(path):
            if(only_file == False):
                file_list.append(path)
            file_list.extend(dir_traversal(path))
        else:
            file_list.append(path)
    return file_list

def get_preprocessed_pairs(pair_path, format='pkl', sample_rate=0.1):
    """

    Args:
        pair_path:
        format: pkl/txt

    Returns:
        a generator that produces pairs: (center_word_id, replace_word_id, context_word_ids)

    """
    #pairs = []

    if os.path.exists(pair_path):   # if it is a dir
        pair_file_paths = dir_traversal(pair_path)
        print('Starting to get pairs from preprocessed dir...')
        #for pair_file_path in tqdm.tqdm(pair_file_paths):
        for pair_file_path in tqdm.tqdm(pair_file_paths):
            if os.path.basename(pair_file_path).startswith('pair_'):
                logging.debug('%s trained')
                if format == 'txt':
                    with codecs.open(pair_file_path, 'r', encoding='utf-8') as fin:
                        for line in fin:
                            matchObj = re.match('([0-9]+) ([0-9]+) \((.*)\)', line)
                            if matchObj is not None:
                                try:
                                    center_word_id = int(matchObj.group(1))
                                    replace_word_id = int(matchObj.group(2))
                                    context_word_ids_str = matchObj.group(3).split(',')

                                    context_word_ids = []
                                    for context_word_id_str in context_word_ids_str:
                                        context_word_ids.append(int(context_word_id_str))
                                    if random.random() <= sample_rate:
                                        yield (center_word_id, replace_word_id, context_word_ids)
                                    #pairs.append((center_word_id, replace_word_id, context_word_ids))
                                except:
                                    pass
                elif format == 'pkl':
                    pairs = load_from_pkl(pair_file_path)
                    for pair in pairs:
                        if random.random() <= sample_rate:
                            yield pair
    elif os.path.isfile(pair_path):
        if format == 'txt':
            with codecs.open(pair_path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    matchObj = re.match('([0-9]+) ([0-9]+) \((.*)\)', line)
                    if matchObj is not None:
                        try:
                            center_word_id = int(matchObj.group(1))
                            replace_word_id = int(matchObj.group(2))
                            context_word_ids_str = matchObj.group(3).split(',')

                            context_word_ids = []
                            for context_word_id_str in context_word_ids_str:
                                context_word_ids.append(int(context_word_id_str))

                            if random.random() <= sample_rate:
                                yield (center_word_id, replace_word_id, context_word_ids)
                            # pairs.append((center_word_id, replace_word_id, context_word_ids))
                        except:
                            pass
        elif format == 'pkl':
            pairs = load_from_pkl(pair_path)
            random.shuffle(pairs)
            for pair in pairs:
                if random.random() <= sample_rate:
                    yield pair

    #return pairs


def get_batch_pairs(pair_generator, batch_size):
    """ generate a batch of pairs from pair generator

    Args:
        pair_generator:

    Returns:
        batch_pairs / None (current epoch is over)

        len(batch_pairs) of the last batch might be less than batch_size

    """
    batch_pairs = []
    for i in range(batch_size):
        try:
            pair = pair_generator.__next__()
            batch_pairs.append(pair)
        except:
            break
    if len(batch_pairs) > 0:
        return batch_pairs
    else:
        return None

def logging_set(log_path):
    """
    Note: if you invoke logging.info or something before basicConfig, some problems may appear because
    the logging module has fabricate a default configuration

    Args:
    log_path:

    Returns:

    """

    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(filename=log_path, filemode='w',
        format='%(asctime)s %(levelname)s %(filename)s %(funcName)s %(lineno)d: %(message)s',
        level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s %(filename)s %(funcName)s %(lineno)d: %(message)s'))
    logging.getLogger().addHandler(console)


def load_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as fin:
        obj = pkl.load(fin)
    return obj

def dump_to_pkl(obj, pkl_path):
    with open(pkl_path, 'wb') as fout:
        pkl.dump(obj, fout)


if __name__ == "__main__":
    a = load_from_pkl('data/pair_zhihu/pair_zhihu.pkl')
    pass



import codecs
import sys
import os
from tqdm import tqdm
import logging
from utils import load_from_pkl, KNeighbor, dump_to_pkl, logging_set
import debugger

class NSselect:
    def __init__(self,
                 input_wvectors,
                 input_word2id,
                 input_id2word,
                 input_vocabulary,
                 pair_file_path,
                 kn_file_name,
                 output_file_name,
                 topn = 20):
        word2id = dict()
        with codecs.open(input_word2id, 'r', encoding='utf-8') as f:
            for lines in f:
                word2id[lines.strip().split()[0]] = int(lines.strip().split()[1])
        id2word = dict()
        with codecs.open(input_id2word, 'r', encoding='utf-8') as f:
            for lines in f:
                id2word[int(lines.strip().split()[0])] = lines.strip().split()[1]
        vocabulary = []
        with codecs.open(input_vocabulary, 'r', encoding='utf-8') as f:
            for lines in f:
                vocabulary.append(int(lines.strip()))

        self.topn = topn
        logging.info("get kneighbors...")
        kneighbor = load_from_pkl(kn_file_name)
        #kneighbor = KNeighbor(input_wvectors, vocabulary, word2id, id2word)
        #dump_to_pkl(kneighbor, kn_file_name)
        logging.info("kneightbors got.")

        logging.info("get pairs...")
        files = os.listdir(pair_file_path)[1:]
        pairs = dict()
        for file in tqdm(files):
            if not os.path.isdir(file):
                path = pair_file_path + "/" + file
                pair = load_from_pkl(path)
                logging.info("pair size: %d" % (len(pair)))
                if len(pairs) == 0:
                    pairs = pair
                else:
                    for key in pair.keys():
                        if key in pairs:
                            pairs[key] += pair[key]
                        else:
                            pairs[key] = pair[key]
                logging.info("current total pair size: %d" % (len(pairs)))
        logging.info("pairs got")

        logging.info("len(word2id): %d" % len(word2id))
        keys_before_sort_set = set([key[0] for key in pairs.keys()])
        logging.info("length of pair.keys[0]: %d" % len(keys_before_sort_set))
        id_missing_in_pairs = set(word2id.values()) - keys_before_sort_set
        logging.info("len(id_missing_in_pairs): %d" % (len(id_missing_in_pairs)))
        if len(id_missing_in_pairs) > 0:
            logging.info("missing word in pairs: %s" % str(id_missing_in_pairs))
            #dump_to_pkl(id_missing_in_pairs, './data/id_missing_in_pairs.pkl')

        logging.info("start calculate score")

        pair_rp = self.select_new(pairs, kneighbor, self.topn)
        logging.info("len(score): %d" % len(pair_rp))

        #score1 = self.select(pairs, kneighbor)
        logging.info("start saving")
        dump_to_pkl(pair_rp, output_file_name)

    def select_new(self, pairs, kneighbor, topn):
        pair_rp = dict()
        ids = ['1428', '1025', '107', '43', '2060', '559', '2720', '4450', '4306', '3442', '3558', '1914', '3516', '3856', '2076', '3323', '2768', '1421', '2124', '3549']
        logging.info("start sorting...")
        key_sorted = sorted(pairs.keys(), key=lambda tup: tup[0])
        logging.info("sort length: %d" % len(key_sorted))
        keys_after_sort_set = set([key[0] for key in key_sorted])
        logging.info('number of keys after sort: %d' % len(keys_after_sort_set))
        logging.info("sort finished.")
        for keyp in tqdm(key_sorted):
            if str(keyp[0]) in ids:
                pair_rp[keyp] = dict()
                for value in kneighbor[keyp[0]]:
                    replace = tuple([value] + list(keyp[1:]))
                    if replace in pairs:
                        pair_rp[keyp][replace] = pairs[replace]
                    else:
                        pair_rp[keyp][replace] = 0
        return pair_rp

if __name__ == '__main__':
    logging_set('NSselect.log')
    ns = NSselect(input_wvectors=sys.argv[1], input_word2id = sys.argv[2], input_id2word = sys.argv[3], input_vocabulary = sys.argv[4], pair_file_path=sys.argv[5], kn_file_name = sys.argv[6], output_file_name=sys.argv[7])

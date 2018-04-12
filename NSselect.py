import codecs
import sys
import os
from tqdm import tqdm
import logging
from utils import load_from_pkl, KNeighbor, dump_to_pkl, logging_set


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
        kneighbor = KNeighbor(input_wvectors, vocabulary, word2id, id2word)
        dump_to_pkl(kneighbor, kn_file_name)

        logging_set('NSselect.log')
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
        logging.info("start calculate score")
        score = self.select_new(pairs, kneighbor, self.topn)
        #score1 = self.select(pairs, kneighbor)
        logging.info("start saving")
        dump_to_pkl(score, output_file_name)

    def select_new(self, pairs, kneighbor, topn):
        score = dict()
        for keyp in pairs.keys():
            if keyp[0] in kneighbor:
                if not keyp[0] in score:
                    score[keyp[0]] = []
                #i = 0
                for value in kneighbor[keyp[0]]:
                    replace = tuple([value] + list(keyp[1:]))
                    if replace in pairs:
                        s = pairs[replace]/pairs[keyp]
                    else:
                        s = 0
                    #i += 1
                    score[keyp[0]].append(s)
        score_f = dict()
        for key in score:
            score_f[key] = []
            iter = len(score[key]) / topn
            for i in range(topn):
                s_f = sum([score[key][x*topn+i] for x in range(int(iter))])/iter
                score_f[key].append(s_f)
        return score_f


    def select(self, pairs, kneighbor):
        score = dict()
        for keyn in tqdm(kneighbor.keys()):
            score[keyn] = []
            for value in kneighbor[keyn]:
                s = 0
                i = 0
                for keyp in pairs.keys():
                    if keyp[0]==keyn:
                        replace = tuple([value]+list(keyp[1:]))
                        if replace in pairs:
                            s += pairs[replace]/pairs[keyp]
                            i += 1
                        else:
                            s += 0
                            i +=1
                score[keyn].append(s/i)
        return score


if __name__ == '__main__':
    ns = NSselect(input_wvectors=sys.argv[1], input_word2id = sys.argv[2], input_id2word = sys.argv[3], input_vocabulary = sys.argv[4], pair_file_path=sys.argv[5], kn_file_name = sys.argv[6], output_file_name=sys.argv[7])

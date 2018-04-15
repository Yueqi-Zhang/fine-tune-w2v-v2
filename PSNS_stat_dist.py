import codecs
import sys
from utils import load_from_pkl

class PSNS:
    def __init__(self,
                 path_sim_word,
                 path_score,
                 path_id2word,
                 path_kneighbor,
                 threshold = 0.01):

        self.path_sim_word = path_sim_word
        self.path_score = path_score
        self.path_id2word = path_id2word
        self.path_kneighbor = path_kneighbor
        self.threshold = threshold
        synset = self.get_synset(self.path_sim_word)
        score = load_from_pkl(self.path_score)
        kneighbor = load_from_pkl(self.path_kneighbor)
        id2word = self.get_id2word(self.path_id2word)
        ps, ns = self.get_psns(score, id2word, kneighbor, self.threshold)
        f1, precision, recall = self.ps_stat(synset, ps)
        print('precision: %f' % precision)
        print('recall: %f' % recall)
        print('f1: %f' % f1)

    def get_synset(self, path):
        synset = dict()
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for lines in f:
                lines = lines.split()
                if lines[1] not in synset.keys():
                    synset[lines[1]] = set()
                    synset[lines[1]].add(lines[0])
                else:
                    synset[lines[1]].add(lines[0])
        return synset

    def get_id2word(self, path):
        id2word = dict()
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for lines in f:
                lines = lines.split()
                id2word[lines[0]] = lines[1]
        return id2word

    def get_psns(self, score, id2word, kneighbor, threshold):
        ps = dict()
        for id in score.keys():
            ps[id2word[id]] = []
            for i in range(len(score[id])):
                if score[id][i] >= threshold:
                    ps[id2word[id]].append(id2word(kneighbor[id][i]))
        ns = dict()
        for id in score.keys():
            ns[id2word[id]] = []
            for i in range(len(score[id])):
                if score[id][i] < threshold:
                    ns[id2word[id]].append(id2word(kneighbor[id][i]))
        return ps, ns

    def ps_stat(self, synset, ps):
        vocab_set = set(ps.keys())
        num_ttlcrct = 0.0
        num_crct = 0.0
        num_select = 0.0
        for word in ps.keys():
            synset_w = set()
            for key in synset.keys():
                if word in synset[key]:
                    synset_w = synset_w | synset[key]
            synset_w = synset_w & vocab_set
            ps_w = set(ps[word])
            num_ttlcrct += len(synset_w) - 1
            num_crct += len(synset_w & ps_w)
            num_select += len(ps_w)
        presicion = num_crct / num_select
        recall = num_crct / num_ttlcrct
        f1 = (2*presicion*recall)/(presicion+recall)

        return f1, presicion, recall


if __name__ == "__main__":
    PSNS(path_sim_word = sys.argv[1], path_score = sys.argv[2], path_id2word=sys.argv[3], path_kneighbor=sys.argv[4])

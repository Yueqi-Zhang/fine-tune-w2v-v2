import logging
from utils import logging_set
from gensim.models import Word2Vec
import argparse
import gensim
from glove import Glove, metrics  #pip install glove_python
from collections import defaultdict
import numpy as np

def analogy_test_by_gensim(model, analogies_file_path):
    """ this function is abandoned

        Example analogy:
            amazing amazingly calm calmly
        This analogy is marked correct if:
            embedding(amazing) - embedding(amazingly) = embedding(calm) - embedding(calmly)

        ##refer to :https://markroxor.github.io/gensim/static/notebooks/Word2Vec_FastText_Comparison.html
    Args:
        model: loaded gensim word2vec model
        analogies_file_path:

    Returns:

    """
    acc = model.accuracy(analogies_file_path)

    '''
    # for question-words.txt, there are 13 types
    sem_correct = sum((len(acc[i]['correct']) for i in range(5)))
    sem_total = sum((len(acc[i]['correct']) + len(acc[i]['incorrect'])) for i in range(5))
    sem_acc = 100 * float(sem_correct) / sem_total
    logging.debug('\nSemantic: {:d}/{:d}, Accuracy: {:.2f}%'.format(sem_correct, sem_total, sem_acc))

    syn_correct = sum((len(acc[i]['correct']) for i in range(5, len(acc) - 1)))
    syn_total = sum((len(acc[i]['correct']) + len(acc[i]['incorrect'])) for i in range(5, len(acc) - 1))
    syn_acc = 100 * float(syn_correct) / syn_total
    logging.debug('Syntactic: {:d}/{:d}, Accuracy: {:.2f}%\n'.format(syn_correct, syn_total, syn_acc))
    return (sem_acc, syn_acc)
    '''
    sem_correct = sum((len(acc[i]['correct']) for i in range(4)))
    sem_total = sum((len(acc[i]['correct']) + len(acc[i]['incorrect'])) for i in range(4))
    sem_acc = 100 * float(sem_correct) / sem_total
    logging.debug('\nSemantic: {:d}/{:d}, Accuracy: {:.2f}%'.format(sem_correct, sem_total, sem_acc))
    return sem_acc



def analogy_test_by_glove(model, analogies_file_path, to_encode, no_threads=1):
    """

        Example analogy:
            amazing amazingly calm calmly
        This analogy is marked correct if:
            embedding(amazing) - embedding(amazingly) = embedding(calm) - embedding(calmly)

        refer to :
            https://github.com/maciejkula/glove-python/blob/master/glove/metrics/accuracy.py
            https://github.com/maciejkula/glove-python/blob/master/examples/analogy_tasks_evaluation.py
    Args:
        model: loaded gensim word2vec model
        analogies_file_path:

    Returns:

    """
    if to_encode:
        encode = lambda words: [x.lower().encode('utf-8') for x in words]
    else:
        encode = lambda words: [x.lower().decode('utf-8') for x in words]

    sections = defaultdict(list)
    evaluation_words = [sections[section].append(encode(words)) for section, words in
                        metrics.read_analogy_file(analogies_file_path)]
    section_ranks = []
    for section, words in sections.items():
        evaluation_ids = metrics.construct_analogy_test_set(words, model.dictionary, ignore_missing=True)

        # Get the rank array.
        ranks = metrics.analogy_rank_score(evaluation_ids, model.word_vectors, no_threads=no_threads)
        section_ranks.append(ranks)

        logging.info('Section %s mean rank: %s, accuracy: %s' % (section, ranks.mean(), (ranks == 0).sum() / float(len(ranks))))

    ranks = np.hstack(section_ranks)
    logging.info('Overall rank: %s, accuracy: %s' % (ranks.mean(), (ranks == 0).sum() / float(len(ranks))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Philly arguments parser")

    parser.add_argument('emb_file_name', type=str)
    parser.add_argument('analogy_test_data', type=str)
    parser.add_argument('--to_encode', type=bool, default=False, help=('If True, words from the '
                              'evaluation set will be utf-8 encoded '
                              'before looking them up in the '
                              'model dictionary'))
    parser.add_argument('--no_threads', type=int, default=1)
    parser.add_argument('--log_path', type=str, default='train.log')
    args, _ = parser.parse_known_args()
    logging_set(args.log_path)

    logging.info('Evaluating analogy...\n')
    #my_model = gensim.models.KeyedVectors.load_word2vec_format(args.emb_file_name, binary=False, unicode_errors='ignore')
    #sem_acc, syn_acc = analogy_test_by_gensim(my_model, args.analogy_test_data)
    #logging.info('Semantic accuracy: %.2f; Syntactic accuracy: %.2f' % (sem_acc, syn_acc))


    model = Glove.load(args.model)
    analogy_test_by_glove(model, args.analogy_test_data, args.to_encode, args.no_threads)



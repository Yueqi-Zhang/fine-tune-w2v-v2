import logging
from utils import logging_set
from gensim.models import Word2Vec
import argparse
import gensim


def analogy_test(model, analogies_file_path):
    """

        Example analogy:
            amazing amazingly calm calmly
        This analogy is marked correct if:
            embedding(amazing) - embedding(amazingly) = embedding(calm) - embedding(calmly)
    Args:
        model: loaded gensim word2vec model
        analogies_file_path:

    Returns:

    """
    acc = model.accuracy(analogies_file_path)

    sem_correct = sum((len(acc[i]['correct']) for i in range(5)))
    sem_total = sum((len(acc[i]['correct']) + len(acc[i]['incorrect'])) for i in range(5))
    sem_acc = 100 * float(sem_correct) / sem_total
    logging.debug('\nSemantic: {:d}/{:d}, Accuracy: {:.2f}%'.format(sem_correct, sem_total, sem_acc))

    syn_correct = sum((len(acc[i]['correct']) for i in range(5, len(acc) - 1)))
    syn_total = sum((len(acc[i]['correct']) + len(acc[i]['incorrect'])) for i in range(5, len(acc) - 1))
    syn_acc = 100 * float(syn_correct) / syn_total
    logging.debug('Syntactic: {:d}/{:d}, Accuracy: {:.2f}%\n'.format(syn_correct, syn_total, syn_acc))
    return (sem_acc, syn_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Philly arguments parser")

    parser.add_argument('emb_file_name', type=str)
    parser.add_argument('analogy_test_data', type=str)
    parser.add_argument('--log_path', type=str, default='train.log')
    args, _ = parser.parse_known_args()
    logging_set(args.log_path)

    logging.info('Evaluating analogy...\n')
    my_model = gensim.models.KeyedVectors.load_word2vec_format(args.emb_file_name, binary=False, unicode_errors='ignore')
    sem_acc, syn_acc = analogy_test(my_model, args.analogy_test_data)
    logging.info('Semantic accuracy: %.2f; Syntactic accuracy: %.2f' % (sem_acc, syn_acc))

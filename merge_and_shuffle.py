from utils import get_preprocessed_pairs, dump_to_pkl
import random
import os

def merge_and_shuffle(input_dir, output_shuffled_dir, max_sample_per_file=10000000):
    """ merge the preprocessed pair files and shuffle them. Finally, the pairs are split again and saved to output_shuffled_dir

    Args:
        input_dir:  the input dir to merge
        output_shuffled_dir:  the output file

    Returns:

    """
    batch_pairs = []
    pro_pairs_generator = get_preprocessed_pairs(input_dir, 'pkl')
    while True:
        try:
            pair = pro_pairs_generator.__next__()
            batch_pairs.append(pair)
        except:
            break

    print("Import data finished")
    random.shuffle(batch_pairs)
    print("Data shuffle finished")

    if not os.path.exists(output_shuffled_dir):
        os.makedirs(output_shuffled_dir)
    if len(batch_pairs) % max_sample_per_file != 0:
        split_num = len(batch_pairs) // max_sample_per_file + 1
    else:
        split_num = len(batch_pairs) / max_sample_per_file
    print("Starting to write data")
    for i in range(split_num):
        dump_to_pkl(batch_pairs[max_sample_per_file * i: max_sample_per_file * (i + 1)], os.path.join(output_shuffled_dir, 'pair_%d.pkl' % i))


if __name__ == '__main__':
    #merge_and_shuffle('data/pair_zhihu/', 'data/pair_zhihu_shuffled/')
    merge_and_shuffle('data/pair_giga_noshuffle/', 'data/pair_giga_shuffled/1st')


from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim

import numpy
from tqdm import tqdm
import sys
import argparse
import codecs
import logging
import os
import tempfile
import random
import string
import shutil

from utils import get_preprocessed_pairs
import debugger
from evaluation import evaluation
from input_data import InputData
from input_data import InputVector
from utils import Topfreq, KNeighbor, Get_pairs, Batch_pairs, Get_VSP, V_Pad, get_batch_pairs, logging_set
from model import FineTuneModel

class Word2Vec:
    def __init__(self,
                 input_file_name,
                 input_wvectors,
                 input_cvectors,
                 output_file_name,
                 preprocessed_pair_dir,
                 input_word2id,
                 input_id2word,
                 input_topfrequent,
                 emb_dimension=100,
                 batch_size=50,
                 window_size=5,
                 iteration=20,
                 initial_lr=0.025,
                 min_count=30,
                 p = 0.0,
                 sigma = 1e-9,
                 clip = 1.0,
                 batch_num_to_show_progress=10000,
                 batch_num_to_valid=100000,
                 ):
        """Initilize class parameters.

        Args:
            input_file_name: Name of a text data from file. Each line is a sentence splited with space.
            output_file_name: Name of the final embedding file.
            emb_dimention: Embedding dimention, typically from 50 to 500.
            batch_size: The count of word pairs for one forward.
            window_size: Max skip length between words.
            iteration: Control the multiple training iterations.
            initial_lr: Initial learning rate.
            min_count: The minimal word frequency, words with lower frequency will be filtered.
            p: The minimum probability distance of center word and neighbor word
            sigma: L2 regularisation constant
            batch_num_to_show_progress: every batch_num_to_show_progress batches, logging the current loss and learning rate.

        Returns:
            None.
        """
        #self.data = InputData(input_file_name, min_count)
        self.input_wvect = InputVector(input_wvectors)
        self.input_cvect = InputVector(input_cvectors)
        self.output_file_name = output_file_name
        self.preprocessed_pair_dir = preprocessed_pair_dir

        #self.word2id = self.data.word2id
        #self.id2word = self.data.id2word
        #self.topfrequent = Topfreq(self.data.word_frequency)
        self.word2id = dict()
        with codecs.open(input_word2id, 'r', encoding='utf-8') as f:
            for lines in f:
                self.word2id[lines.strip().split()[0]] = int(lines.strip().split()[1])
        logging.info('word2id got!')
        self.id2word = dict()
        with codecs.open(input_id2word, 'r', encoding='utf-8') as f:
            for lines in f:
                self.id2word[int(lines.strip().split()[0])] = lines.strip().split()[1]
        logging.info('id2word got!')
        self.topfrequent = []
        with codecs.open(input_topfrequent, 'r', encoding='utf-8') as f:
            for lines in f:
                self.topfrequent.append(int(lines.strip()))
        logging.info('topfrequent got!')

        #self.emb_size = len(self.data.word2id)
        self.emb_size = len(self.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.window_size = window_size
        self.iteration = iteration
        self.initial_lr = initial_lr
        self.p = p
        self.sigma = sigma
        self.clip = clip

        self.batch_num_to_show_progress = batch_num_to_show_progress
        self.batch_num_to_valid = batch_num_to_valid

        #self.kneighbor = KNeighbor(input_wvectors, self.topfrequent, self.data.word2id, self.data.id2word)
        self.kneighbor = KNeighbor(input_wvectors, self.topfrequent, self.word2id, self.id2word)
        self.fine_tune_model = FineTuneModel(self.emb_size, self.emb_dimension, self.p, self.sigma, self.input_wvect, self.input_cvect)
        self.fine_tune_model = nn.DataParallel(self.fine_tune_model)
        self.use_cuda = torch.cuda.is_available()
        '''
        if self.use_cuda:
            self.fine_tune_model.cuda()
        '''

        self.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.fine_tune_model.parameters()), lr=self.initial_lr, momentum=0.9)


    def train(self, similarity_test_paths, synset_paths, analogy_paths, sample_rate=1):
        """Multiple training.

        Returns:
            None.
        """
        #pair_count = self.data.evaluate_pair_count(self.window_size)
        #pro_pairs = Get_pairs(self.data, self.topfrequent, self.kneighbor, self.window_size)

        #pro_pairs = get_preprocessed_pairs(self.preprocessed_pair_dir)

        #vsp_pairs = Get_VSP(self.input_wvect, self.topfrequent, rho=self.rho)
        #pair_count = len(pro_pairs)
        #batch_count = self.iteration * pair_count / self.batch_size
        #process_bar = tqdm(range(int(batch_count)))
        # self.skip_gram_model.save_embedding(
        #     self.data.id2word, 'begin_embedding.txt', self.use_cuda)

        tmp_emb_dir = os.path.join(tempfile.gettempdir(), 'embedding')
        tmp_emb_path = os.path.join(tmp_emb_dir, ''.join(random.sample(string.ascii_letters + string.digits, 16)))
        if not os.path.exists(tmp_emb_dir):
            os.makedirs(tmp_emb_dir)

        #for i in process_bar:
        batch_count = 0
        best_scores = dict()
        for epoch in range(self.iteration):
            pro_pairs_generator = get_preprocessed_pairs(self.preprocessed_pair_dir, 'pkl', sample_rate=sample_rate)
            i = 0
            tot_loss = 0
            while True:
                i += 1
                #pos_pairs = self.data.get_batch_pairs(self.batch_size,
                                                      #self.window_size)
                #batch_pairs = Batch_pairs(pro_pairs, self.batch_size, i, self.iteration)
                #batch_pairs = Batch_pairs(pro_pairs, self.batch_size)
                batch_pairs = get_batch_pairs(pro_pairs_generator, self.batch_size)
                if batch_pairs is None:
                    break
                if epoch == 0:
                    batch_count += 1

                #neg_v = self.data.get_neg_v_neg_sampling(pos_pairs, 5)
                #batch_pair = batch_pairs[i % self.iteration]
                batch_u, batch_n, batch_v = [], [], []
                for pair in batch_pairs:
                    batch_u.append(pair[0])
                    batch_n.append(pair[1])
                    batch_v.append(pair[2])
                batch_v_pad, batch_v_mask = V_Pad(batch_pairs, self.window_size)
                batch_vsp = Get_VSP(batch_u, batch_n, batch_v)
                #batch_vsp_pad, batch_vsp_mask = VSP_Pad(batch_vsp, self.window_size, self.batch_size)

                batch_u = Variable(torch.LongTensor(batch_u))
                batch_n = Variable(torch.LongTensor(batch_n))
                batch_v_pad = Variable(torch.LongTensor(batch_v_pad))
                batch_v_mask = Variable(torch.FloatTensor(batch_v_mask))
                batch_vsp = Variable(torch.LongTensor(batch_vsp))
                #batch_vsp_mask = Variable(torch.LongTensor(batch_vsp_mask))

                if self.use_cuda:
                    batch_u = batch_u.cuda()
                    batch_n = batch_n.cuda()
                    batch_v_pad = batch_v_pad.cuda()
                    batch_v_mask = batch_v_mask.cuda()
                    batch_vsp = batch_vsp.cuda()
                    #batch_vsp_mask = batch_vsp_mask.cuda()

                self.optimizer.zero_grad()
                loss = self.fine_tune_model.forward(batch_u, batch_n, batch_v_pad, batch_v_mask, batch_vsp)
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.fine_tune_model.parameters(), self.clip)
                self.optimizer.step()
                tot_loss += loss.data[0]

                #process_bar.set_description("Loss: %0.8f, lr: %0.6f" % (tot_loss/(i+1), self.optimizer.param_groups[0]['lr']))
                if i % self.batch_num_to_show_progress == 0:
                    logging.info("Loss: %0.8f, lr: %0.6f" % (tot_loss/(i+1), self.optimizer.param_groups[0]['lr']))

                if i % self.batch_num_to_valid == 0:
                    logging.info('epoch%d_batch%d, evaluating...' % (epoch, i))
                    self.save_embedding(self.id2word, tmp_emb_path, self.use_cuda)

                    best_scores, save_flag = evaluation(tmp_emb_path, similarity_test_paths, synset_paths, analogy_paths, best_scores)
                    if save_flag == True:
                        emb_save_path = self.output_file_name + "_epoch%d_batch%d" % (epoch, i)
                        shutil.move(tmp_emb_path, emb_save_path)
                        logging.info('Save current embedding to %s' % emb_save_path)

                if epoch > 0:
                    if i * self.batch_size % 10000 == 0:
                        lr = previous_lr * (1.0 - 1.0 * i / batch_count)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr
                        previous_lr = lr
                else:
                    previous_lr = self.initial_lr
            #self.fine_tune_model.save_embedding(self.id2word, self.output_file_name + "_%d" % epoch, self.use_cuda)
            logging.info('final evaluating...')
            self.save_embedding(self.id2word, tmp_emb_path, self.use_cuda)
            best_scores, save_flag = evaluation(tmp_emb_path, similarity_test_paths, synset_paths, analogy_paths, best_scores)
            if save_flag == True:
                emb_save_path = self.output_file_name + "_epoch%d" % epoch
                shutil.move(tmp_emb_path, emb_save_path)
                logging.info('Save current embedding to %s' % emb_save_path)

    def save_embedding(self, id2word, file_name, use_cuda):
        """Save all embeddings to file.

        As this class only record word id, so the map from id to word has to be transfered from outside.

        Args:
            id2word: map from word id to word.
            file_name: file name.
        Returns:
            None.
        """
        if use_cuda:
            embedding = self.fine_tune_model.module.u_embeddings.weight.cpu().data.numpy()
        else:
            embedding = self.fine_tune_model.module.u_embeddings.weight.data.numpy()
        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(id2word), self.fine_tune_model.module.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Philly arguments parser")

    parser.add_argument('input_file_name', type=str)
    parser.add_argument('input_wvectors', type=str)
    parser.add_argument('input_cvectors', type=str)
    parser.add_argument('output_file_name', type=str)
    parser.add_argument('preprocessed_pair_dir', type=str)
    parser.add_argument('input_word2id', type=str)
    parser.add_argument('input_id2word', type=str)
    parser.add_argument('input_topfrequent', type=str)
    parser.add_argument('--similarity_test_paths', type=str, default='data/240.txt|data/297.txt')
    parser.add_argument('--synset_paths', type=str, default='data/nsem3-adjusted.txt')
    parser.add_argument('--analogy_test_paths', type=str, default='data/analogy.txt')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--iteration', type=int, default=1)
    parser.add_argument('--min_count', type=int, default=30)
    parser.add_argument('--initial_lr', type=float, default=0.01)
    parser.add_argument('--p', type=float, default=0.0)
    parser.add_argument('--sigma', type=float, default=1e-9)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--batch_num_to_show_progress', type=int, default=10000)
    parser.add_argument('--batch_num_to_valid', type=int, default=100000)
    parser.add_argument('--log_path', type=str, default='train.log')
    parser.add_argument('--sample_rate', type=float, default=1)
    args, _ = parser.parse_known_args()

    logging_set(args.log_path)
    #w2v = Word2Vec(input_file_name=sys.argv[1], input_wvectors = sys.argv[2], input_cvectors = sys.argv[3], output_file_name=sys.argv[4])
    w2v = Word2Vec(input_file_name=args.input_file_name, input_wvectors=args.input_wvectors, input_cvectors = args.input_cvectors,
        output_file_name=args.output_file_name, preprocessed_pair_dir=args.preprocessed_pair_dir, input_word2id=args.input_word2id,
        input_id2word=args.input_id2word, input_topfrequent=args.input_topfrequent,
        batch_size=args.batch_size, window_size=args.window_size, iteration=args.iteration, min_count=args.min_count,
        initial_lr=args.initial_lr, p=args.p, sigma=args.sigma, clip=args.clip, batch_num_to_show_progress=args.batch_num_to_show_progress,
        batch_num_to_valid=args.batch_num_to_valid)
    w2v.train(similarity_test_paths=args.similarity_test_paths, synset_paths=args.synset_paths, analogy_paths=args.analogy_test_paths,
        sample_rate=args.sample_rate)

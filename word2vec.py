from input_data import InputData
from input_data import InputVector
from utils import Topfreq, KNeighbor, Get_pairs, Batch_pairs, Get_VSP, V_Pad
import numpy
from model import FineTuneModel
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import argparse

class Word2Vec:
    def __init__(self,
                 input_file_name,
                 input_wvectors,
                 input_cvectors,
                 output_file_name,
                 emb_dimension=100,
                 batch_size=50,
                 window_size=5,
                 iteration=20,
                 initial_lr=0.025,
                 min_count=30,
                 p = 0.0,
                 sigma = 1e-9,
                 clip = 1.0):
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

        Returns:
            None.
        """
        self.data = InputData(input_file_name, min_count)
        self.input_wvect = InputVector(input_wvectors)
        self.input_cvect = InputVector(input_cvectors)
        self.output_file_name = output_file_name
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.window_size = window_size
        self.iteration = iteration
        self.initial_lr = initial_lr
        self.p = p
        self.sigma = sigma
        self.clip = clip
        self.topfrequent = Topfreq(self.data.word_frequency)
        self.kneighbor = KNeighbor(input_wvectors, self.topfrequent, self.data.word2id, self.data.id2word)
        self.fine_tune_model = FineTuneModel(self.emb_size, self.emb_dimension, self.p, self.sigma)
        self.fine_tune_model.init_emb(self.input_wvect, self.input_cvect)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.fine_tune_model.cuda()
        self.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.fine_tune_model.parameters()), lr=self.initial_lr)

    def train(self):
        """Multiple training.

        Returns:
            None.
        """
        #pair_count = self.data.evaluate_pair_count(self.window_size)
        pro_pairs = Get_pairs(self.data, self.topfrequent, self.kneighbor, self.window_size)
        #vsp_pairs = Get_VSP(self.input_wvect, self.topfrequent, rho=self.rho)
        pair_count = len(pro_pairs)
        batch_count = self.iteration * pair_count / self.batch_size
        process_bar = tqdm(range(int(batch_count)))
        # self.skip_gram_model.save_embedding(
        #     self.data.id2word, 'begin_embedding.txt', self.use_cuda)
        tot_loss = 0
        for i in process_bar:
            #pos_pairs = self.data.get_batch_pairs(self.batch_size,
                                                  #self.window_size)
            #batch_pairs = Batch_pairs(pro_pairs, self.batch_size, i, self.iteration)
            batch_pairs = Batch_pairs(pro_pairs, self.batch_size)
            #neg_v = self.data.get_neg_v_neg_sampling(pos_pairs, 5)
            #batch_pair = batch_pairs[i % self.iteration]
            batch_u = [pair[0] for pair in batch_pairs]
            batch_n = [pair[1] for pair in batch_pairs]
            batch_v = [pair[2] for pair in batch_pairs]
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

            process_bar.set_description("Loss: %0.8f, lr: %0.6f" %
                                        (tot_loss/(i+1),
                                         self.optimizer.param_groups[0]['lr']))
            if i * self.batch_size % 100000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        self.fine_tune_model.save_embedding(
            self.data.id2word, self.output_file_name, self.use_cuda)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Philly arguments parser")

    parser.add_argument('input_file_name', type=str)
    parser.add_argument('input_wvectors', type=str)
    parser.add_argument('input_cvectors', type=str)
    parser.add_argument('output_file_name', type=str)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--iteration', type=int, default=20)
    parser.add_argument('--min_count', type=int, default=30)
    parser.add_argument('--initial_lr', type=float, default=0.025)
    args, _ = parser.parse_known_args()
    #w2v = Word2Vec(input_file_name=sys.argv[1], input_wvectors = sys.argv[2], input_cvectors = sys.argv[3], output_file_name=sys.argv[4])
    w2v = Word2Vec(input_file_name=args.input_file_name, input_wvectors=args.input_wvectors, input_cvectors = args.input_cvectors, output_file_name=args.output_file_name,
        batch_size=args.batch_size, window_size=args.window_size, iteration=args.iteration, min_count=args.min_count,
        initial_lr=args.initial_lr)
    w2v.train()

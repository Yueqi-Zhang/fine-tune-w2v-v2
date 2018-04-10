import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class FineTuneModel(nn.Module):
    """Skip gram model of word2vec.

    Attributes:
        emb_size: Embedding size.
        emb_dimention: Embedding dimention, typically from 50 to 500.
        u_embedding: Embedding for center word.
        v_embedding: Embedding for neibor words.
    """

    def __init__(self, emb_size, emb_dimension, p, sigma, wvector, cvector):
        """Initialize model parameters.

        Apply for two embedding layers.
        Initialize layer weight

        Args:
            emb_size: Embedding size.
            emb_dimention: Embedding dimention, typically from 50 to 500.

        Returns:
            None
        """
        super(FineTuneModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.p = p
        self.sigma = sigma
        self.i_embeddings = nn.Embedding(emb_size, emb_dimension) # unchange pretrain embedding
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension) # fine-tune embedding
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension) # context embedding

        # to init emb
        #initrange = 0.5 / self.emb_dimension
        self.i_embeddings.weight = nn.Parameter(torch.Tensor(wvector))
        self.i_embeddings.weight.requires_grad = False
        self.u_embeddings.weight = nn.Parameter(torch.Tensor(wvector))
        self.v_embeddings.weight = nn.Parameter(torch.Tensor(cvector))
        #self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        #self.v_embeddings.weight.data.uniform_(-0, 0)


    def forward(self, batch_u, batch_n, batch_v_pad, batch_v_mask, batch_vsp):
        """Forward process.

        As pytorch designed, all variables must be batch format, so all input of this method is a list of word id.

        Args:
            batch_u: list of center word ids for positive word pairs.
            batch_n: list of neibor word ids for positive word pairs.
            batch_v: list of context word ids for word pairs.
            batch_v_mask: [bs, 2*window_size]
            batch_vsp: all words in pair

        Returns:
            Loss of this process, a pytorch variable.
        """
        #diff = self.u_embeddings.weight - self.i_embeddings.weight
        emb_u = self.u_embeddings(batch_u) # center word  [bs, emb_dim]
        emb_n = self.u_embeddings(batch_n) # neighbor word
        emb_v = self.v_embeddings(batch_v_pad) # context word [bs, 2*window_size, emb_dim]
        batch_v_mask = torch.unsqueeze(batch_v_mask, 2).expand(batch_v_mask.size()[0], batch_v_mask.size()[1], self.emb_dimension)
        emb_v = torch.sum(torch.mul(emb_v, batch_v_mask), dim = 1) # dim: bs*emb_dim
        score_c = torch.mul(emb_u, emb_v).squeeze()
        score_c = torch.sum(score_c, dim = 1)
        score_c = F.logsigmoid(score_c)
        score_n = torch.mul(emb_n, emb_v).squeeze()
        score_n = torch.sum(score_n, dim = 1)
        score_n = F.logsigmoid(score_n)
        score1 = torch.sum(F.relu(self.p+score_n-score_c))
        emb_vsp_o = self.i_embeddings(batch_vsp)  # [bs*(2+2*window_size), emb_dim]
        #batch_vsp_mask = torch.unsqueeze(batch_vsp_mask, 2).expand(batch_vsp_mask.size()[0], batch_vsp_mask.size()[1],
        #                                                       self.emb_dimension)
        #emb_vsp_o = torch.mul(emb_vsp_o, batch_vsp_mask)
        emb_vsp_n = self.u_embeddings(batch_vsp)
        #emb_vsp_n = torch.mul(emb_vsp_n, batch_vsp_mask)
        emb_vsp_diff = emb_vsp_n-emb_vsp_o # self.u_embeddings.weight-self.i_embeddings.weight
        score_vsp = self.sigma*torch.sum(torch.mul(emb_vsp_diff, emb_vsp_diff).squeeze())
        return score1+score_vsp




def test():
    model = SkipGramModel(100, 100)
    id2word = dict()
    for i in range(100):
        id2word[i] = str(i)
    model.save_embedding(id2word)


if __name__ == '__main__':
    test()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch import Tensor
import math


class PrtLayer(nn.Module):
    def __init__(self, input_dim, hadden_dim, fc_dim):
        super(PrtLayer, self).__init__()
        self.input_dim = input_dim
        self.hadden_dim = hadden_dim
        self.fc_dim = fc_dim

        init_hx = torch.zeros(1, self.hadden_dim)
        init_hx = init_hx.cuda()
        self.init_hx = Variable(init_hx, requires_grad=False)

    def forward(self, x, n_src = None, n_tgt = None):
        # assert x.shape[1] == x.shape[2]

        batch_size = x.shape[0]
        n_nodes = x.shape[1]
        input_size = x.shape[2]
        assert input_size == self.input_dim

        # # take outer product, result is [batch_size, N, N]
        # x = torch.bmm(g2, torch.transpose(g1, 2, 1))


        self.gru = nn.GRU(self.input_dim, self.hadden_dim).cuda()
        # self.fc0 = nn.Linear(n_nodes, self.hadden_dim).cuda()
        self.fc1 = nn.Linear(self.hadden_dim, self.fc_dim).cuda()

        x = torch.transpose(x, 0, 1)
        init_hx = self.init_hx.unsqueeze(1).repeat(1, batch_size, 1)
        h, _ = self.gru(x, init_hx)
        # h is [n_nodes, batch_size, rnn_dim]
        h = torch.transpose(h, 0, 1)
        # h = self.fc0(x)
        # result h is [batch_size, n_nodes, n_nodes]
        h_new = self.fc1(h)
        return F.relu(h_new)

class Siamese_Banch(nn.Module):
    """
    Perform graph convolution on two input graphs (g1, g2)
    """
    def __init__(self, input_dim, hadden_dim, fc_dim):
        super(Siamese_Banch, self).__init__()
        self.prt_net = PrtLayer(input_dim, hadden_dim, fc_dim)

        # self.lam = Parameter(Tensor(fc_dim, fc_dim))

        # stdv = 1. / math.sqrt(fc_dim)
        # self.lam.data.uniform_(-stdv, stdv)
        # self.lam.data += torch.eye(fc_dim)

    def forward(self, g1, g2):
        out1 = self.prt_net(*g1)
        out2 = self.prt_net(*g2)
        # embx are tensors of size (bs, N, num_features)
        # s = torch.matmul(out1, out2.transpose(1, 2))
        # M = torch.matmul(out1, (self.lam + self.lam.transpose(0, 1)) / 2)
        # M = torch.matmul(out1, out2.transpose(1, 2))
        return out1, out2



class MergeLayers(nn.Module):

    def __init__(self, hadden_dim):
        super(MergeLayers, self).__init__()
        self.hadden_dim = hadden_dim


    def forward(self, X, n_src = None, n_tgt = None):
        assert torch.all(X >= 0.)
        batch_size = X.shape[0]
        n_nodes = X.shape[1]

        # self.W1 = Parameter(Tensor(n_nodes, n_nodes)).cuda()
        # stdv = 1. / math.sqrt(self.hadden_dim)
        # self.W1.data.uniform_(-stdv, stdv)
        # self.W1.data += torch.ones(n_nodes).cuda()
        w_shape = Tensor(batch_size, n_nodes, n_nodes).cuda()
        self.W1 = Parameter(torch.ones_like(w_shape))
        stdv = 1. / math.sqrt(self.hadden_dim)
        self.W1.data += torch.zeros_like(w_shape).uniform_(-stdv, stdv)
        #M = torch.matmul(X, self.A)
        h = torch.mul(X, (self.W1 + self.W1.transpose(1, 2)) / 2)
        
        h = torch.mul(h, X)

        # return F.leaky_relu(h)
        return h


    # def forward(self, x, n_src = None, n_tgt = None):
    #     assert torch.all(torch.eq(n_src, n_tgt))
    #     batch_size = x.shape[0]

    #     for b in range(batch_size):
    #         n_nodes = n_src[b]
    #         self.merge_1 = nn.Linear(n_nodes, self.hadden_dim)
    #         self.merge_2 = nn.Linear(self.hadden_dim, n_nodes)

    #         x[b] = F.leaky_relu(self.merge_1(x[b]))
    #         x[b] = F.leaky_relu(self.merge_2(x[b]))
    #     s = x
    #     return s

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
import math

def to_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x

class SequencePro(nn.Module):
    def __init__(self, alpha = 20, is_GRU=False):
        super().__init__()
        self.alpha = alpha
        self.is_GRU = is_GRU
        self.softmax = nn.Softmax(dim=-1)
        self.pro_model = CellPro(self.is_GRU)

    def forward(self, input):
        batch_size = input.shape[0]
        decoder_seq_len = input.shape[1]

        probs = []
        input_ = input.transpose(0, 1)
        for i in range(decoder_seq_len):  
            input_i = input_[i]

            hidden = self.pro_model(input_i)

            out = self.softmax(self.alpha * hidden)
            probs.append(out)

        probs = torch.stack(probs, dim=1)           
        return probs

class CellPro(nn.Module):
    def __init__(self, is_GRU):
        super(CellPro, self).__init__()

        self.is_GRU = is_GRU

        if self.is_GRU:
            self.RNN = nn.GRU
            self.RNNCell = nn.GRUCell
        else:
            self.RNN = nn.LSTM
            self.RNNCell = nn.LSTMCell

    def forward(self, input_i):
        batch_size = input_i.shape[0]
        input_size = input_i.shape[1]
        out_size = input_size

        self.decoder = self.RNNCell(input_size, out_size, bias=False).cuda()

        if self.is_GRU:
            mutiplier = 3
        else:
            mutiplier = 4
        self.w_ih = Parameter(Tensor(mutiplier*input_size, input_size)).cuda()
        self.w_hh = Parameter(Tensor(mutiplier*input_size, input_size)).cuda()
        # self.b_ih = torch.zeros(mutiplier*input_size).cuda()
        # self.b_hh = torch.zeros(mutiplier*input_size).cuda()

        stdv = 1. / math.sqrt(1024)
        self.w_ih.data.uniform_(-stdv, stdv)
        self.w_ih.data += torch.eye(input_size).repeat(mutiplier, 1).cuda()
        self.w_hh.data.uniform_(-stdv, stdv)
        self.w_hh.data += torch.eye(input_size).repeat(mutiplier, 1).cuda()
        
        self.decoder.weight_ih = Parameter(self.w_ih).cuda()
        self.decoder.weight_hh = Parameter(self.w_hh).cuda()
        # self.decoder.bias_ih = Parameter(self.b_ih).cuda()
        # self.decoder.bias_hh = Parameter(self.b_hh).cuda()

        init_h = torch.zeros(batch_size, input_size).cuda()
        init_c = torch.zeros(batch_size, input_size).cuda()
        hidden_init = Variable(init_h, requires_grad=False)
        cell_init = Variable(init_c, requires_grad=False)

        if self.is_GRU:
            hidden = self.decoder(input_i, hidden) 
        else:
            hidden, decoder_hc = self.decoder(input_i, (hidden_init, cell_init))

        return hidden

class FullPro(nn.Module):
    def __init__(self, alpha=200, pixel_thresh=None):
        super(FullPro, self).__init__()
        # self.batch_size = batch_size
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=-1)
        self.pixel_thresh = pixel_thresh

    def forward(self, s, nrow_gt, ncol_gt=None):
        # batch_size = s.shape[0]
        # n_nodes = s.shape[1]

        # w_shape = Tensor(batch_size, n_nodes, n_nodes).cuda()
        self.W1 = Parameter(torch.ones_like(s))
        stdv = 1. / math.sqrt(1024)
        self.W1.data += torch.zeros_like(s).uniform_(-stdv, stdv)
        # s = torch.mul(self.W1, s)

        ret_s = torch.zeros_like(s)
        # filter dummy nodes
        for b, n in enumerate(nrow_gt):
            if ncol_gt is None:
                ret_s[b, 0:n, :] = \
                    self.softmax(torch.mul(self.W1[b, 0:n, :], self.alpha * s[b, 0:n, :]))
                    # self.softmax(self.alpha * s[b, 0:n, :])
            else:
                ret_s[b, 0:n, 0:ncol_gt[b]] =\
                    self.softmax(torch.mul(self.W1[b, 0:n, 0:ncol_gt[b]], self.alpha * s[b, 0:n, 0:ncol_gt[b]]))
                    # self.softmax(self.alpha * s[b, 0:n, 0:ncol_gt[b]])

        return ret_s

class PrtLayer(nn.Module):
    def __init__(self, hadden_dim):
        super(PrtLayer, self).__init__()
        self.hadden_dim = hadden_dim

        init_hx = torch.zeros(1, self.hadden_dim)
        init_hx = init_hx.cuda()
        self.init_hx = Variable(init_hx, requires_grad=False)

    def forward(self, x, n_src = None, n_tgt = None):
        assert x.shape[1] == x.shape[2]

        batch_size = x.shape[0]
        n_nodes = x.shape[1]

        self.gru = nn.GRU(n_nodes, self.hadden_dim).cuda()
        self.fc0 = nn.Linear(n_nodes, self.hadden_dim).cuda()
        self.fc1 = nn.Linear(self.hadden_dim, n_nodes).cuda()

        x = torch.transpose(x, 0, 1)
        init_hx = self.init_hx.unsqueeze(1).repeat(1, batch_size, 1)
        h, _ = self.gru(x, init_hx)
        # h is [n_nodes, batch_size, rnn_dim]
        h = torch.transpose(h, 0, 1)
        # h = self.fc0(x)
        # result M is [batch_size, n_nodes, n_nodes]
        h = self.fc1(h)
        return F.leaky_relu(h)
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.autograd import Variable
import math

eps = 1e-6

def compute_cosine(x, y):
    # x [....,n1, 1, d]
    # y [....,1, n2, d]
    cosin_numerator = torch.sum(torch.mul(x, y), dim=-1)
    x_norm = torch.sqrt(torch.clamp_min(torch.sum(x ** 2, dim=-1), eps))
    y_norm = torch.sqrt(torch.clamp_min(torch.sum(y ** 2, dim=-1), eps))
    return cosin_numerator / x_norm / y_norm

def cal_similarity_matrix(em1, em2):

    # [batch_size, n1, 1, dims]
    em1_tmp = torch.unsqueeze(em1, 2)
    # [batch_size, 1, n2, dims]
    em2_tmp = torch.unsqueeze(em2, 1)
    # [batch_size, n1, n2]
    s_matrix = compute_cosine(em1_tmp, em2_tmp)
    
    return s_matrix

class Cosine_Affinity(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    """
    def __init__(self, d):
        super(Cosine_Affinity, self).__init__()
        # self.batch_size = batch_size
        self.d = d
        # self.A_1 = Parameter(Tensor(self.batch_size, self.d, self.d))
        # self.A_2 = Parameter(Tensor(self.batch_size, self.d, self.d))
        # self.reset_parameters()

    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.d)
    #     self.A_1.data.uniform_(-stdv, stdv)
    #     self.A_1.data += torch.ones(self.batch_size, self.d, self.d)

    #     self.A_2.data.uniform_(-stdv, stdv)
    #     self.A_2.data += torch.ones(self.batch_size, self.d, self.d)


    def forward(self, X, Y):
        batch_size = X.shape[0]
        assert X.shape[2] == Y.shape[2] == self.d
        self.A_1 = Parameter(torch.ones_like(X))
        self.A_2 = Parameter(torch.ones_like(Y))
        stdv = 1. / math.sqrt(self.d)
        self.A_1.data += torch.zeros_like(X).uniform_(-stdv, stdv)
        self.A_2.data += torch.zeros_like(Y).uniform_(-stdv, stdv)

        X_ = torch.mul(self.A_1, X)
        Y_ = torch.mul(self.A_2, Y)
        M = cal_similarity_matrix(X_, Y_)
        return M
        # return torch.exp(M)

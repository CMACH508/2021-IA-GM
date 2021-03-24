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

def cal_attention_embedding(x, m):
    rows_sum = torch.sum(m, dim=-1)
    rows_sum = torch.unsqueeze(rows_sum, 2)
    y = torch.div(torch.matmul(m, x), rows_sum)
    return y

def verify_cosine(em1, em2):
    s = torch.zeros(em1.size(0), em1.size(1), em2.size(1))
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    for batch in range(em1.size(0)):
        for row in range(em1.size(1)):
            for col in range(em2.size(1)):
                s[batch, row, col] = cos(em1[batch][row], em2[batch][col])

    return s


class Attention_Layer(nn.Module):
    def __init__(self, state_dim, out_dim):
        super(Attention_Layer, self).__init__()
        self.state_dim = state_dim
        # self.hadden_dim = hadden_dim
        self.out_dim = out_dim

        self.softmax = torch.nn.Softmax(dim=-1)

        # init_hx = torch.zeros(1, self.hadden_dim)
        # init_hx = init_hx.cuda()
        # self.init_hx = Variable(init_hx, requires_grad=False)
        self.fc = nn.Linear(self.state_dim, self.out_dim)
        self.relu = nn.ReLU()

    def forward(self, em1, em2, n_src = None, n_tgt = None):
        atten_matrix = cal_similarity_matrix(em1, em2)
        soft_matrix = self.softmax(atten_matrix)
        x = torch.matmul(soft_matrix, em2)
        # x = cal_attention_embedding(em2, soft_matrix)
        emb = torch.cat((em1, x), dim=-1)
        # batch_size = x.shape[0]

        # self.gru = nn.GRU(self.state_dim, self.hadden_dim).cuda()
        # # self.fc0 = nn.Linear(n_nodes, self.hadden_dim).cuda()
        # self.fc1 = nn.Linear(self.hadden_dim, self.out_dim).cuda()
        # self.relu = nn.ReLU()

        # x = torch.transpose(x, 0, 1)
        # init_hx = self.init_hx.unsqueeze(1).repeat(1, batch_size, 1)
        # h, _ = self.gru(x, init_hx)
        # # h is [n_nodes, batch_size, rnn_dim]
        # h = torch.transpose(h, 0, 1)
        # # h = self.fc0(x)
        # # result M is [batch_size, n_nodes, n_nodes]
        # h = self.fc1(h)
        h = self.relu(self.fc(emb))
        return h

# class Siamese_Attention(nn.Module):
    
#     def __init__(self, state_dim, hadden_dim, out_dim):
#         super(Siamese_Attention, self).__init__()
#         self.attention_layer = Attention_Layer(state_dim, hadden_dim, out_dim)

#     def forward(self, g1, g2):
#         emb1 = self.attention_layer(*g1)
#         emb2 = self.attention_layer(*g2)
#         # embx are tensors of size (bs, N, num_features)
#         return emb1, emb2

# if __name__ == '__main__':

#     a = torch.randn(2,3,5)
#     b = torch.randn(2,4,5)

#     s0 = cal_similarity_matrix(a, b)
#     s1 = verify_cosine(a, b)

#     print("s0 shape: ",s0.shape)
#     print("s1 shape: ",s1.shape)
#     print(s0.min(), s0.max())
#     print(s1.min(), s1.max())
#     print(torch.sum(s0, dim=2))
#     print(s1)

#     for batch in range(s0.size(0)):
#         for row in range(s0.size(1)):
#             for col in range(s0.size(2)):
#                 if (s0[batch, row, col]-s1[batch, row, col]) != 0:
#                     print(s0[batch, row, col]-s1[batch, row, col])

#     print(torch.all((s0-s1) == 0))
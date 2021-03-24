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

class SequenceLayer(nn.Module):
    def __init__(self, hidden_size, alpha = 20, is_GRU=False):
        super().__init__()
        # self.input_size = input_size
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.is_GRU = is_GRU

        if self.is_GRU:
            self.RNN = nn.GRU
            self.RNNCell = nn.GRUCell
        else:
            self.RNN = nn.LSTM
            self.RNNCell = nn.LSTMCell

    def forward(self, input):
        batch_size = input.shape[0]
        decoder_seq_len = input.shape[1]
        input_size = decoder_seq_len

        self.encoder = self.RNN(input_size, decoder_seq_len, batch_first=True).cuda()
        self.decoder = self.RNNCell(input_size, decoder_seq_len, bias=False).cuda()

        if self.is_GRU:
            mutiplier = 3
        else:
            mutiplier = 4
        self.w_ih = Parameter(Tensor(mutiplier*input_size, input_size)).cuda()
        self.w_hh = Parameter(Tensor(mutiplier*input_size, input_size)).cuda()
        # self.b_ih = torch.zeros(mutiplier*input_size).cuda()
        # self.b_hh = torch.zeros(mutiplier*input_size).cuda()

        stdv = 1. / math.sqrt(2048)
        self.w_ih.data.uniform_(-stdv, stdv)
        self.w_ih.data += torch.eye(input_size).repeat(mutiplier, 1).cuda()
        self.w_hh.data.uniform_(-stdv, stdv)
        self.w_hh.data += torch.eye(input_size).repeat(mutiplier, 1).cuda()
        
        self.decoder.weight_ih = Parameter(self.w_ih).cuda()
        self.decoder.weight_hh = Parameter(self.w_hh).cuda()
        # self.decoder.bias_ih = Parameter(self.b_ih).cuda()
        # self.decoder.bias_hh = Parameter(self.b_hh).cuda()


        encoder_output, hc = self.encoder(input)

        # Decoding states initialization
        # hidden = encoder_output[:, -1, :] #hidden state for decoder is the last timestep's output of encoder 
        # if not self.is_GRU: #For LSTM, cell state is the sencond state output
        #     cell = hc[1][-1, :, :]
        
        init_h = torch.zeros(batch_size, input_size).cuda()
        init_c = torch.zeros(batch_size, input_size).cuda()
        hidden_init = Variable(init_h, requires_grad=False)
        cell_init = Variable(init_c, requires_grad=False)

        probs = []
        input_ = input.transpose(0, 1)
        for i in range(decoder_seq_len):  
            decoder_input = input_[i]
            if self.is_GRU:
                hidden = self.decoder(decoder_input, hidden) 
            else:
                hidden, decoder_hc = self.decoder(decoder_input, (hidden_init, cell_init))

            out = F.softmax(self.alpha * hidden, -1)
            probs.append(out)

        probs = torch.stack(probs, dim=1)           
        return probs

class PointerLayer(nn.Module):
    def __init__(self, hidden_size, weight_size, is_GRU=False, is_encoder = True):
        super(PointerLayer, self).__init__()
        # self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_size = weight_size
        self.is_GRU = is_GRU

        if self.is_GRU:
            self.RNN = nn.GRU
            self.RNNCell = nn.GRUCell
        else:
            self.RNN = nn.LSTM
            self.RNNCell = nn.LSTMCell
        
        self.W1 = nn.Linear(hidden_size, weight_size, bias=False) 
        self.W2 = nn.Linear(hidden_size, weight_size, bias=False) 
        self.vt = nn.Linear(weight_size, 1, bias=False)

    def forward(self, input):
        batch_size = input.shape[0]
        decoder_seq_len = input.shape[1]
        input_size = decoder_seq_len

        self.encoder = self.RNN(input_size, self.hidden_size, batch_first=True).cuda()
        self.decoder = self.RNNCell(input_size, self.hidden_size).cuda()

        encoder_output, hc = self.encoder(input) 

        # Decoding states initialization
        hidden = encoder_output[:, -1, :] #hidden state for decoder is the last timestep's output of encoder 
        if not self.is_GRU: #For LSTM, cell state is the sencond state output
            cell = hc[1][-1, :, :]
        # decoder_input = to_cuda(torch.rand(batch_size, self.input_size))  
        
        # print("encoder_output shape", encoder_output.shape)
        # print("h_n shape", hc[0].shape)
        # print("c_n shape", hc[1].shape)
        # print("cell shape", cell.shape)
        # print("decoder_input shape", decoder_input.shape)
        # Decoding with attention             
        probs = []
        encoder_output = encoder_output.transpose(1, 0) #Transpose the matrix for mm
        input_ = input.transpose(0, 1)
        for i in range(decoder_seq_len):  
            decoder_input = input_[i]
            if self.is_GRU:
                hidden = self.decoder(decoder_input, hidden) 
            else:
                hidden, decoder_hc = self.decoder(decoder_input, (hidden, cell)) 
            # Compute attention
            sum = torch.tanh(self.W1(encoder_output) + self.W2(hidden))    
            # print("sum shape", sum.shape) [10, 250, 256]
            out = self.vt(sum).squeeze()  
            # print("out_0 shape", out.shape)      (decoder_seq_len, batch_size)
            out = F.log_softmax(out.transpose(0, 1).contiguous(), -1) 
            # print("out_1 shape", out.shape) (batch_size, decoder_seq_len)
            probs.append(out)

        probs = torch.stack(probs, dim=1)           
        return probs

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
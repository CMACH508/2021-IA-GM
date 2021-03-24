import torch
import torch.nn as nn

from utils.sinkhorn import Sinkhorn
from utils.hungarian import hungarian
from utils.voting_layer import Voting
from utils.feature_align import feature_align
from BIIA.gcns import Siamese_Net
from BIIA.merge_layers import MergeLayers
from BIIA.affinity_layer import Affinity
from BIIA.cosine_affinity import Cosine_Affinity
import torch.nn.functional as F
from BIIA.probability_layer import SequencePro, FullPro

from utils.config import cfg

import utils.backbone
CNN = eval('utils.backbone.{}'.format(cfg.BACKBONE))

class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()

        self.gnn_layers = 1
        self.iteration_ = 4
        self.a1 = 0.5
        self.a2 = 1.5

        # resnet
        self.R = nn.Linear(cfg.BIIA.FEATURE_CHANNEL, cfg.BIIA.FEATURE_CHANNEL, bias=False)
        self.relu = nn.ReLU()

        self.merge_layer = MergeLayers(cfg.BIIA.FEATURE_CHANNEL).cuda()
        self.bi_stochastic = Sinkhorn(max_iter=cfg.BIIA.BS_ITER_NUM, epsilon=cfg.BIIA.BS_EPSILON)
        self.voting_layer_0 = FullPro(alpha=cfg.BIIA.VOTING_ALPHA)
        self.voting_layer_1 = FullPro(alpha=(cfg.BIIA.VOTING_ALPHA/2))
        self.hung = hungarian
        self.l2norm = nn.LocalResponseNorm(cfg.BIIA.FEATURE_CHANNEL, alpha=cfg.BIIA.FEATURE_CHANNEL, beta=0.5, k=0)
        self.layernorm = nn.LayerNorm(torch.randn(8, 30, cfg.BIIA.FEATURE_CHANNEL).size()[1:])

        for i in range(self.gnn_layers):
            ggnn = Siamese_Net(cfg.BIIA.FEATURE_CHANNEL)
            self.add_module('ggnn_{}'.format(i), ggnn)

        for k in range(self.iteration_):
            self.add_module('affinity_{}'.format(k), Affinity(cfg.BIIA.FEATURE_CHANNEL))

    def forward(self, fea_src, fea_tgt, A_src, A_tgt, ns_src, ns_tgt, inp_type):

        # fea_src = self.layernorm(fea_src)
        # fea_tgt = self.layernorm(fea_tgt)
        emb1, emb2 = fea_src, fea_tgt

        # adjacency matrices
        A_src = A_src.repeat(1, 1, 2)
        A_tgt = A_tgt.repeat(1, 1, 2)

        emb1_init, emb2_init = emb1, emb2

        for i in range(self.gnn_layers):
            ggnn_layer = getattr(self, 'ggnn_{}'.format(i))
            emb1, emb2 = ggnn_layer([emb1, A_src], [emb2, A_tgt])
            # emb1 = emb1_new
            # emb2 = emb1_new
        emb1 = self.relu(emb1 + self.R(emb1_init))
        emb2 = self.relu(emb2 + self.R(emb2_init))

        for k in range(self.iteration_):
            affinity = getattr(self, 'affinity_{}'.format(k))
            if k > 0:
                s = self.a1*affinity(emb1, emb2) + self.a2*affinity(torch.bmm(p, emb2), torch.bmm(p.transpose(1, 2), emb1))
                s = self.voting_layer_1(s, ns_src, ns_tgt)
            else:
                s = affinity(emb1, emb2)
                s = self.voting_layer_0(s, ns_src, ns_tgt)

            pro = s
            s = self.bi_stochastic(s, ns_src, ns_tgt)
            dsm = s
            p = self.hung(s, ns_src, ns_tgt)
            # self.a1 -= 0.25
            # self.a2 += 0.25

        return pro, dsm, p

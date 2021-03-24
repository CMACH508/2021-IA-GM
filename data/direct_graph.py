import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
import math
import random

def ToDirectGraph(A, rate = 0.5):
    A_ = A
    batch_szie = A.size(0)
    rows = A.size(1)
    cols = A.size(2)
    for b in range(batch_szie):
        for i in range(rows):
            for j in range(cols):
                ran_num = random.random()
                if (ran_num < rate):
                    A_[b, i, j] = 0
    return A_

def BuildDirect(A, rate = 0.5):
    A_ = A
    batch_size = A.size(0)
    rows = A.size(1)
    cols = A.size(2)

    for b in range(batch_size):
        edges_nums = torch.sum(A[b])
        remained = int(rate * edges_nums) + 1
        to_del = edges_nums - remained
        k, i, j = 0, 0, 0
        while(k < to_del):
            if (j >= cols):
                j = 0
                i += 1
            if (i >= rows):
                i = 0
                j = 0
            # print(i, j)
            if A_[b, i, j] == 0:
                j += 1
                continue
            ran_num = random.uniform(0, 1)
            if (ran_num < 0.5 and A_[b, i, j] == 1):
                A_[b, i, j] = 0
                k += 1
            j += 1
        assert torch.sum(A_[b]) == remained
            
    return A_

if __name__ == '__main__':
    a = torch.ones((3,6,6))
    a_ = BuildDirect(a, 0.3)
    print(a_)
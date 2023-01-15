import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.4, act=nn.ReLU):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.act(x)
        return x


class CrossCompressUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.vv = nn.Linear(dim, 1)
        self.ve = nn.Linear(dim, 1)
        self.ev = nn.Linear(dim, 1)
        self.ee = nn.Linear(dim, 1)

    def forward(self, v, e):
        v = v.unsqueeze(2)
        e = e.unsqueeze(1)

        c_matrix = torch.mm(v, e)
        c_matrix_transpose = c_matrix.T

        c_matrix = c_matrix.view(-1, self.dim)
        c_matrix_transpose = c_matrix_transpose.view(-1, self.dim)

        v_output = self.vv(c_matrix) + self.ev(c_matrix_transpose)
        e_output = self.ve(c_matrix) + self.ee(c_matrix_transpose)
        return v_output.view(-1, self.dim), e_output.view(-1, self.dim)


class MRK(nn.Module):
    def __init__(self, dim, user_num, item_num, relation_num, so_num):
        super().__init__()

        self.user_embedding = nn.Embedding(user_num, dim)
        self.item_embedding = nn.Embedding(item_num, dim)
        self.relation_embedding = nn.Embedding(relation_num, dim)
        self.so_embedding = nn.Embedding(so_num, dim)

        self.user_mlp = MLP(dim, dim)
        self.tail_mlp = MLP(dim, dim)
        self.cc_unit = CrossCompressUnit(dim)

    def forward(self, user_ids, item_ids, relation_ids, s_ids, o_ids, flag):
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)


class LowLayer(nn.Module):
    def __init__(self):
        super().__init__()





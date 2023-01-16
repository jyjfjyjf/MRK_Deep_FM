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

    def get_weights(self):
        return [self.linear.weight]


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

    def get_weights(self):
        return [self.vv.weight, self.ve.weight, self.ev.weight, self.ee.weight]


class MRK(nn.Module):
    def __init__(self, dim, user_num, item_num, relation_num, so_num, low_layer_num, high_layer_num):
        super().__init__()

        self.user_embedding = nn.Embedding(user_num, dim)
        self.item_embedding = nn.Embedding(item_num, dim)
        self.relation_embedding = nn.Embedding(relation_num, dim)
        self.so_embedding = nn.Embedding(so_num, dim)
        self.high_layer_num = high_layer_num

        self.low_layer = LowLayer(dim, low_layer_num)

    def forward(self, user_ids, item_ids, relation_ids, s_ids, o_ids, flag):
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
        relation_embedding = self.relation_embedding(relation_ids)
        head_embedding = self.so_embedding(s_ids)
        tail_embedding = self.so_embedding(o_ids)

        user_embedding, item_embedding, tail_embedding = self.low_layer(
            user_embedding, item_embedding, head_embedding, tail_embedding
        )

        scores = (user_embedding * item_embedding).sum(1)
        sigmoid = nn.Sigmoid()
        scores_normalized = sigmoid(scores)


        for _ in range(self.high_layer_num):


class HighLayer(nn.Module):
    def __init__(self, dim, layer_num):
        super().__init__()
        self.layer_num = layer_num
        self.kg_mlp = MLP(2 * dim, 2 * dim)
        self.kg_pred_mlp = MLP(2 * dim, dim)

    def forward(self, head_embedding, relation_embedding):
        head_relation_concat = torch.cat([head_embedding, relation_embedding], dim=1)

        kg_mlp_weight = []
        for _ in range(self.layer_num - 1):
            head_relation_concat = self.kg_mlp(head_relation_concat)
            kg_mlp_weight.append(self.kg_mlp.get_weights())

        tail_pred = self.kg_pred_mlp(head_relation_concat)
        kg_mlp_weight.append(self.kg_pred_mlp.get_weights())
        sigmoid = nn.Sigmoid()
        tail_pred = sigmoid(tail_pred)



class LowLayer(nn.Module):
    def __init__(self, dim, layer_num):
        super().__init__()
        self.user_mlp = MLP(dim, dim)
        self.tail_mlp = MLP(dim, dim)
        self.cc_unit = CrossCompressUnit(dim)
        self.layer_num = layer_num

    def forward(self, user_embedding, item_embedding, head_embedding, tail_embedding):
        for _ in range(self.layer_num):
            user_embedding = self.user_mlp(user_embedding)
            item_embedding, head_embedding = self.cc_unit(item_embedding, head_embedding)
            tail_embedding = self.tail_mlp(tail_embedding)

        return user_embedding, item_embedding, tail_embedding

    def get_weights(self):
        return self.user_mlp.get_weights() + self.tail_mlp.get_weights() + self.cc_unit.get_weights()





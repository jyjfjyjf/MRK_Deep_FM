import torch
from torch import nn

from transformers.models.bert.modeling_bert import BertModel

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2, act=nn.ReLU()):
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
        self.vv = nn.Linear(dim, 1, bias=False)
        self.ve = nn.Linear(dim, 1, bias=False)
        self.ev = nn.Linear(dim, 1, bias=False)
        self.ee = nn.Linear(dim, 1, bias=False)

        self.bias_v = nn.Parameter(torch.zeros(dim))
        self.bias_e = nn.Parameter(torch.zeros(dim))

    def forward(self, v, e):
        v = v.unsqueeze(2)
        e = e.unsqueeze(1)

        c_matrix = torch.bmm(v, e)
        c_matrix_transpose = c_matrix.transpose(2, 1)

        c_matrix = c_matrix.view(-1, self.dim)
        c_matrix_transpose = c_matrix_transpose.reshape(-1, self.dim)

        v_output = self.vv(c_matrix) + self.ev(c_matrix_transpose)
        e_output = self.ve(c_matrix) + self.ee(c_matrix_transpose)
        return v_output.view(-1, self.dim) + self.bias_v, e_output.view(-1, self.dim) + self.bias_e

    def get_weights(self):
        return [self.vv.weight, self.ve.weight, self.ev.weight, self.ee.weight]


class MRK(nn.Module):
    def __init__(self, dim, user_num, item_num, relation_num, so_num, low_layer_num, high_layer_num):
        super().__init__()
        self.dim = dim
        self.user_embedding = nn.Embedding(user_num, dim)
        self.item_embedding = nn.Embedding(item_num, dim)
        self.relation_embedding = nn.Embedding(relation_num, dim)
        self.so_embedding = nn.Embedding(so_num, dim)

        self.user_mlp = MLP(dim, dim)
        self.tail_mlp = MLP(dim, dim)
        self.cc_unit = CrossCompressUnit(dim)
        self.layer_num = low_layer_num

        self.layer_num = high_layer_num
        self.kg_mlp = MLP(2 * dim, 2 * dim)
        self.kg_pred_mlp = MLP(2 * dim, dim)

        self.apply(self.init_weight)

    def init_weight(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)

    def forward(self,
                user_ids=None,
                item_ids=None,
                relation_ids=None,
                s_ids=None,
                o_ids=None,
                flag=True,
                labels=None):
        item_embedding = self.item_embedding(item_ids)
        head_embedding = self.so_embedding(s_ids)
        item_embeddings, head_embeddings = self.cc_unit(item_embedding, head_embedding)

        if flag:
            user_embedding = self.user_embedding(user_ids)
            user_embedding = self.user_mlp(user_embedding)
            scores = (user_embedding * item_embeddings).sum(1)
            score_normalized = torch.sigmoid(scores)
            if labels is not None:
                ce_loss_fn = nn.BCEWithLogitsLoss()
                rs_loss = ce_loss_fn(scores.view(-1, 1), labels.to(torch.float).view(-1, 1))
                rs_l2_loss = (user_embedding ** 2).sum() / 2 + (item_embeddings ** 2).sum() / 2
                for w in self.user_mlp.get_weights() + self.cc_unit.get_weights():
                    rs_l2_loss += (w ** 2).sum() / 2

                return scores, score_normalized, rs_l2_loss * 1e-6 + rs_loss
        else:
            relation_embedding = self.relation_embedding(relation_ids)
            tail_embedding = self.so_embedding(o_ids)
            tail_embedding = self.tail_mlp(tail_embedding)
            head_relation_concat = torch.cat([head_embeddings, relation_embedding], dim=1)
            head_relation_concat = self.kg_mlp(head_relation_concat)
            tail_pred = self.kg_pred_mlp(head_relation_concat)
            tail_pred = torch.sigmoid(tail_pred)
            score_kg = torch.sigmoid((tail_embedding * tail_pred).sum(1))
            rmse = torch.sqrt(
                ((tail_embedding - tail_pred) ** 2).sum(1) / self.dim
            ).mean()
            kg_loss = -score_kg
            kg_l2_loss = (head_embedding ** 2).sum() / 2 + (tail_embedding ** 2).sum() / 2
            for w in self.tail_mlp.get_weights() + self.cc_unit.get_weights() + self.kg_mlp.get_weights() + self.kg_pred_mlp.get_weights():
                kg_l2_loss += (w ** 2).sum() / 2

            return kg_loss + kg_l2_loss * 1e-6
            # return rmse


class HighLayer(nn.Module):
    def __init__(self, dim, layer_num):
        super().__init__()
        self.dim = dim
        self.layer_num = layer_num
        self.kg_mlp = MLP(2 * dim, 2 * dim)
        self.kg_pred_mlp = MLP(2 * dim, dim)

    def forward(self, head_embedding, relation_embedding, tail_embedding):
        head_relation_concat = torch.cat([head_embedding, relation_embedding], dim=1)

        kg_mlp_weight = []
        for _ in range(self.layer_num - 1):
            head_relation_concat = self.kg_mlp(head_relation_concat)
            kg_mlp_weight.append(self.kg_mlp.get_weights())

        tail_pred = self.kg_pred_mlp(head_relation_concat)
        kg_mlp_weight.append(self.kg_pred_mlp.get_weights())
        sigmoid = nn.Sigmoid()
        tail_pred = sigmoid(tail_pred)

        score_kg = sigmoid((tail_embedding * tail_pred).sum(1))
        rmse_loss = torch.sqrt(
            ((tail_embedding - tail_pred) ** 2).sum(1) / self.dim
        ).mean()

        return score_kg, rmse_loss

    def get_weights(self):
        return self.kg_mlp.get_weights() + self.kg_pred_mlp.get_weights()


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

    def get_rs_weights(self):
        return self.user_mlp.get_weights() + self.cc_unit.get_weights()

    def get_kg_weights(self):
        return self.tail_mlp.get_weights() + self.cc_unit.get_weights()





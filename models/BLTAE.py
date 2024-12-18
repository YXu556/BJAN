"""
This script is for re-implement of Garnot & Landrieu 2020
http://arxiv.org/abs/2007.00586
todo:
    *
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# heavy dropout
# class DLTAE(nn.Module):
#     def __init__(self, input_dim=10, num_classes=9, d_model=256, n_head=16,
#                  d_k=8, dropout=0.2, max_len=366, max_seq_len=70, T=1000, max_temporal_shift=30):
#         super(DLTAE, self).__init__()
#
#         self.modelname = self._get_name()
#         self.max_seq_len = max_seq_len
#
#         self.mlp_dim = [input_dim, 32, 64, 128]  # 128 do not modify it
#         layers = []
#         for i in range(len(self.mlp_dim) - 1):
#             layers.append(linlayer(self.mlp_dim[i], self.mlp_dim[i + 1], dropout=dropout))
#         self.mlp1 = nn.Sequential(*layers)
#
#         self.inlayernorm = nn.LayerNorm(128)
#         self.inconv = LinearLayer(128, d_model, dropout=dropout)
#         self.position_enc = PositionalEncoding(d_model, max_len=max_len + 2 * max_temporal_shift, T=T)
#
#         self.attention_heads = MultiHeadAttention(
#             n_head=n_head, d_k=d_k, d_in=d_model)
#
#         layers = []
#         n_neurons = [d_model, 128]
#         for i in range(len(n_neurons) - 1):
#             layers.extend([nn.Linear(n_neurons[i], n_neurons[i + 1]),
#                            nn.BatchNorm1d(n_neurons[i + 1]),
#                            nn.ReLU(),
#                            nn.Dropout(dropout)
#                            ])
#         self.mlp2 = nn.Sequential(*layers)
#         self.dropout = nn.Dropout(dropout)
#         self.outlayernorm = nn.LayerNorm(n_neurons[-1])
#
#         layers = []
#         decoder = [128, 64, 32, num_classes]
#         for i in range(len(decoder) - 1):
#             layers.append(nn.Linear(decoder[i], decoder[i + 1]))
#             if i < (len(decoder) - 2):
#                 layers.extend([
#                     nn.BatchNorm1d(decoder[i + 1]),
#                     nn.ReLU(),
#                     nn.Dropout(dropout)
#                 ])
#         self.decoder = nn.Sequential(*layers)
#
#     def forward(self, x, return_feats=False):
#         x, mask, doy = x
#         sz_b, sz_s, _ = x.shape
#
#         x = x.permute((0, 2, 1))
#         x = self.mlp1(x)
#         x = x.permute((0, 2, 1))
#
#         x = self.inlayernorm(x)
#         x = self.inconv(x)
#
#         enc_output = self.dropout(x + self.position_enc(doy))
#
#         enc_output, attn = self.attention_heads(enc_output, enc_output, enc_output)
#         enc_output = enc_output.permute(1, 0, 2).contiguous().view(sz_b, -1)  # Concatenate heads
#         enc_output = self.outlayernorm(self.dropout(self.mlp2(enc_output)))
#
#         out = self.decoder(enc_output)
#
#         if return_feats:
#             return out, enc_output
#         else:
#             return out


# weak dropout
class DLTAE(nn.Module):
    def __init__(self, input_dim=10, num_classes=9, d_model=256, n_head=16,
                 d_k=8, dropout=0.5, max_len=366, max_seq_len=70, T=1000, max_temporal_shift=30):
        super(DLTAE, self).__init__()

        self.modelname = self._get_name()
        self.max_seq_len = max_seq_len

        self.mlp_dim = [input_dim, 32, 64, 128]  # 128 do not modify it
        layers = []
        for i in range(len(self.mlp_dim) - 1):
            layers.append(linlayer(self.mlp_dim[i], self.mlp_dim[i + 1]))
        self.mlp1 = nn.Sequential(*layers)

        self.inlayernorm = nn.LayerNorm(128)
        # self.inconv = LinearLayer(128, d_model, dropout=0)
        self.inconv = nn.Sequential(nn.Conv1d(128, d_model, 1),
                                    nn.LayerNorm((d_model, max_seq_len)))

        self.position_enc = PositionalEncoding(d_model, max_len=max_len + 2 * max_temporal_shift, T=T)

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=d_model)

        layers = []
        n_neurons = [d_model, 128]
        for i in range(len(n_neurons) - 1):
            layers.extend([nn.Linear(n_neurons[i], n_neurons[i + 1]),
                           nn.BatchNorm1d(n_neurons[i + 1]),
                           nn.ReLU(),
                           ])
        self.mlp2 = nn.Sequential(*layers)
        # self.dropout = nn.Dropout(dropout)
        self.outlayernorm = nn.LayerNorm(n_neurons[-1])

        layers = []
        decoder = [128, 64, 32, num_classes]
        for i in range(len(decoder) - 1):
            layers.append(nn.Linear(decoder[i], decoder[i + 1]))
            if i < (len(decoder) - 2):
                layers.extend([
                    nn.BatchNorm1d(decoder[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
        self.decoder = nn.Sequential(*layers)

    def forward(self, x, return_feats=False):
        x, mask, doy = x
        sz_b, sz_s, _ = x.shape

        x = x.permute((0, 2, 1))
        x = self.mlp1(x)
        x = x.permute((0, 2, 1))

        x = self.inlayernorm(x)
        # x = self.inconv(x)
        x = self.inconv(x.permute(0, 2, 1)).permute(0, 2, 1)

        enc_output = x + self.position_enc(doy)

        enc_output, attn = self.attention_heads(enc_output, enc_output, enc_output)
        enc_output = enc_output.permute(1, 0, 2).contiguous().view(sz_b, -1)  # Concatenate heads
        enc_output = self.outlayernorm(self.mlp2(enc_output))
        # enc_output = self.outlayernorm(self.dropout(self.mlp2(enc_output)))

        out = self.decoder(enc_output)

        if return_feats:
            return out, enc_output
        else:
            return out


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, q, k, v):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = q.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
        output, attn = self.attention(q, k, v)
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn


class linlayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(linlayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.lin = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, input):
        out = input.permute((0, 2, 1))  # to channel last
        out = self.lin(out)

        out = out.permute((0, 2, 1))  # to channel first
        out = self.bn(out)
        out = F.relu(out)

        return out


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.BatchNorm1d(out_dim)
        self.activation = nn.ReLU()
        self.do = nn.Dropout(p=dropout)

    def forward(self, x):
        # x = (B, C) or (B, S, C)
        x = self.linear(x)  # linear expect channels last
        if x.dim() == 3:
            # BatchNorm1d expects channels first, move to (B, C, S)
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        else:  # (B, C)
            x = self.norm(x)
        return self.do(self.activation(x))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, T: int = 10000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(T) / d_model))
        pe = torch.zeros(max_len + 1, d_model)
        pe[1:, 0::2] = torch.sin(position * div_term)
        pe[1:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, doy):
        """
        Args:
            doy: Tensor, shape [batch_size, seq_len]
        """
        return self.pe[doy]


if __name__ == '__main__':
    def get_ntrainparams(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = DLTAE()

    print(get_ntrainparams(model))
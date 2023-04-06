"""Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive
Transformers with Linear Attention" Modified from:

https://github.com/idiap/fast-
transformers/blob/master/fast_transformers/attention/linear_attention.py.
"""
import pdb

import torch
from torch import nn
from torch.nn import Dropout, Module


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        # pdb.set_trace()
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum('nshd,nshv->nhdv', K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum('nlhd,nhd->nlh', Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum('nlhd,nhdv,nlh->nlhv', Q, KV,
                                      Z) * v_length

        return queried_values.contiguous()


class MultiHeadAttention(Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        kdim=None,
        vdim=None,
        attention='linear',
    ):
        super(MultiHeadAttention, self).__init__()
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        # self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.nhead = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        assert (self.head_dim * num_heads == self.embed_dim
                ), 'embed_dim must be divisible by num_heads'

        # multi-head attention
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.attention = LinearAttention(
        ) if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, q, k, v, q_mask=None, kv_mask=None):
        bs = q.size(0)
        # multi-head attention
        query = self.q_proj(q).view(bs, -1, self.nhead,
                                    self.head_dim)  # [N, L, (H, D)]
        key = self.k_proj(k).view(bs, -1, self.nhead,
                                  self.head_dim)  # [N, S, (H, D)]
        value = self.v_proj(v).view(bs, -1, self.nhead, self.head_dim)
        message = self.attention(query,
                                 key,
                                 value,
                                 q_mask=q_mask,
                                 kv_mask=kv_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead *
                                          self.head_dim))  # [N, L, C]

        return message


class SimAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

        self.linear = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        # pdb.set_trace()
        Q = self.feature_map(queries)
        K = self.feature_map(keys)
        # Q = queries
        # K = keys

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
            queries = queries * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        # v_length = values.size(1)
        # values = values / v_length  # prevent fp16 overflow
        # KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        # Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        # queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        qk = torch.einsum('nlhd,nshd->nlsh', Q, K)  # [n,l,s,h]
        qk_mean = torch.mean(qk, dim=2, keepdim=True)  # [n,l,1,h]
        qk_max, _ = torch.max(qk, dim=2, keepdim=True)  # [n,l,1,h]
        atten = torch.cat([qk_mean, qk_max], dim=2).permute(0, 1, 3,
                                                            2)  # [n,l,h,2]
        atten = self.sigmoid(self.linear(atten).squeeze(-1))  # [n,l,h]
        queried_values = torch.einsum('nlhd,nlh->nlhd', queries, atten)
        # queried_values = torch.einsum('nlhd,nlh->nlhd', Q, atten)
        return queried_values.contiguous()


class FullAttention(Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-head scaled dot-product attention, a.k.a full attention.

        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum('nlhd,nshd->nlsh', queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(
                ~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]),
                float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1.0 / queries.size(3)**0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum('nlsh,nshd->nlhd', A, values)

        return queried_values.contiguous()

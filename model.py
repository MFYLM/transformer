import torch
import math
import warnings
import copy
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torch.nn import functional as F
import torch.distributed as dist
from torch import Tensor


class GPTConfig():
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0
    bias: bool = True


def clone(layer, n_layer):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])


class LayerNorm(nn.Module):
    """
    standard layer normalization
    """
    def __init__(self, ndim: int, eps: float = 1e-6, bias = None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        
    def forward(self, inputs: Tensor):
        mean = inputs.mean(-1, keepdim=True)
        std = inputs.std(-1, keepdim=True)
        return (inputs - mean) / math.sqrt(std + self.eps) * self.weight + self.bias


class AddNorm(nn.Module):
    """
    apply residual connection and complete add & norm layer
    """
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, y: Tensor):
        # FIXME: normalization order?
        return self.norm(x + self.dropout(y))


# TODO: add regularization: attention, residual (where)
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.W_Q = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        self.W_K = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        self.W_V = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)

        self.n_head = config.n_head
        self.n_layer = config.n_layer
        self.n_embed = config.n_embed

    def forward(self, x: Tensor, isMask: bool = False, mask = None):
        assert isMask == (mask is not None)
        B, L, E = x.size()                                                         # batch size, sequence length, embedding size
        q, k, v = self.W_Q(x), self.W_K(x), self.W_V(x)
        q = q.view(B, L, self.n_head, self.n_embed // self.n_head).transpose(1, 2) # B * H * L * (E // H)
        k = k.view(B, L, self.n_head, self.n_embed // self.n_head).transpose(1, 2) # B * H * L * (E // H)
        v = v.view(B, L, self.n_head, self.n_embed // self.n_head).transpose(1, 2) # B * H * L * (E // H)

        attn_score = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))       # B * H * L * L
        if isMask:
            attn_score = attn_score.masked_fill(mask == 0, -1e10)                  # forbid model to see the future
        attn_score = F.softmax(attn_score, dim=-1)
        y: Tensor = attn_score @ v                                                 # B * H * L * (E // H) 
        y = y.transpose(1, 2).contiguous().view(B, L, E)                           # reassemble outputs from different heads   
        return y


class PositionwiseFeedforward(nn.Module):
    def __init__(self, size, hid_size, dropout: float = 0.1):
        self.layer1 = nn.Linear(size, hid_size)
        self.layer2 = nn.Linear(hid_size, size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.layer1(x).relu()
        x = self.layer2(self.dropout(x))
        return x


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: Tensor, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """encoder layer consists of self attention layer and feedforward layer"""
    def __init__(self, size, self_attn: MultiHeadSelfAttention, feed_forward: PositionwiseFeedforward, dropout: int):
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.layer = clone(AddNorm(size, dropout), 2)

    def forward(self, x: Tensor, mask):
        assert x.size(-1) == self.size
        y = self.layer[0](x, self.self_attn(x, True, mask))
        y = self.layer[1](x, self.feed_forward(y))
        return y


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embdd, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embdd
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
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


def stackLayers(layer, n_layer):
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


class AddNormConnetion(nn.Module):
    """
    apply residual connection and complete add & norm layer
    """
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, x_direct: Tensor):
        return x_direct + self.dropout(self.norm(x))

        


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

        # TODO: regularization: attention, residual


    def forward(self, x: Tensor):
        B, L, E = x.size()  # batch size, sequence length, embedding size
        q, k, v = self.W_Q(x), self.W_K(x), self.W_V(x)
        q = q.view(B, L, self.n)

        attn_score = q @ k.transpose(1, 2)



class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = stackLayers(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: Tensor, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """encoder layer consists of self attention layer and feedforward layer"""
    def __init__(self, size, self_attn: MultiHeadSelfAttention, feed_forward, dropout: int):
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.layer = stackLayers(AddNormConnetion(size, dropout), 2)

    def forward(self, x: Tensor, mask):
        x_direct = self.self_attn(x, )


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = stackLayers(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# class EncoderDecoder(nn.Module):
#     def __init__(self, encoder, decoder, src_embdd, tgt_embed, generator):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.src_embed = src_embdd
#         self.tgt_embed = tgt_embed
#         self.generator = generator

#     def encode(self, src, src_mask):
#         return self.encoder(self.src_embed(src), src_mask)
    
#     def decode(self, memory, src_mask, tgt, tgt_mask):
#         return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

#     def forward(self, src, tgt, src_mask, tgt_mask):
#         return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
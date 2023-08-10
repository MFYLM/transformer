import torch
import math
import copy
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
    n_head: int = 8
    n_embed: int = 768
    dropout: float = 0.1
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
        
    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (x - mean) / math.sqrt(std + self.eps) * self.weight + self.bias


class AddNorm(nn.Module):
    """
    apply residual connection and complete add & norm layer
    """
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        x is original vector (residual)
        """
        return self.norm(x + self.dropout(y))


class PositionEncoding(nn.Module):
    def __init__(self, d_model: int, dropout, max_length = 1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_length, d_model)
        pos = torch.arange(0, max_length).unsqueeze(1)      # adding extra dimension
        divide_term = torch.exp( 
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = math.sin(pos * divide_term)
        pe[:, 1::2] = math.cos(pos * divide_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


# TODO: add regularization: attention, residual (where)
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_head: int, n_embed: int, bias: bool = True):
        super().__init__()
        assert n_embed % n_head == 0
        self.W_Q = nn.Linear(n_embed, n_embed, bias=bias)
        self.W_K = nn.Linear(n_embed, n_embed, bias=bias)
        self.W_V = nn.Linear(n_embed, n_embed, bias=bias)

        self.n_head = n_head
        self.n_embed = n_embed

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask = None) -> Tensor:
        B, L, E = query.size()                                                         # batch size, sequence length, embedding size
        q, k, v = self.W_Q(query), self.W_V(key), self.W_K(value)
        q = q.view(B, L, self.n_head, self.n_embed // self.n_head).transpose(1, 2) # B * H * L * (E // H)
        k = k.view(B, L, self.n_head, self.n_embed // self.n_head).transpose(1, 2) # B * H * L * (E // H)
        v = v.view(B, L, self.n_head, self.n_embed // self.n_head).transpose(1, 2) # B * H * L * (E // H)

        attn_score: Tensor = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))       # B * H * L * L
        del query
        del key
        del value
        if mask:
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
    
    def forward(self, x) -> Tensor:
        x = self.layer1(x).relu()
        x = self.layer2(self.dropout(x))
        return x


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: Tensor, mask) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerEncoderLayer(nn.Module):
    """encoder layer consists of self attention layer and feedforward layer"""
    def __init__(self, size, self_attn: MultiHeadSelfAttention, feed_forward: PositionwiseFeedforward, dropout: float):
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.layer = clone(AddNorm(size, dropout), 2)

    def forward(self, x: Tensor, mask) -> Tensor:
        assert x.size(-1) == self.size
        y = self.layer[0](x, self.self_attn(x, True, mask))
        y = self.layer[1](x, self.feed_forward(y))
        return y


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, size: int, masked_attn: MultiHeadSelfAttention, self_attn: MultiHeadSelfAttention, feed_forward: PositionwiseFeedforward, dropout: float = 0.1):
        self.size = size
        self.masked_attn = masked_attn
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.addnorms = clone(AddNorm(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.addnorms[0](x, self.masked_attn(x, x, x, mask=src_mask))
        x = self.addnorms[1](x, self.self_attn(x, m, m, mask=tgt_mask))
        return self.addnorms[2](x, self.feed_forward(x))


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


class Generator(nn.Module):
    def __init__(self, d_model: int, n_vocab: int):
        super().__init__()
        self.layer = nn.Linear(d_model, n_vocab)

    def forward(self, x):
        # convert to probability
        return F.log_softmax(self.layer(x), dim=-1)


class TransformerBlock(nn.Module):
    def __init__(
        self, 
        src_vocab: int, 
        tgt_vocab, N: int = 6, 
        d_model: int = 512, 
        hid_size: int = 2048, 
        n_head: int = 8, 
        dropout: float = 0.1
    ):
        """
        hid_size: size of hidden layer in feed forward block
        """
        super().__init__()
        c = copy.deepcopy
        attn = MultiHeadSelfAttention(n_head, d_model)
        feed_forward = PositionwiseFeedforward(d_model, hid_size, dropout)
        position = PositionEncoding(d_model, dropout)
        self.model = EncoderDecoder(
            encoder=TransformerEncoderLayer(d_model, c(attn), c(feed_forward), dropout),
            decoder=TransformerDecoderLayer(d_model, c(attn), c(attn), c(feed_forward), dropout),
            src_embdd=nn.Sequential(nn.Embedding(d_model, src_vocab), c(position)),
            tgt_embed=nn.Sequential(nn.Embedding(d_model, tgt_vocab), c(position)),
            generator=Generator(d_model, tgt_vocab),
        )

        del attn
        del feed_forward
        del position
        # initial parameters for the model
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


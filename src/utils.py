import torch
from torch import nn, Tensor
from typing import Callable


class FastGLU(nn.Module):
    """
    implement GLP activation function, allow flexible activation function
    """
    def __init__(self, size: int):
        self.size = size
        self.layer = nn.Linear(size, size * 2)

    def forward(self, x: Tensor, activation: Callable[[Tensor], Tensor]):
        x = self.layer(x)
        return x[:, :self.size] * activation(x[:, self.size:])


class Batch:
    """
    batch holds src and target sequence as well as generating mask
    """
    def __init__(self, src, tgt = None, pad: int = 2) -> None:
        """
        pad: token ID (2: <blank> in this case)
        """
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
    
    # @staticmethod
    # def 


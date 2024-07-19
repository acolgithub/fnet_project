import torch
import torch.nn as nn
import numpy as np

from .parameters import Params
from .layers import Embedder, Fourier, FeedForward, Pooler



class FNet(nn.Module):

    def __init__(self, params: Params):
        super().__init__()
        self.model = nn.ModuleList([])

        # append embedder
        self.model.append(nn.ModuleList([Embedder(params)]))

        # append Fourier blocks
        for _ in range(params.number_of_layers):
            self.model.append(nn.ModuleList([
                Fourier(),
                FeedForward(),
            ]))

        # add pooler
        self.model.append(nn.ModuleList([Pooler(params)]))


    def forward(self, X: torch.tensor, params: Params):

        for module in self.model.modules():
            if isinstance(module, Embedder):
                X = module(X)
            else:
                X = module(X, params)

        return X


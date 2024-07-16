import torch
import torch.nn as nn

import scipy as sp
import numpy as np

from src.parameters import Params
from src.layers import Embedder, Fourier, Normalize, FeedForward, Dense
    

class FNet(nn.Module):

    def __init__(self, params: Params):
        super().__init__()
        self.model = nn.ModuleList([])

        # append embedder
        self.model.append(nn.ModuleList([Embedder(
            params.num_embeddings,
            params.embedding_dim
        )]))

        # append Fourier blocks
        for _ in range(params.N):
            self.model.append(nn.ModuleList([
                Fourier(),
                Normalize(params.embedding_dim),
                FeedForward(),
                Normalize(params.embedding_dim)
            ]))

        # add dense layer
        self.model.append(nn.ModuleList([Dense(
                params.input_size,
                params.num_classes,
                params.dropout_rate
        )]))


    def forward(self, X: np.array, params: Params):

        for module in self.model.modules():
            if isinstance(module, Embedder):
                X = module(X)
            else:
                X = module(X, params)

        return X
    

# embedder = Embedder(10, 3)
# input = torch.LongTensor([[1, 2, 4, 5],[4, 3, 2, 9]])
# print(embedder(input))

params = Params()

four = Fourier()
print(four(np.array([[1,0],[0,1]]), params))




















import torch
import torch.nn as nn

import scipy as sp
import numpy as np

from .parameters import Params


# define embedder for FNet architecture
class Embedder(nn.Module):

    # num_embeddings is the size of the dictionary of embeddings
    # embedding_dim is the size of each embedding vector
    def __init__(self, params: Params):
        super().__init__()
        self.embed = nn.Embedding(
            params.num_embeddings,
            params.embedding_dim
        )

    def forward(self, X: np.array):

        output = self.embed(X)
        return output


# create neural net module
class Fourier(nn.Module):

    def __init__(self):
        super().__init__()

    
    def forward(self, X: np.array, params: Params):

        # iterate over different axes of array and apply fft
        for s in range(X.ndim):
            X = sp.fft.fftn(X, axes=s)

        # return the real part
        return X.real / np.sqrt(params.N)
    

class Normalize(nn.Module):

    def __init__(self, params: Params):
        super().__init__()
        self.layer_norm = nn.LayerNorm(params.embedding_dim)

    def forward(self, X: np.array):
        return self.layer_norm(X)
    
class FeedForward(nn.Module):

    def __init__(self, params: Params):
        super().__init__()
        self.linear1 = nn.Linear(
            params.input_size,
            params.num_classes,
            params.dropout_rate
        )
        self.dropout = nn.Dropout(params.dropout_rate)
    

class Dense(nn.Module):

    def __init__(self, params: Params):
        super().__init__()

        # include dense layer and activation layer
        self.linear = nn.Linear(params.input_size, params.num_classes)
        self.dropout = nn.Dropout(params.dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, X: np.array):

        linear_ouput = self.linear(X)
        dropout_output = self.dropout(linear_ouput)
        activation_output = self.relu(dropout_output)

        return activation_output
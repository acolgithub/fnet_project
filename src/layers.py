import torch
import torch.nn as nn

import scipy as sp
import numpy as np

from .parameters import Params


# define embedder for FNet architecture
class Embedder(nn.Module):

    def __init__(self, params: Params):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            num_embeddings = params.vocab_size,
            embedding_dim = params.embedding_dimension#,
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings = params.maximum_position_embeddings,
            embedding_dim = params.embedding_dimension
        )
        self.token_type_embeddings = nn.Embedding(
            num_embeddings = params.token_type_embedding_hidden_size,
            embedding_dim = params.embedding_dimension
        )

        # creates tensor of shape (1, maximum_position_embeddings) in memory
        # can be referred to by position_ids
        self.register_buffer(
            "position_ids",
            torch.arange(params.maximum_position_embeddings).expand((1, -1)),
            persistent=False
        )

        # additional layers found in the google implementation
        self.embedding_layer_norm = nn.LayerNorm(
            normalized_shape = params.embedding_dimension,
            eps = params.norm_epsilon
        )
        self.embedding_linear_layer = nn.Linear(
            in_features = params.embedding_dimension,
            out_features = params.embedding_dimension
        )
        self.embedding_dropout = nn.Dropout(params.dropout_rate)


    def forward(self, input_ids, token_type_ids):

        seq_length = input_ids.size()[0]
        position_ids = self.position_ids[:,:seq_length]

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # add embeddings
        embeddings = word_embeddings + position_embeddings + token_type_embeddings

        # apply layer norm, linear layer, and dropout
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.embedding_linear_layer(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        return embeddings


# create neural net module
class Fourier(nn.Module):

    def __init__(self):
        super().__init__()

    
    def forward(self, X: np.array, params: Params):

        # iterate over different axes of array and apply fft
        for s in range(X.ndim):
            X = sp.fft.fftn(X, axes=s)

        # return the real part
        return X.real / np.sqrt(params.number_of_layers)
    

class FeedForward(nn.Module):

    def __init__(self, params: Params):
        super().__init__()

        self.linear = nn.Linear(
            params.feed_forward_input_size,
            params.feed_forward_num_classes
        )
        self.gelu = nn.GELU()
        self.linear = nn.Linear(params.input_size, params.num_classes)
        self.dropout = nn.Dropout(params.dropout_rate)


    def forward(self, X: np.array):

        linear_ouput = self.linear(X)
        dropout_output = self.dropout(linear_ouput)
        activation_output = self.gelu(dropout_output)

        return activation_output
    

class Pooler(nn.Module):

    def __init__(self, params: Params):
        super().__init__()
        self.linear = nn.Linear(
            params.pooler_linear_input_size,
            params.pooler_linear_num_classes,
            params.dropout_rate
        )
        self.pooler_linear = nn.Linear()
        self.pooler_activation = nn.Tanh()

    def forward(self, hidden_states):

        pooler_output = self.pooler_linear(hidden_states)
        pooler_output = self.pooler_activation(pooler_output)
        return pooler_output
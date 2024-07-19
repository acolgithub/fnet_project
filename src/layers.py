import torch
import torch.nn as nn
import numpy as np

from .parameters import Params


# define embedder for FNet architecture
class Embedder(nn.Module):

    def __init__(self, params: Params):
        super().__init__()

        # embeddings used in fnet architecture
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


    def forward(self, input_ids: torch.tensor, token_type_ids: torch.tensor):

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


# create fourier layer of model
class Fourier(nn.Module):

    def __init__(self):
        super().__init__()

    
    def forward(self, hidden_layer: torch.tensor, params: Params):

        # iterate over different axes of array and apply fft
        for s in range(hidden_layer.ndim):
            X = torch.fft.fftn(hidden_layer, dim=s)

        # return the real part
        return X.real / np.sqrt(params.number_of_layers)
    

class FeedForward(nn.Module):

    def __init__(self, params: Params):
        super().__init__()

        self.linear1 = nn.Linear(
            params.embedding_dimension,
            params.embedding_dimension
        )
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(
            params.embedding_dimension,
            params.embedding_dimension
        )
        self.dropout = nn.Dropout(params.dropout_rate)


    def forward(self, hidden_layer: torch.tensor):

        linear1_ouput = self.linear1(hidden_layer)
        activation_output = self.gelu(linear1_ouput)
        linear2_output = self.linear2(activation_output)
        dropout_output = self.dropout(linear2_output)


        return dropout_output
    

class Pooler(nn.Module):

    def __init__(self, params: Params):
        super().__init__()
        self.pooler_linear = nn.Linear(
            params.embedding_dimension,
            params.pooler_linear_num_classes,
        )
        self.pooler_activation = nn.Tanh()

    def forward(self, hidden_states):

        pooler_output = self.pooler_linear(hidden_states)
        pooler_output = self.pooler_activation(pooler_output)
        return pooler_output
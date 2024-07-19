import torch
import torch.nn as nn
import math

from .parameters import Params


# define embedder for FNet architecture
class Embedder(nn.Module):

    def __init__(self, params: Params) -> None:
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
            eps = params.layer_norm_epsilon
        )
        self.embedding_linear_layer = nn.Linear(
            in_features = params.embedding_dimension,
            out_features = params.embedding_dimension
        )
        self.embedding_dropout = nn.Dropout(params.dropout_rate)


    def forward(self, input_ids: torch.tensor, token_type_ids: torch.tensor) -> torch.tensor:

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
    """
    Class to implement the layer which applies the fourier transform.
    """

    def __init__(self) -> None:
        super().__init__()

    
    def forward(self, hidden_layer: torch.tensor, params: Params) -> torch.tensor:

        # iterate over different axes of array and apply fft
        for s in range(hidden_layer.ndim):
            X = torch.fft.fftn(hidden_layer, dim=s)

        # return the real part after dividing by sqrt(number_of_layers)
        return X.real / math.sqrt(params.number_of_layers)
    

class FeedForward(nn.Module):

    def __init__(self, params: Params) -> None:
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
        self.feed_forward_layer_norm = nn.LayerNorm(
            normalized_shape = params.embedding_dimension,
            eps = params.layer_norm_epsilon
        )


    def forward(self, hidden_layer: torch.tensor) -> torch.tensor:

        linear1_ouput = self.linear1(hidden_layer)
        activation_output = self.gelu(linear1_ouput)
        linear2_output = self.linear2(activation_output)
        dropout_output = self.dropout(linear2_output)
        layer_norm_output = self.feed_forward_layer_norm(dropout_output)


        return layer_norm_output
    

class FNetLayer(nn.Module):
    """
    Class to combine the fourier transform layer with the feed forward layer.
    """

    def __init__(self, params: Params) -> None:
        super().__init__()
        self.fourier = Fourier()
        self.feed_forward = FeedForward(params)

    def forward(self, hidden_layer: torch.tensor, params: Params) -> torch.tensor:

        fourier_output = self.fourier(hidden_layer, params)
        feed_forward_output = self.feed_forward(fourier_output)
        return feed_forward_output
    

class FNetEncoder(nn.Module):
    """
    Class to group together all the layers involving the fourier transform
    layer combined with the feed forward layer.
    """

    def __init__(self, params: Params) -> None:
        super().__init__()
        self.iterated_layer = nn.ModuleList([])
        for _ in range(params.number_of_layers):
            self.iterated_layer.append(FNetLayer(params))

    def forward(self, hidden_layer: torch.tensor, params: Params) -> torch.tensor:

        for fnetlayer_module in self.iterated_layer:

            # apply each fnet layer in module list
            hidden_layer = fnetlayer_module(hidden_layer, params)

        return hidden_layer
    

class Pooler(nn.Module):

    def __init__(self, params: Params) -> None:
        super().__init__()
        self.pooler_linear = nn.Linear(
            params.embedding_dimension,
            params.pooler_linear_num_classes,
        )
        self.pooler_activation = nn.Tanh()

    def forward(self, hidden_states) -> torch.tensor:

        pooler_output = self.pooler_linear(hidden_states)
        pooler_output = self.pooler_activation(pooler_output)
        return pooler_output
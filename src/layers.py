import math
import torch
import torch.nn as nn

from .parameters import Config


# define embedder for FNet architecture
class Embedder(nn.Module):
    """Embedder layer for FNet architecture.

    Attributes:
        word_embeddings: Embedder applied to input_ids.
        position_embeddings: Embedder applied to position_ids.
        token_type_embeddings: Embedder applied to token_type_ids.
        embedding_layer_norm: Normalization layer contained in Embedder.
        embedding_linear_layer: Linear layer contained in Embedder.
        embedding_dropout: Dropout layer to randomly zero out linear layer.
    """

    def __init__(self, config: Config) -> None:
        """Initializes Embedder layer of FNet architecture.

        Args:
            config: Configuration parameters of the model.
        """

        super().__init__()

        # embeddings used in fnet architecture
        self.word_embeddings = nn.Embedding(
            num_embeddings = config.embedding_params.vocab_size,
            embedding_dim = config.embedding_params.embedding_dimension
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings = config.embedding_params.maximum_position_embeddings,
            embedding_dim = config.embedding_params.embedding_dimension
        )
        self.token_type_embeddings = nn.Embedding(
            num_embeddings = config.embedding_params.token_type_embedding_size,
            embedding_dim = config.embedding_params.embedding_dimension
        )

        # creates tensor of shape (1, maximum_position_embeddings) in memory
        # can be referred to by position_ids
        self.register_buffer(
            "position_ids",
            torch.arange(config.embedding_params.maximum_position_embeddings).expand((1, -1)),
            persistent=False
        )

        # additional layers found in the google implementation
        self.embedding_layer_norm = nn.LayerNorm(
            normalized_shape = config.embedding_params.embedding_dimension,
            eps = config.embedding_params.layer_norm_epsilon
        )
        self.embedding_linear_layer = nn.Linear(
            in_features = config.embedding_params.embedding_dimension,
            out_features = config.embedding_params.embedding_dimension
        )
        self.embedding_dropout = nn.Dropout(config.embedding_params.dropout_rate)


    def forward(self, input_ids: torch.tensor, token_type_ids: torch.tensor) -> torch.tensor:
        """Defines forward pass for Embedder layer.

        Args:
            input_ids: Torch tensor containing input ids from tokenizer.
            token_type_ids: Torch tensor containing token type ids from tokenizer.
        """

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
    """Class to implement the layer which applies the fourier transform.
    
    Attributes:
        fourier_layer_norm: Layer norm applied to sum for normalization.
    """

    def __init__(self, config: Config) -> None:
        """Initializes Fourier transform layer of FNet architecture.

        Args:
            config: Configuration parameters of the model.
        """

        super().__init__()
        self.fourier_layer_norm = nn.LayerNorm(config.fnetencoder_params.fourier_layer_norm)

    
    def forward(self, hidden_layer: torch.tensor) -> torch.tensor:
        """Defines forward pass for Fourier Transform layer.

        Args:
            hidden_layer: Input torch tensor output from previous layers.
        """

        # pass hidden layer
        output = hidden_layer

        # iterate over last two axes of array and apply fft
        for s in range(-1, -3 ,-1):
            output = torch.fft.fftn(output, dim=s)

        # obtain the real part
        output = output.real

        # add and apply layer norm
        output = self.fourier_layer_norm(hidden_layer + output)

        return output
    

class FeedForward(nn.Module):
    """Feedforward neural network in FNet architecture.

    Attributes:
        linear1: First linear layer within neural network.
        gelu: Gaussian error linear unit activation.
        linear2: Second linear layer within neural network.
        dropout: Dropout layer to randomly zero out second linear layer
        feed_forward_layer_norm: Layer applied to sum for normalization.
    """

    def __init__(self, config: Config) -> None:
        """Initalizes feed forward layer of FNet architecture.

        Args:
            config: Configuration parameters of the model.
        """

        super().__init__()

        self.linear1 = nn.Linear(
            config.fnetencoder_params.fnet_layer1_input_dimension,
            config.fnetencoder_params.fnet_layer1_output_dimension
        )
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(
            config.fnetencoder_params.fnet_layer2_input_dimension,
            config.fnetencoder_params.fnet_layer2_output_dimension
        )
        self.dropout = nn.Dropout(config.fnetencoder_params.dropout_rate)
        self.feed_forward_layer_norm = nn.LayerNorm(
            normalized_shape = config.fnetencoder_params.fnet_layer_norm_dimension,
            eps = config.fnetencoder_params.layer_norm_epsilon
        )


    def forward(self, hidden_layer: torch.tensor) -> torch.tensor:
        """Defines forward pass for feed forward layer.

        Args:
            hidden_layer: Input torch tensor output from previous layer.
        """

        output = self.linear1(hidden_layer)
        output = self.gelu(output)
        output = self.linear2(output)
        output = self.dropout(output)
        output = self.feed_forward_layer_norm(hidden_layer + output)  # add to mix


        return output
    

class FNetLayer(nn.Module):
    """Class to combine the fourier transform layer with the feed forward layer.

    Attributes:
        fourier: A fourier transform layer of the FNet architecture.
        feed_forward: A feed forward layer of the FNet architecture.
    """

    def __init__(self, config: Config) -> None:
        """Initalizes layer combining fourier layer and feedforward layer.

        Args:
            config: Configuration parameters of the model.
        """

        super().__init__()
        self.fourier = Fourier(config)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden_layer: torch.tensor) -> torch.tensor:
        """Defines forward pass for the combined Fourier transform and feed forward layers.

        Args:
            hidden_layer: Input torch tensor output from previous layer.
        """

        fourier_output = self.fourier(hidden_layer)
        feed_forward_output = self.feed_forward(fourier_output)
        return feed_forward_output
    

class FNetEncoder(nn.Module):
    """Class to group together all the layers involving the fourier transform
       layer combined with the feed forward layer.

    Attributes:
        iterated_layer: List of FNetLayer blocks.
    """

    def __init__(self, config: Config) -> None:
        """Initializes FNetEncoder block.

        Args:
            config: Configuration parameters of the model.
        """

        super().__init__()
        self.iterated_layer = nn.ModuleList([])
        for _ in range(config.fnetencoder_params.number_of_layers):
            self.iterated_layer.append(FNetLayer(config))

    def forward(self, hidden_layer: torch.tensor) -> torch.tensor:
        """Defines forward pass for FNetEncoder block.

        Args:
            hidden_layer: Input torch tensor output from previous layer.
        """

        for fnetlayer_module in self.iterated_layer:

            # apply each fnet layer in module list
            hidden_layer = fnetlayer_module(hidden_layer)

        return hidden_layer
    

class Pooler(nn.Module):
    """Pooling layer for FNet architecture.

    Attributes:
        pooler_linear: Linear layer of Pooling layer.
        pooler_activation: Tanh activation layer.
    """

    def __init__(self, config: Config) -> None:
        """Initializes Pooler layer of FNet architecture.

        Args:
            config: Configuration parametes of the model.
        """

        super().__init__()
        self.pooler_linear = nn.Linear(
            config.pooler_params.pooler_linear_input_dimensions,
            config.pooler_params.pooler_linear_output_dimension,
        )
        self.pooler_activation = nn.Tanh()

    def forward(self, hidden_layer: torch.tensor) -> torch.tensor:
        """Defines forward pass for Pooler layer of Fnet architecture.

        Args:
            hidden_layer: Input torch tensor output from previous layer.
        """

        pooler_output = self.pooler_linear(hidden_layer)
        pooler_output = self.pooler_activation(pooler_output)
        return pooler_output
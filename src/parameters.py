class Params():
    """Class to store the general configuration parameters of the model.

    Attributes:
        layer_norm_epsilon: A value added to denominator of nn.LayerNorm for numerical stability.
        dropout_rate:  Probability of an element to be zeroed.
        number_of_layers: Number of FNetLayer blocks in FNetEncoder.
        hidden_dimension: Dimension of intermediate hidden layers.
    """

    def __init__(
        self,

        # general parameters
        layer_norm_epsilon = 1e-12,
        dropout_rate = 0.2,
        number_of_layers = 2,
        hidden_dimension = 10
    ) -> None:
        """Initializes Params class with the general model configuration parameters.

        """
        
        # general parameters
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout_rate = dropout_rate
        self.number_of_layers = number_of_layers
        self.hidden_dimension = hidden_dimension



class EmbeddingParams(Params):
    """Class to store the Embedding layer parameters.

    Attributes:
        vocab_size: Size of the base vocabulary.
        maximum_position_embeddings: The embedding size for the position embedder.
        token_type_embedding_size: The embedding size for the token embedder.
        embedding_dimension: Internal hidden dimension embedder size.
    """

    def __init__(
            self,

            # embedding parameters
            vocab_size = 31748,
            maximum_position_embeddings = 31748,
            token_type_embedding_size = 10,
    ) -> None:
        """Initializes class with the embedding layer configuration parameters.

        """
        
        super().__init__()
        
        # embedding parameters
        self.vocab_size = vocab_size
        self.maximum_position_embeddings = maximum_position_embeddings
        self.token_type_embedding_size = token_type_embedding_size
        self.embedding_dimension = self.hidden_dimension



class FNetEncoderParams(Params):
    """Class to store the FNet Encoder layer parameters.

    Attributes:
        fourier_layer_norm: Dimension used by Layer norm in Fourier layer.
        fnet_layer1_input_dimension: Input dimension for first linear layer.
        fnet_layer1_output_dimension: Output dimension for first linear layer.
        fnet_layer2_input_dimension: Input dimension for second linear layer.
        fnet_layer2_output_dimension: Output dimension for second linear layer.
        fnet_layer_norm_dimension: Dimension used by Layer norm.
    """

    def __init__(self) -> None:

        super().__init__()

        # fourier layer params
        self.fourier_layer_norm = self.hidden_dimension

        # FNet layer configuration parameters
        self.fnet_layer1_input_dimension = self.hidden_dimension
        self.fnet_layer1_output_dimension = self.hidden_dimension
        self.fnet_layer2_input_dimension = self.hidden_dimension
        self.fnet_layer2_output_dimension = self.hidden_dimension
        self.fnet_layer_norm_dimension = self.hidden_dimension




class PoolerParams(Params):
    """Class to store the Pooler layer parameters.

    Attributes:
        pooler_linear_input_dimensions: Input dimension of linear layer.
        pooler_linear_output_dimension: Number of output classes for pooler layer.
    """

    def __init__(
            self,

            # pooler parameters
            pooler_linear_num_classes = 5
    ) -> None:
        """Initializes class with the pooler layer configuration parameters.

        """
        
        super().__init__()

        # pooler parameters
        self.pooler_linear_input_dimensions = self.hidden_dimension
        self.pooler_linear_output_dimension = pooler_linear_num_classes



class Config():
    """Class to store the configuration parameters of the model.

    Attributes:
        embedding_params: Parameters of the Embedding layer.
        fnet_params: Parameteres of the FNet layer.
        pooler_params: Parameters of the Pooler layer.
    """

    def __init__(self):
        """Initializes class with all configuration parameters.

        """

        self.embedding_params = EmbeddingParams()
        self.fnetencoder_params = FNetEncoderParams()
        self.pooler_params = PoolerParams()

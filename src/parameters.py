class Params():
    """Class to store the configuration parameters of the model.

    Attributes:
        layer_norm_epsilon: A value added to denominator of nn.LayerNorm for numerical stability.
        dropout_rate:  Probability of an element to be zeroed.
        number_of_layers: Number of FNetLayer blocks in FNetEncoder.
        vocab_size: Size of the base vocabulary.
        maximum_position_embeddings: The embedding size for the position embedder.
        token_type_embedding_size: The embedding size for the token embedder.
        embedding_dimension: Internal hidden dimension embedder size.
        pooler_linear_num_classes: Number of output classes for pooler layer.
    """

    def __init__(
        self,

        # general parameters
        layer_norm_epsilon = 1e-12,
        dropout_rate = 0.2,
        number_of_layers = 2,

        # embedding parameters
        vocab_size = 31748,
        maximum_position_embeddings = 31748,
        token_type_embedding_size = 10,
        embedding_dimension = 3,

        # pooler parameters
        pooler_linear_num_classes = 5
    ) -> None:
        """Initializes Params class with all model configuration parameters.

        """
        
        # general parameters
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout_rate = dropout_rate
        self.number_of_layers = number_of_layers

        # embedding parameters
        self.vocab_size = vocab_size
        self.maximum_position_embeddings = maximum_position_embeddings
        self.token_type_embedding_size = token_type_embedding_size
        self.embedding_dimension = embedding_dimension

        # pooler parameters
        self.pooler_linear_num_classes = pooler_linear_num_classes

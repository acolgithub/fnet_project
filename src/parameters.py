# class to hold model parameters
class Params():

    def __init__(
        self,

        # general parameters
        norm_epsilon = 1e-12,
        dropout_rate = 0.2,
        number_of_layers = 2,

        # embedding parameters
        vocab_size = 100000,
        maximum_position_embeddings = 100000,
        token_type_embedding_hidden_size = 10,
        embedding_dimension = 3,

        # pooler parameters
        pooler_linear_num_classes = 5
    ) -> None:
        
        # general parameters
        self.norm_epsilon = norm_epsilon
        self.dropout_rate = dropout_rate
        self.number_of_layers = number_of_layers

        # embedding parameters
        self.vocab_size = vocab_size
        self.maximum_position_embeddings = maximum_position_embeddings
        self.token_type_embedding_hidden_size = token_type_embedding_hidden_size
        self.embedding_dimension = embedding_dimension

        # pooler parameters
        self.pooler_linear_num_classes = pooler_linear_num_classes

# class to hold model parameters
class Params():

    def __init__(
        self,
        input_size = 10,
        num_embeddings = 10,
        embedding_dim = 10,
        input = 10,
        num_classes = 2,
        dropout_rate = 0.2,
        N = 1
    ) -> None:
        self.input_size = input_size
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.input = input
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.N = N
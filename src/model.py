import math
import torch
import torch.nn as nn

from .parameters import Config
from .layers import Embedder, FNetEncoder, Pooler



class FNet(nn.Module):
    """Class to store the layers of the FNet architecture.

    Attributes:
        embedder: Embedder for the FNet architecture.
        fnet_encoder: FNet blocks for the FNet architecture.
        pooler: Pooler layer for the FNet architecture.
    """

    def __init__(self, config: Config):
        """Initializes FNet model.

        Args:
            config: Configuration parameters of the model.
        """

        super().__init__()
        self.embedder = Embedder(config)
        self.fnet_encoder = FNetEncoder(config)
        self.pooler = Pooler(config)


    def forward(
        self,
        input_ids: torch.tensor,
        token_type_ids: torch.tensor
    ):
        """Defines forward pass for the FNet architecture.

        Args:
            input_ids: Torch tensor containing input ids from tokenizer.
            token_type_ids: Torch tensor containing token type ids from tokenizer.
        """

        output = self.embedder(input_ids, token_type_ids)
        output = self.fnet_encoder(output)
        output = self.pooler(output)
        return output
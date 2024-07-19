import torch
import torch.nn as nn
import numpy as np

from .parameters import Params
from .layers import Embedder, FNetEncoder, Pooler



class FNet(nn.Module):

    def __init__(self, params: Params):
        super().__init__()
        self.embedder = Embedder(params)
        self.fnet_encoder = FNetEncoder(params)
        self.pooler = Pooler(params)


    def forward(
        self,
        input_ids: torch.tensor,
        token_type_ids: torch.tensor,
        params: Params
    ):

        output = self.embedder(input_ids, token_type_ids)
        output = self.fnet_encoder(output, params)
        output = self.pooler(output)
        return output


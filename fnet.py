import torch
import torch.nn as nn
import numpy as np

from src.parameters import Params
from src.layers import Embedder, Fourier, FeedForward, Pooler
from src.preprocessing import preprocess
from src.model import FNet



# initalize model parameters
params = Params()

sentence = "This is a test sequence of words"
input_ids, token_type_ids = preprocess(sentence)

# instantiate embedder and apply to input_ids and token_type_ids
embedder = Embedder(params)
output = embedder(input_ids, token_type_ids)

for _ in range(params.number_of_layers):

    # apply fourier transform layer
    four = Fourier()
    output = four(output, params)

    # apply feed forward layer
    ff = FeedForward(params)
    output = ff(output)

# apply pooler layer
pool = Pooler(params)
pool(output)


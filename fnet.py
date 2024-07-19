import torch
import torch.nn as nn
import numpy as np

from src.parameters import Params
from src.preprocessing import preprocess
from src.model import FNet



# initalize model parameters
params = Params()

sentence = "This is a test sequence of words"
input_ids, token_type_ids = preprocess(sentence)

# apply fnet model
model = FNet(params)
model(input_ids, token_type_ids, params)


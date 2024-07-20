import math
import torch
import torch.nn as nn
from transformers import BertTokenizer

from src.parameters import Params
from src.preprocessing import preprocess
from src.model import FNet



if __name__ == "__main__":
    
    # initalize model parameters
    params = Params()

    # create test sentence and preprocess it
    sentence = "This is a test sequence of words"
    input_ids, token_type_ids = preprocess(sentence)

    # apply fnet model
    model = FNet(params)
    model(input_ids, token_type_ids, params)

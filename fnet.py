import math
import torch
import torch.nn as nn
from transformers import FNetTokenizer

from src.parameters import Config
from src.preprocessing import preprocess
from src.model import FNet



if __name__ == "__main__":
    
    # initalize model parameters
    config = Config()

    # create test sentence and preprocess it
    sentence = "This is a test sequence of words"
    input_ids, token_type_ids = preprocess(sentence)

    # apply fnet model
    model = FNet(config)
    model(input_ids, token_type_ids)

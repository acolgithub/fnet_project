import torch
import torch.nn as nn

import scipy as sp
import numpy as np

from transformers import BertTokenizer

from src.parameters import Params
from src.layers import Embedder, Fourier, FeedForward, Pooler
from src.model import FNet



# tokenizer sentence
def tokenize(sentence: str):

    # get tokenizer
    tokenizer = BertTokenizer.from_pretrained("DaNLP/da-bert-tone-subjective-objective")

    # tokenized words
    tokenized_sentence = tokenizer(sentence)

    return tokenized_sentence

sentence = "This is a test sequence of words"
tokenized_sentence = tokenize(sentence)

# get input_ids, token_type_ids
input_ids = torch.tensor(tokenized_sentence["input_ids"])
token_type_ids = torch.tensor(tokenized_sentence["token_type_ids"])

# initalize model parameters
params = Params()

# instantiate embedder and apply to input_ids and token_type_ids
embedder = Embedder(params)
embedder_output = embedder(input_ids, token_type_ids)

# apply fourier transform layer
four = Fourier()
fourier_output = four(embedder_output, params)

# apply feed forward layer
ff = FeedForward(params)
ff(fourier_output)



















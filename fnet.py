import torch
import torch.nn as nn

import scipy as sp
import numpy as np

from transformers import BertTokenizer

from src.parameters import Params
from src.layers import Embedder, Fourier, FeedForward, Pooler
from src.model import FNet



# get input_ids and token_type_ids
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

params = Params()

embedder = Embedder(params)
embedder(input_ids, token_type_ids)


# embedder = Embedder(params)
# input = torch.LongTensor([[1, 2, 4, 5],[4, 3, 2, 9]])
# print(embedder(input))

# four = Fourier()
# print(four(np.array([[1,0],[0,1]]), params))




















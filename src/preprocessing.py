import torch
from transformers import BertTokenizer


def preprocess(sentence: str):
    """
    A function to preprocess text before it is passed to the FNet model.

    Args:
        sentence (str): A sentence to tokenize and obtain the corresponding input ids and token type ids.
    """

    # get tokenizer
    tokenizer = BertTokenizer.from_pretrained("DaNLP/da-bert-tone-subjective-objective")

    # tokenize words
    tokenized_sentence = tokenizer(sentence)

    # get input_ids, token_type_ids
    input_ids = torch.tensor(tokenized_sentence["input_ids"])
    token_type_ids = torch.tensor(tokenized_sentence["token_type_ids"])

    return input_ids, token_type_ids
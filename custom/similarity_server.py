import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
from gluonnlp.data import SentencepieceTokenizer
import numpy as np
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer

import sys

model, vocab = get_pytorch_kobert_model()

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

""" input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
model, vocab  = get_pytorch_kobert_model()
sequence_output, pooled_output = model(input_ids, input_mask, token_type_ids) """


sp = SentencepieceTokenizer(tokenizer)

print(sys.argv)
tokens1 = sp(sys.argv[1])
tokens2 = sp(sys.argv[2])

ids1 = []
for token in tokens1:
    ids1.append(vocab[token])

ids2 = []
for token in tokens2:
    ids2.append(vocab[token])

max_seq_length = 50

mask = [1 for i in range(len(ids1))]
input_mask1 = [1] * len(ids1)
padding1 = [0]*(max_seq_length - len(ids1))
ids1 += padding1
input_mask1 += padding1

input_mask2 = [1] * len(ids2)
padding2 = [0]*(max_seq_length - len(ids2))
ids2 += padding2
input_mask2 += padding2

ids = torch.LongTensor([ids1, ids2])
masks = torch.LongTensor([input_mask1, input_mask2])

sequence_output, pooled_output = model(ids, masks)

cos_fn = nn.CosineSimilarity(dim=0, eps=1e-6)
sim_t = cos_fn(pooled_output[0], pooled_output[1])

print(float(sim_t))

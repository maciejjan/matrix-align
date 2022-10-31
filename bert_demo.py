import torch
from transformers import DistilBertTokenizer, DistilBertModel
from matrix_align import similarity

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

sentences = [
    'The quick brown fox jumps over the lazy dog.',
    'A black cat crossed the street.',
    'Let me tell you a story about a swift red squirrel hopping over a sleepy coyote.',
]

def embed(encoded_input, tokenizer, model):
    y = model(**encoded_input)['last_hidden_state']
    return torch.divide(y[0].T, torch.linalg.norm(y[0], axis=1)).T

inputs = [tokenizer(s, return_tensors='pt') for s in sentences]
embs = [embed(inp, tokenizer, model) for inp in inputs]
xb = torch.tensor([0] + [e.shape[0] for e in embs]).cumsum(0)
x = torch.vstack(embs) 

tokens = []
for inp in inputs:
    tokens.extend(tokenizer.convert_ids_to_tokens(inp['input_ids'][0]))

# The first sentence in the list will be compared to the others.
sim_raw, a, w = similarity(x[:xb[1]], x[xb[1]:], xb[1:]-xb[1], threshold=0.7, return_alignments=True)
for i in range(a.shape[0]):
    if torch.isin(i+xb[1], xb):
        print('---')                   # sentence boundary
    if a[i] > -1:
        # show the aligned token pair and their similarity
        print(tokens[xb[1]+i], tokens[a[i]], float(w[i]))


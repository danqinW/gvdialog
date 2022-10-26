import itertools
import torch

def zero_padding(l, fillvalue=0, batch_first=False):
    res = list(itertools.zip_longest(*l, fillvalue=fillvalue))
    if batch_first:
        res = list(zip(*res))
    return res

def binary_mask(l):
    m = []
    for s in l:
        m.append(list(map(lambda x: 0 if x == 0 else 1, s)))
    return m

def generate_pad_mask(input_var):
    '''
    input_var: [seq_len, batch_size]
    '''
    seq_len = input_var.size(0)
    pad_mask = (input_var.T == 0).unsqueeze(1)
    return pad_mask.expand(-1, seq_len, -1)

def generate_square_mask(seq_len):
    trg_range = torch.arange(0, seq_len)
    trg_seq_mask = trg_range.repeat(seq_len, 1) > trg_range.view(-1, 1)
    return trg_seq_mask
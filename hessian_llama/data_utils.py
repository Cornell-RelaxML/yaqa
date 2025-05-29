import random

import torch
from lightning.pytorch.utilities.seed import isolate_rng


class DataLoader():

    def __init__(self, dataset, tok, bs, ctx_size):
        self.dataset = dataset  #iter(dataset.to_iterable_dataset())
        self.tok = tok
        self.bs = bs
        self.ctx_size = ctx_size
        self.cache = None

    def next(self):
        ids = torch.zeros(self.bs, self.ctx_size, dtype=torch.int64)
        for i in range(self.bs):
            while True:
                seq = next(self.dataset)
                seq = self.tok(seq['text'],
                               return_tensors='pt')['input_ids'][0]
                if len(seq) >= self.ctx_size:
                    break
            st = random.randint(0, len(seq) - self.ctx_size)
            ids[i] = seq[st:st + self.ctx_size]
        return ids


class FullCtx(torch.utils.data.Dataset):

    def __init__(self, dataset, tok, ctx_size):
        self.dataset = dataset
        self.tok = tok
        self.ctx_size = ctx_size

    def __len__(self):
        # doesn't matter
        return 1 << 31

    def __getitem__(self, idx):
        while True:
            seq = next(self.dataset)
            seq = self.tok(seq['text'], return_tensors='pt')['input_ids'][0]
            if len(seq) >= self.ctx_size:
                break
        st = seq.sum().int().item() % max(1, (len(seq) - self.ctx_size))
        seq = seq[st:st + self.ctx_size]
        return seq

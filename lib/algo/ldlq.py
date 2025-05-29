import copy
import os

import glog
import torch
from tqdm import tqdm

from lib import utils

_PERMUTE = torch.arange(256).reshape(2, 8, 2, 4, 2).permute(1, 3, 2, 0,
                                                            4).flatten()
_INV_PERMUTE = torch.zeros(256, dtype=torch.int64)
_INV_PERMUTE[_PERMUTE] = torch.arange(256)


def LDLQ_2hess(W, Lin, Lout, td_x, td_y, V, cb, for_kernel=True):
    if for_kernel:
        assert td_x == td_y == 16
    m, n = W.shape
    hatW = torch.zeros_like(W)
    Qidxs = torch.zeros(m, n // V, dtype=cb.idx_dtype, device=W.device)
    assert m % td_x == 0 and n % td_y == 0 and td_y % V == 0
    starts = [
        *[(m // td_x - i - 1, n // td_y - 1) for i in range(m // td_x)],
        *[(0, n // td_y - i - 1) for i in range(n // td_y)]
    ]

    for i in tqdm(range(m // td_x + n // td_y)):
        target = []
        target_idx = []
        start = starts[i]
        jmax = max(start[0], start[1])
        jm, jn = start
        while 0 <= jm < m // td_x and 0 <= jn < n // td_y:
            thing = W[jm*td_x:(jm+1)*td_x, jn*td_y:(jn+1)*td_y] + (
                Lout[jm*td_x:, jm*td_x:(jm+1)*td_x].T @ (W[jm*td_x:, jn*td_y:] - hatW[jm*td_x:, jn*td_y:]) @ Lin[jn*td_y:, jn*td_y:(jn+1)*td_y] + \
                Lout[jm*td_x:, jm*td_x:(jm+1)*td_x].T @ (W[jm*td_x:, jn*td_y:(jn+1)*td_y] - hatW[jm*td_x:, jn*td_y:(jn+1)*td_y]) + \
                (W[jm*td_x:(jm+1)*td_x, jn*td_y:] - hatW[jm*td_x:(jm+1)*td_x, jn*td_y:]) @ Lin[jn*td_y:, jn*td_y:(jn+1)*td_y])
            target.append(thing)
            target_idx.append((jm, jn))
            jm += 1
            jn -= 1

        target = torch.stack(target, dim=0).reshape(-1, td_x * td_y)
        if for_kernel:
            qtarget, targetidx = cb.quantize(target[..., _PERMUTE])
            qtarget = qtarget[..., _INV_PERMUTE].reshape(-1, td_x, td_y)
        else:
            qtarget, targetidx = cb.quantize(target)
            qtarget = qtarget.reshape(-1, td_x, td_y)
        targetidx = targetidx.reshape(-1, td_x, td_y // V)

        for j in range(len(target_idx)):
            jm, jn = target_idx[j]
            hatW[jm * td_x:(jm + 1) * td_x,
                 jn * td_y:(jn + 1) * td_y] = qtarget[j]
            Qidxs[jm * td_x:(jm + 1) * td_x,
                  jn * td_y // V:(jn + 1) * td_y // V] = targetidx[j]
    return hatW, Qidxs

# This adapts SparseGPT process: https://github.com/IST-DASLab/sparsegpt
# E. Frantar et al SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot, https://arxiv.org/abs/2301.00774
# portions copyright by the authors licensed under the Apache License 2.0


import torch
import os
from contextlib import contextmanager
import warnings
import math


class SparseGPT:

    def __init__(
            self, 
            linear_module, 
            sparsity,
            prunen=0,
            prunem=0,
            blocksize=128,
            percdamp=.01,

    ):
        assert isinstance(linear_module, torch.nn.Linear)

        self.linear_module = linear_module
        self.dev = self.linear_module.weight.device
        self.rows = linear_module.weight.shape[0]
        self.columns = linear_module.weight.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.block_size = blocksize
        self.sparsity = sparsity
        self.perdamp = percdamp
        self.prunen = prunen
        self.prunem = prunem

        def collect_input_stats(self, _1, inp, _2):
            inp = inp[0].detach()
            self.last_inp = inp
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            tmp = inp.shape[0]
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
            self.H *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp
            inp = math.sqrt(2 / self.nsamples) * inp.float()
            self.H += inp.matmul(inp.t())

        def sparsify(self):
            W = self.linear_module.weight.detach().to(dtype=torch.float, copy=True)

            H = self.H
            del self.H
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0


            Losses = torch.zeros_like(W)
            Q = torch.zeros_like(W)

            damp = self.percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=self.dev)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H

            mask = None

            for i1 in range(0, self.columns, self.blocksize):
                i2 = min(i1 + self.blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]


                if prunen == 0: 
                    if mask is not None:
                        mask1 = mask[:, i1:i2]
                    else:
                        tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                        thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                        mask1 = tmp <= thresh
                else:
                    mask1 = torch.zeros_like(W1) == 1

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if prunen != 0 and i % prunem == 0:
                        tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                        mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                    q = w.clone()
                    q[mask1[:, i]] = 0


                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d ** 2

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1
           
                Q[:, i1:i2] = Q1
                Losses += torch.sum(Losses1, 1) / 2

                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            pruned_weights = Q.reshape(self.linear_module.weight.shape).to(
                          self.linear_module.weight.data.dtype
            )

            self.linear_module.weight.data = pruned_weights
            del pruned_weights
            error = torch.sum(Losses).item()

            return error


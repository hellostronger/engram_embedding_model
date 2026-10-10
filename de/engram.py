"""
engram.py — the Engram fusion module, adapted for domain embeddings.

Faithful to DeepSeek Engram Sec 2.3/2.4:
    k_t = W_K e_t,  v_t = W_V e_t
    alpha_t = sigmoid( RMSNorm(h_t)^T RMSNorm(k_t) / sqrt(d) )
    y_t     = SiLU(Conv1D(RMSNorm(alpha*v))) + (alpha*v)
    H_t    <- H_t + y_t

Adaptations for Embedding + dynamic CRUD:
  * e_t is now the concatenation of Layer-A (registry slots matched at position t) and
    Layer-B (hashed N-grams), so an operator-written term and a trained statistical
    pattern enter through the *same* gate and projection.
  * W_V has NO bias and slot 0 of the registry is a frozen NULL row => when nothing
    matches, e_t = 0 and y_t = 0 exactly. The backbone is then bit-exact unchanged.
  * Multi-branch fusion (Engram Sec 2.4) is supported: one shared table + one shared W_V,
    M branch-specific W_K^(m). For a plain transformer encoder M == 1.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseCausalConv(nn.Module):
    """Short depthwise causal conv (Engram Eq. 5): kernel w, dilation = max N-gram order.

    Zero-initialised so that at step 0 the module is the identity (paper Sec 4.1:
    "convolution parameters are initialized to zero to strictly preserve the identity
    mapping at the start of training").
    """

    def __init__(self, dim: int, kernel_size: int = 4, dilation: int = 1):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, groups=dim, bias=False,
            padding=(kernel_size - 1) * dilation, dilation=dilation,
        )
        nn.init.zeros_(self.conv.weight)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B, T, D)
        T = x.shape[1]
        y = self.conv(x.transpose(1, 2))[..., :T]
        return self.act(y.transpose(1, 2))


class EngramFusion(nn.Module):
    """Fuse one memory vector into a hidden state at a single position."""

    def __init__(
        self,
        d_mem: int,
        d_model: int,
        n_branches: int = 1,
        kernel_size: int = 4,
        dilation: int = 3,
        gate_bias_init: float = -2.0,
        gate_sqrt: bool = False,
    ):
        super().__init__()
        self.d_mem = d_mem
        self.d_model = d_model
        self.n_branches = n_branches
        self.gate_sqrt = gate_sqrt

        # shared Value projection, NO bias (see NULL-inertness note above)
        self.value_proj = nn.Linear(d_mem, d_model, bias=False)
        # branch-specific Key projections (Engram Sec 2.4)
        self.key_projs = nn.ModuleList(
            [nn.Linear(d_mem, d_model, bias=False) for _ in range(n_branches)]
        )
        self.norm_q = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(n_branches)])
        self.norm_k = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(n_branches)])
        # learnable gate bias: when e_t == 0 the logit reduces to this value, so the
        # network can learn to close the gate on "nothing retrieved" / collisions.
        self.gate_bias = nn.Parameter(torch.full((n_branches,), float(gate_bias_init)))
        self.norm_v = nn.RMSNorm(d_model)
        self.conv = DepthwiseCausalConv(d_model, kernel_size=kernel_size, dilation=dilation)

    def forward(
        self,
        hidden: torch.Tensor,      # (B, T, D)   or (B, T, M, D)
        e_mem: torch.Tensor,       # (B, T, d_mem)
    ):
        if hidden.dim() == 3:
            h = hidden.unsqueeze(2)                       # (B, T, 1, D)
            squeeze = True
        else:
            h = hidden                                   # (B, T, M, D)
            squeeze = False
        B, T, M, D = h.shape
        assert M == self.n_branches, f"branch mismatch {M} vs {self.n_branches}"

        k = torch.stack([self.key_projs[m](e_mem) for m in range(M)], dim=2)  # (B,T,M,D)
        logit = torch.stack(
            [
                (self.norm_q[m](h[:, :, m]) * self.norm_k[m](k[:, :, m])).sum(-1) / math.sqrt(D)
                for m in range(M)
            ],
            dim=2,
        )                                                                        # (B,T,M)
        logit = logit + self.gate_bias.view(1, 1, M)
        if self.gate_sqrt:  # signed sqrt variance reduction, as in the official demo
            logit = logit.abs().clamp_min(1e-6).sqrt() * logit.sign()
        alpha = torch.sigmoid(logit).unsqueeze(-1)                               # (B,T,M,1)

        v = self.value_proj(e_mem).unsqueeze(2)                                  # (B,T,1,D)
        vt = alpha * v                                                           # (B,T,M,D)

        # depthwise conv over the (single-branch) value stream, broadcast to branches
        conv_in = vt[:, :, 0]                                                    # (B,T,D)
        conv_out = self.conv(self.norm_v(conv_in)).unsqueeze(2)                 # (B,T,1,D)
        y = conv_out + vt                                                        # (B,T,M,D)

        out = y.squeeze(2) if squeeze else y
        return out, alpha.squeeze(3) if squeeze else alpha.squeeze(-1)


class EngramTableHead(nn.Module):
    """One Engram injection site: hashed N-gram tables + fusion.

    Owns its own ``nn.Embedding`` tables (one per (order, head) as in Engram Eq. 1-2),
    concatenated into e_t. Registry vectors are supplied by the caller and concatenated
    on top, so a single fusion module serves both memory layers.
    """

    def __init__(
        self,
        table_sizes: list[list[int]],
        d_head: int,
        d_model: int,
        registry_dim: int,
        n_branches: int = 1,
        kernel_size: int = 4,
        dilation: int = 3,
        gate_bias_init: float = -2.0,
        gate_sqrt: bool = False,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.d_head = d_head
        self.registry_dim = registry_dim
        offsets, total = [], 0
        self.tables = nn.ModuleList()
        for sizes in table_sizes:
            for size in sizes:
                t = nn.Embedding(size, d_head)
                nn.init.normal_(t.weight, mean=0.0, std=init_std)
                self.tables.append(t)
                offsets.append(total)
                total += size
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.num_tables = len(self.tables)

        d_mem = self.num_tables * d_head + registry_dim
        self.d_mem = d_mem
        self.fusion = EngramFusion(
            d_mem=d_mem, d_model=d_model, n_branches=n_branches,
            kernel_size=kernel_size, dilation=dilation,
            gate_bias_init=gate_bias_init, gate_sqrt=gate_sqrt,
        )

    def lookup(self, hash_ids: torch.Tensor) -> torch.Tensor:
        """hash_ids: (B, T, num_tables) -> (B, T, num_tables*d_head).

        Each (order, head) pair has its OWN prime-sized table, and hash_ids[..., i] is
        already reduced modulo that table's size, so NO offset is added here (offsets
        are only meaningful if the tables were merged into one flat nn.Embedding).
        """
        chunks = [self.tables[i](hash_ids[..., i]) for i in range(self.num_tables)]
        return torch.cat(chunks, dim=-1)

    def forward(self, hidden, hash_ids, registry_vecs):
        e_mem = self.lookup(hash_ids)
        if registry_vecs is not None:
            e_mem = torch.cat([e_mem, registry_vecs], dim=-1)
        return self.fusion(hidden, e_mem)
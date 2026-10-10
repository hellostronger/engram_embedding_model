"""
losses.py — contrastive objectives for embedding training (replaces Engram's next-token CE).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(
    q: torch.Tensor,
    d: torch.Tensor,
    temperature: float = 0.02,
    in_batch_negatives: bool = True,
) -> torch.Tensor:
    """Symmetric InfoNCE. q/d are assumed L2-normalized.

    in_batch_negatives=True treats all other elements of the batch as negatives.
    """
    logits = (q @ d.t()) / temperature
    if not in_batch_negatives:
        logits = logits.fill_diagonal_(-1e4)
    labels = torch.arange(q.shape[0], device=q.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def info_nce_with_hard_negatives(
    q: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    temperature: float = 0.02,
) -> torch.Tensor:
    """InfoNCE where the candidate pool is {all in-batch positives} U {explicit negatives}.

    neg: (B, K, D). Returns the query->doc direction only (hard negatives are typically
    doc-side, matching BGE-style training).
    """
    B = q.shape[0]
    pos_logits = (q @ pos.t()) / temperature                      # (B,B)
    neg_logits = torch.einsum("bd,bkd->bk", q, neg) / temperature  # (B,K)
    logits = torch.cat([pos_logits, neg_logits], dim=1)            # (B, B+K)
    labels = torch.arange(B, device=q.device)
    return F.cross_entropy(logits, labels)
"""
evaluation.py — retrieval metrics + memory-controlled evaluation.

Metrics: Recall@k, MRR, nDCG@10 (as used in the plan). The evaluate() helper can toggle
model.memory_enabled to reproduce Engram Sec 6.3's memory-suppression sensitivity test.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


def _rank_of_positive(sim: torch.Tensor, pos_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Return the 0-based rank of each query's gold document.

    sim: (n_queries, n_docs). If pos_idx is given (n_queries,), it holds the gold column
    per query; otherwise the gold is assumed to be on the diagonal (1:1 case).
    """
    if pos_idx is None:
        gold = sim.diag().unsqueeze(1)
    else:
        gold = sim.gather(1, pos_idx.view(-1, 1))
    return (sim > gold).sum(dim=1)


def retrieval_metrics(sim: torch.Tensor, pos_idx: Optional[torch.Tensor] = None,
                      ks: Sequence[int] = (1, 5, 10)) -> Dict[str, float]:
    """Recall@k / MRR / nDCG@k over a full sim matrix.

    NOTE: nDCG uses the *ideal* DCG of a single relevant document at rank 1, i.e. IDCG
    is 1/log2(2) = 1.0, giving nDCG@k = 1/log2(rank+2). Using the query count for IDCG
    (as a naive implementation does) silently rescales the metric and is wrong.
    """
    ranks = _rank_of_positive(sim, pos_idx)
    return retrieval_metrics_from_scores(
        -ranks.float().unsqueeze(1), gold_col=0, ks=ks
    )


def retrieval_metrics_from_scores(scores: torch.Tensor, gold_col: int = 0,
                                  ks: Sequence[int] = (1, 5, 10),
                                  n_relevant: int = 1) -> Dict[str, float]:
    """scores: (nq, n_cand) higher = better; gold at column gold_col.

    nDCG@k uses the standard ideal ranking of `n_relevant` relevant docs at ranks 1..k,
    truncated at min(k, n_relevant).
    """
    gold = scores[:, gold_col: gold_col + 1]
    ranks = (scores > gold).sum(dim=1)
    out = {}
    for k in ks:
        out[f"recall@{k}"] = (ranks < k).float().mean().item() * 100
    out["mrr"] = (1.0 / (ranks.float() + 1)).mean().item() * 100

    k = 10
    dcg = (1.0 / torch.log2(ranks.float() + 2.0)) * (ranks < k)
    ideal_n = max(1, min(n_relevant, k))
    idcg = float(sum(1.0 / torch.log2(r + 2.0) for r in range(ideal_n)))
    out["ndcg@10"] = (dcg.sum(dim=0) / idcg).mean().item() * 100
    return out


@torch.no_grad()
def evaluate_retrieval(
    model,
    corpus_texts: List[str],
    query_texts: List[str],
    positive_idx: List[int],
    batch_size: int = 64,
    max_len: int = 256,
    device: Optional[torch.device] = None,
    suppress_memory: Optional[bool] = None,
) -> Dict[str, float]:
    """Encode corpus+queries, compute cosine sim, report retrieval metrics.

    query_texts[i] is relevant to corpus_texts[positive_idx[i]].
    suppress_memory: force memory on/off (Sec 6.3 ablation); None = current setting.
    """
    tok = model.tokenizer if hasattr(model, "tokenizer") else None
    if tok is None:
        raise RuntimeError("attach model.tokenizer before calling evaluate_retrieval")

    prev = model.memory_enabled
    if suppress_memory is not None:
        model.memory_enabled = suppress_memory and bool(model.layer_ids)
    model.eval()
    device = device or model.device

    def enc(texts):
        all_ids = []
        chunks = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            e = tok(batch, padding=True, truncation=True, max_length=max_len,
                    add_special_tokens=True)
            all_ids.extend(e["input_ids"])
            chunks.append((e["input_ids"], e["attention_mask"]))
        return all_ids

    # Build tensors in chunks (matcher needs CPU numpy) then encode on device.
    corpus_emb = []
    query_emb = []
    for i in range(0, len(corpus_texts), batch_size):
        batch = corpus_texts[i: i + batch_size]
        e = tok(batch, padding=True, truncation=True, max_length=max_len,
                add_special_tokens=True)
        ids = e["input_ids"]
        slots = model.prepare_registry_slots(ids).to(device)
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        mask_t = torch.tensor(e["attention_mask"], dtype=torch.long, device=device)
        corpus_emb.append(model(ids_t, mask_t, slots).float().cpu())
    for i in range(0, len(query_texts), batch_size):
        batch = query_texts[i: i + batch_size]
        e = tok(batch, padding=True, truncation=True, max_length=max_len,
                add_special_tokens=True)
        ids = e["input_ids"]
        slots = model.prepare_registry_slots(ids).to(device)
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        mask_t = torch.tensor(e["attention_mask"], dtype=torch.long, device=device)
        query_emb.append(model(ids_t, mask_t, slots).float().cpu())

    C = F.normalize(torch.cat(corpus_emb), dim=-1)
    Q = F.normalize(torch.cat(query_emb), dim=-1)
    sim = Q @ C.t()
    # restrict scoring to the relevant positives only (queries correspond 1:1 to pos docs)
    idx = torch.tensor(positive_idx, dtype=torch.long)
    sub = sim[torch.arange(sim.shape[0]), idx].unsqueeze(1)  # pos score per query
    # build a small matrix [query, its positive + 9 random distractors] for ranking
    g = torch.Generator().manual_seed(0)
    distract = torch.stack([
        torch.randperm(C.shape[0], generator=g) for _ in range(sim.shape[0])
    ])  # (nq, N)
    cand = torch.cat([idx.unsqueeze(1), distract], dim=1)    # (nq, N+1)
    cand_sim = torch.gather(sim, 1, cand)
    gold_col = 0  # our positive is placed first
    out = retrieval_metrics_from_scores(cand_sim, gold_col)
    model.memory_enabled = prev
    return out


def retrieval_metrics_from_scores(scores: torch.Tensor, gold_col: int = 0,
                                  ks: Sequence[int] = (1, 5, 10)) -> Dict[str, float]:
    """scores: (nq, n_cand) higher = better; gold at column gold_col."""
    gold = scores[:, gold_col: gold_col + 1]
    ranks = (scores > gold).sum(dim=1)
    out = {}
    for k in ks:
        out[f"recall@{k}"] = (ranks < k).float().mean().item() * 100
    out["mrr"] = (1.0 / (ranks.float() + 1)).mean().item() * 100
    dcg = 1.0 / torch.log2(ranks.float() + 2.0)
    idcg = 1.0 / torch.log2(torch.arange(ranks.shape[0], device=scores.device).float() + 2.0)
    out["ndcg@10"] = (dcg / idcg.clamp_min(1e-9)).mean().item() * 100
    return out
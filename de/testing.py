"""
testing.py — offline helpers so the full pipeline (incl. the A/B/C/D
experiment) can be exercised with NO network access.

The real deployment uses a downloaded backbone (e.g. BAAI/bge-small-zh-v1.5).
These helpers build a tiny randomly-initialised BERT + a character-level
tokenizer whose vocabulary is DERIVED FROM THE ACTUAL DATA, which guarantees
every domain term tokenises into real tokens (never [UNK]) and therefore that
registry phrase-matching actually fires.
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Sequence

import numpy as np
import torch


# --------------------------------------------------------------------------- #
def build_tiny_backbone(vocab_size: int, hidden: int = 64, layers: int = 4,
                          heads: int = 4, max_pos: int = 128):
    """A randomly-initialised BERT (no download)."""
    from transformers import BertConfig, BertModel

    torch.manual_seed(0)
    cfg = BertConfig(
        vocab_size=vocab_size, hidden_size=hidden, num_hidden_layers=layers,
        num_attention_heads=heads, intermediate_size=hidden * 4,
        max_position_embeddings=max_pos,
    )
    return BertModel(cfg)


def build_char_tokenizer(texts: Sequence[str], tmpdir: str = None):
    """Character-level BertTokenizer whose vocab covers every char in `texts`.

    Guarantees no [UNK] for in-domain text, which is what makes registry
    phrase matching meaningful in offline tests.
    """
    from transformers import BertTokenizer

    chars = set()
    for t in texts:
        chars.update(t)
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    # deterministic order
    vocab = specials + sorted(chars)
    tmpdir = tmpdir or tempfile.mkdtemp(prefix="de_vocab_")
    vp = os.path.join(tmpdir, "vocab.txt")
    with open(vp, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab))
    return BertTokenizer(vocab_file=vp, do_lower_case=True), len(vocab)


# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_texts(model, tokenizer, texts: List[str], max_len: int = 64,
                   batch_size: int = 64, device=None) -> torch.Tensor:
    """Encode a list of strings into L2-normalised embeddings."""
    device = device or model.device
    model.eval()
    outs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i: i + batch_size]
        enc = tokenizer(
            chunk, padding=True, truncation=True, max_length=max_len,
            add_special_tokens=True, return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        slots = model.prepare_registry_slots(ids.tolist()).to(device)
        emb = model(ids, mask, slots)
        outs.append(emb.float().cpu())
    return torch.cat(outs, dim=0)


def load_rows(path: str) -> List[dict]:
    import json
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def eval_rows(model, tokenizer, rows: List[dict], max_len: int = 64,
                batch_size: int = 64, device=None) -> dict:
    """Per-row mini-corpus retrieval: each row has its own corpus + gold index.

    Returns Recall@{1,3,5,10}, MRR and nDCG@10 averaged over rows. This is
    the metric shape used by the synthetic data and the cold-start test.
    """
    queries = [r["query"] for r in rows]
    gold_idx = [int(r["gold"]) for r in rows]
    flat_docs, doc_row = [], []
    # gold_flat[ri] = flat index (into flat_docs) of row ri's gold document
    gold_flat = []
    for ri, r in enumerate(rows):
        base = len(flat_docs)
        for d in r["corpus"]:
            flat_docs.append(d)
            doc_row.append(ri)
        if not (0 <= gold_idx[ri] < len(r["corpus"])):
            raise ValueError(f"row {ri}: gold={gold_idx[ri]} out of range "
                             f"(corpus has {len(r['corpus'])} docs)")
        gold_flat.append(base + gold_idx[ri])

    q_emb = encode_texts(model, tokenizer, queries, max_len, batch_size, device)
    d_emb = encode_texts(model, tokenizer, flat_docs, max_len, batch_size, device)
    q_emb = torch.nn.functional.normalize(q_emb, dim=-1)
    d_emb = torch.nn.functional.normalize(d_emb, dim=-1)

    ks = (1, 3, 5, 10)
    recall = {k: 0.0 for k in ks}
    mrr, ndcg = 0.0, 0.0
    n = len(rows)
    for ri in range(n):
        didx = [j for j, rr in enumerate(doc_row) if rr == ri]
        scores = (q_emb[ri] @ d_emb[didx].t())          # (n_docs,)
        order = torch.argsort(scores, descending=True)
        # position of this row's gold doc within its own candidate list
        local_gold = didx.index(gold_flat[ri])
        rank = int((order == local_gold).nonzero()[0]) + 1
        for k in ks:
            recall[k] += (rank <= k)
        mrr += 1.0 / rank
        ndcg += 1.0 / np.log2(rank + 1)
    for k in ks:
        recall[k] = recall[k] / n * 100
    return {"recall@1": recall[1], "recall@3": recall[3],
            "recall@5": recall[5], "recall@10": recall[10],
            "mrr": mrr / n * 100, "ndcg@10": ndcg / n * 100}
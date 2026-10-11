"""
train_baseline.py — train on a real dataset and report whether the loss ACTUALLY moves.

Motivation: every previous number came from 2 epochs where loss went 1.358 -> 1.318
(a 0.04 drop is not training). This script makes the loss curve the primary output,
so "did we actually train?" is answered before any downstream comparison is trusted.

It also sweeps a couple of learning rates, because --lr 1e-3 has never been validated.

Usage:
    python train_baseline.py --data data/real/eval/t2 --epochs 10
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from de.config import default_config
from de.data import DECollator, TripletDataset
from de.losses import info_nce, info_nce_with_hard_negatives
from de.model import DomainEngramEncoder
from de.testing import eval_rows, load_rows
from de.train import make_optimizer, set_seed

BACKBONE = "BAAI/bge-small-zh-v1.5"


@torch.no_grad()
def eval_asym(model, tok, rows, query_len, doc_len, batch_size, device):
    """Evaluate with SEPARATE lengths for queries and documents.

    Real retrieval has short queries (T2: mean 10.7 tokens) and long documents
    (mean 404). Using one max_len for both either truncates the docs (max_len=64
    keeps only 19% of the gold document) or wastes compute padding queries.
    """
    from de.testing import encode_texts
    import numpy as np

    model.eval()
    queries = [r["query"] for r in rows]
    flat_docs, doc_row, gold_flat = [], [], []
    for ri, r in enumerate(rows):
        base = len(flat_docs)
        for d in r["corpus"]:
            flat_docs.append(d)
            doc_row.append(ri)
        gold_flat.append(base + int(r["gold"]))

    q_emb = encode_texts(model, tok, queries, query_len, batch_size, device)
    d_emb = encode_texts(model, tok, flat_docs, doc_len, batch_size, device)
    q_emb = torch.nn.functional.normalize(q_emb, dim=-1)
    d_emb = torch.nn.functional.normalize(d_emb, dim=-1)

    ks = (1, 3, 5, 10)
    recall = {k: 0.0 for k in ks}
    mrr, ndcg = 0.0, 0.0
    n = len(rows)
    for ri in range(n):
        didx = [j for j, rr in enumerate(doc_row) if rr == ri]
        scores = q_emb[ri] @ d_emb[didx].t()
        order = torch.argsort(scores, descending=True)
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


def build(backbone, tok, layer_ids, registry_slots, table_size, d_head,
          n_head, d_registry):
    cfg = default_config(
        backbone_name=backbone, layer_ids=layer_ids,
        max_registry_slots=registry_slots, table_size_per_ngram=table_size,
        d_head=d_head, d_registry=d_registry, n_head_per_ngram=n_head,
        freeze_backbone=True, pooling="cls",
    )
    return DomainEngramEncoder(cfg, backbone=backbone, tokenizer=tok)


def run(lr, epochs, train_rows, eval_rows_, backbone, tok, device, bs, max_len,
        layer_ids=(1, 3), out_prefix="", query_len=None, eval_every=1,
        use_hard_neg=True, temperature=0.02, n_hard_neg=3):
    set_seed(0)
    m = build(copy.deepcopy(backbone), tok, list(layer_ids), 2048, 1024, 8, 2, 8)
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    collate = DECollator(tok, m.matcher, max_len=max_len,
                         lookup_table=m.compressed.lookup_table,
                         query_max_len=query_len)
    # TripletDataset(train_rows) passes dicts straight through and keeps ALL
    # negatives; use from_jsonl only if we need truncation semantics. Build
    # explicitly so hard-negative count is deterministic and visible.
    from de.data import Triplet
    ds = TripletDataset([
        Triplet(query=r["query"], positive=r["positive"],
                negatives=list(r.get("negatives") or [])[:n_hard_neg])
        for r in train_rows
    ])
    print(f"    train triplets={len(ds)}  hard_neg_per_sample={n_hard_neg}  "
          f"trainable={n_tr:,}", flush=True)
    loader = DataLoader(ds, batch_size=bs, shuffle=True,
                        collate_fn=collate, drop_last=True, num_workers=0)
    opt = make_optimizer(m, base_lr=lr, memory_lr_mult=5.0, warmup=0)

    m.eval()
    before = eval_asym(m, tok, eval_rows_, query_len, max_len, 128, device)
    # Print the untrained baseline IMMEDIATELY. It used to be printed only in the
    # end-of-run summary, so an interrupted run could not be interpreted at all
    # (E-010a got killed at epoch 2 and had nothing to compare against).
    print(f"  [baseline / memory OFF] R@1 {before['recall@1']:5.1f}  "
          f"nDCG {before['ndcg@10']:5.1f}  MRR {before['mrr']:5.1f}", flush=True)

    m.train()
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        tot, n, n_hn = 0.0, 0, 0
        for b in loader:
            q, d = b["query"], b["positive"]
            qe = m(q["input_ids"], q["attention_mask"], q["registry_slots"])
            de = m(d["input_ids"], d["attention_mask"], d["registry_slots"])
            # E-007c: the old version always took the `else` branch and silently threw
            # away every hard negative the collator had already built. Use them.
            if use_hard_neg and "negative" in b and b["negative"]["input_ids"].shape[0] > 0:
                neg = b["negative"]
                ne = m(neg["input_ids"], neg["attention_mask"], neg["registry_slots"])
                K = ne.shape[0] // qe.shape[0]
                ne = ne.view(qe.shape[0], K, -1)
                loss = info_nce_with_hard_negatives(qe, de, ne, temperature)
                n_hn += 1
            else:
                loss = info_nce(qe, de, temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            n += 1
        row = {"epoch": ep, "loss": tot / max(n, 1), "elapsed": time.time() - t0,
               "hard_neg_steps": n_hn}
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            m.eval()
            met = eval_asym(m, tok, eval_rows_, query_len, max_len, 128, device)
            m.train()
            row.update({"ndcg": met["ndcg@10"], "r1": met["recall@1"]})
            extra = (f"  R@1 {met['recall@1']:5.1f} ({met['recall@1']-before['recall@1']:+5.1f})"
                     f"  nDCG {met['ndcg@10']:5.1f} ({met['ndcg@10']-before['ndcg@10']:+5.1f})")
        else:
            extra = ""
        hist.append(row)
        delta = ""
        if "ndcg" in row:
            delta = (f"   vs base: R@1 {row['r1']-before['recall@1']:+.1f}  "
                     f"nDCG {row['ndcg']-before['ndcg@10']:+.1f}")
        print(f"    ep{ep:02d}  loss {row['loss']:.4f}{extra}  "
              f"[{row['elapsed']:.0f}s]{delta}", flush=True)

    after = hist[-1]
    print(f"  [lr={lr:g}] loss {hist[0]['loss']:.4f} -> {after['loss']:.4f} "
          f"(drop {hist[0]['loss']-after['loss']:+.4f})   "
          f"nDCG {before['ndcg@10']:.1f} -> {after.get('ndcg', float('nan')):.1f} "
          f"   [{time.time()-t0:.0f}s]", flush=True)
    return hist, before, after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/eval/t2")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lrs", default="1e-3,3e-4")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512, help="doc length")
    ap.add_argument("--query-len", type=int, default=64,
                    help="query length (T2 queries: mean 10.7 tok)")
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--n-hard-neg", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.02)
    ap.add_argument("--no-hard-neg", action="store_true",
                    help="reproduce the E-007 defect (ignore collated hard negatives)")
    ap.add_argument("--leak-check", action="store_true",
                    help="report overlap between train negatives and eval corpus")
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--backbone", default=BACKBONE)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    device = torch.device(a.device)
    train_rows = load_rows(os.path.join(a.data, "train.jsonl"))[: a.n_train]
    ev = load_rows(os.path.join(a.data, "eval.jsonl"))[: a.n_eval]
    print(f"train={len(train_rows)}  eval={len(ev)}  device={device}")
    print(f"lengths: query<={a.query_len}  doc<={a.max_len}  "
          f"bs={a.bs}  eval_every={a.eval_every}")

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.backbone)
    backbone = AutoModel.from_pretrained(a.backbone)

    # ---- E-014 leak check: do train negatives coincide with eval candidates? ----
    if a.leak_check:
        eval_docs = set()
        for r in ev:
            eval_docs.update(r["corpus"])
        train_negs = set()
        for r in train_rows:
            train_negs.update(r.get("negatives") or [])
        train_pos = set(r["positive"] for r in train_rows)
        inter = train_negs & eval_docs
        print(f"[leak] train negatives={len(train_negs)}  eval corpus={len(eval_docs)}")
        print(f"[leak] negatives also present in eval corpus: {len(inter)} "
              f"({100*len(inter)/max(len(train_negs),1):.2f}% of negatives)")
        print(f"[leak] positives also present in eval corpus: {len(train_pos & eval_docs)}")
        print(f"[leak] eval queries also used as train query: "
              f"{len(set(r['query'] for r in train_rows) & set(r['query'] for r in ev))}")

    results = {}
    for lr in [float(x) for x in a.lrs.split(",")]:
        print(f"\n=== lr={lr:g}, {a.epochs} epochs, "
              f"hard_neg={not a.no_hard_neg}, temp={a.temperature} ===", flush=True)
        hist, before, after = run(lr, a.epochs, train_rows, ev, backbone, tok,
                                   device, a.bs, a.max_len,
                                   query_len=a.query_len, eval_every=a.eval_every,
                                   use_hard_neg=not a.no_hard_neg,
                                   temperature=a.temperature,
                                   n_hard_neg=a.n_hard_neg)
        results[f"lr={lr:g}"] = {"before": before, "after": after, "hist": hist,
                                 "hard_neg": not a.no_hard_neg}

    print("\n" + "=" * 70)
    print("SUMMARY: does the loss actually move?")
    print("=" * 70)
    for k, v in results.items():
        h = v["hist"]
        print(f"  {k:10s} hn={str(v['hard_neg']):5s} "
              f"loss {h[0]['loss']:.4f} -> {h[-1]['loss']:.4f} "
              f"(drop {h[0]['loss']-h[-1]['loss']:+.4f})   "
              f"nDCG {v['before']['ndcg@10']:.1f} -> {v['after'].get('ndcg', float('nan')):.1f}   "
              f"[{h[-1]['elapsed']:.0f}s]")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
"""
run_cold_start.py — E-009: the decisive cold-start experiment.

This is the FIRST run in the project that can actually falsify P0
("a term unseen at training time can be made useful by writing it into the
CRUD registry, with no retraining"). Every earlier D-vs-C number came from a
platform that was invalid -- either a randomly-initialised tiny backbone
(E-001/E-002) or a train/eval-leaking dataset (E-005/E-007).

Dataset contract (see build_cold_law.py, verified in E-008):
    train.jsonl       built ONLY from publish_date < T0  (14,034 source rows)
    eval_cold.jsonl   rows with publish_date >= T0 whose title AND first 200
                      chars of content never occur in TRAIN   (1,500 rows)
    cold_start.jsonl  the terms to write at inference time  (1,500 terms)
    -> zero leakage: no eval term appears in train

Groups
    A  baseline        memory OFF
    B  random memory   memory ON, untrained (noise control)
    C  trained memory  memory ON, trained with InfoNCE + hard negatives
    D  C + cold-start  C, then write the held-out terms, NO retraining

Usage:
    python run_cold_start.py --data data/real/eval/law --epochs 10 --lr 1e-3
    python run_cold_start.py --data data/real/eval/law --refine-steps 20 \
        --ridge 1e-2 --device cuda
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
from de.cold_start import cold_start
from de.data import DECollator, Triplet, TripletDataset
from de.losses import info_nce_with_hard_negatives
from de.model import DomainEngramEncoder
from de.testing import load_rows
from de.train import make_optimizer, set_seed
from train_baseline import eval_asym

BACKBONE = "BAAI/bge-small-zh-v1.5"


def build_model(cfg_kw, backbone, tok):
    cfg = default_config(freeze_backbone=True, pooling="cls", **cfg_kw)
    return DomainEngramEncoder(cfg, backbone=backbone, tokenizer=tok)


@torch.no_grad()
def cos_on_off(model, tok, texts, query_len, doc_len, device):
    """cos(emb memory ON, emb memory OFF) -- is the cold-start write a NUDGE or an OVERWRITE?

    E-004 measured 0.894 on fp32/CPU with the real backbone. A value near 0 or
    negative means the write replaced the backbone's answer instead of
    correcting it (the failure mode suspected in E-002).
    """
    from de.testing import encode_texts
    on = copy.deepcopy(model)
    on.memory_enabled = True
    off = copy.deepcopy(model)
    off.memory_enabled = False
    e_on = encode_texts(on, tok, texts, query_len, 128, device)
    e_off = encode_texts(off, tok, texts, query_len, 128, device)
    c = torch.nn.functional.cosine_similarity(e_on, e_off, dim=-1)
    return float(c.mean()), float(c.min())


@torch.no_grad()
def match_rate(model, tok, texts, query_len):
    """Fraction of eval queries whose tokens hit at least one registry slot."""
    import numpy as np
    hit = 0
    lut = model.compressed.lookup_table
    for t in texts:
        ids = tok(t, truncation=True, max_length=query_len)["input_ids"]
        canon = np.asarray([lut[i] for i in ids], dtype=np.int64)
        if int(np.max(model.matcher.match(canon))) > 0:
            hit += 1
    return hit / max(len(texts), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real/eval/law")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--query-len", type=int, default=96)
    ap.add_argument("--doc-len", type=int, default=512)
    ap.add_argument("--n-hard-neg", type=int, default=3)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--backbone", default=BACKBONE)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layer-ids", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--table-size", type=int, default=1024)
    ap.add_argument("--max-ngram", type=int, default=3)
    ap.add_argument("--registry-slots", type=int, default=4096)
    ap.add_argument("--refine-steps", type=int, default=0,
                    help="gradient steps per cold slot (0 = closed-form only)")
    ap.add_argument("--ridge", type=float, default=1e-3,
                    help="closed-form ridge; larger = smaller injection")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    device = torch.device(a.device)
    train_rows = load_rows(os.path.join(a.data, "train.jsonl"))[: a.n_train]
    eval_rows = load_rows(os.path.join(a.data, "eval_cold.jsonl"))[: a.n_eval]
    cold_terms = load_rows(os.path.join(a.data, "cold_start.jsonl"))[: a.n_eval]
    queries = [r["query"] for r in eval_rows]
    print(f"train={len(train_rows)}  eval_cold={len(eval_rows)}  "
          f"cold_terms={len(cold_terms)}  device={device}")
    print(f"layer_ids={a.layer_ids}  table_size={a.table_size}  "
          f"max_ngram={a.max_ngram}  ridge={a.ridge}  refine={a.refine_steps}")

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.backbone)
    backbone = AutoModel.from_pretrained(a.backbone)

    def cfg_kw():
        return dict(backbone_name=a.backbone, layer_ids=list(a.layer_ids),
                    max_registry_slots=a.registry_slots,
                    table_size_per_ngram=a.table_size,
                    max_ngram_size=a.max_ngram,
                    d_head=8, d_registry=8, n_head_per_ngram=2)

    results = {}

    def report(tag, m):
        met = eval_asym(m, tok, eval_rows, a.query_len, a.doc_len, 128, device)
        results[tag] = met
        print(f"  {tag:22s} R@1 {met['recall@1']:5.1f}  R@5 {met['recall@5']:5.1f}  "
              f"nDCG {met['ndcg@10']:5.1f}  MRR {met['mrr']:5.1f}", flush=True)
        return met

    # ---------------- A: memory OFF ---------------- #
    print("\n=== A: baseline (memory OFF) ===", flush=True)
    mA = build_model(cfg_kw(), copy.deepcopy(backbone), tok)
    mA.memory_enabled = False
    mA.eval()
    report("A_baseline", mA)

    # ---------------- B: untrained memory ---------------- #
    if not a.skip_b:
        print("\n=== B: random memory (noise control) ===", flush=True)
        mB = build_model(cfg_kw(), copy.deepcopy(backbone), tok)
        mB.eval()
        report("B_random", mB)

    # ---------------- C: trained memory ---------------- #
    print(f"\n=== C: trained memory ({a.epochs} epochs) ===", flush=True)
    set_seed(0)
    mC = build_model(cfg_kw(), copy.deepcopy(backbone), tok)
    n_tr = sum(p.numel() for p in mC.parameters() if p.requires_grad)
    collate = DECollator(tok, mC.matcher, max_len=a.doc_len,
                         lookup_table=mC.compressed.lookup_table,
                         query_max_len=a.query_len)
    ds = TripletDataset([
        Triplet(query=r["query"], positive=r["positive"],
                negatives=list(r.get("negatives") or [])[: a.n_hard_neg])
        for r in train_rows
    ])
    loader = DataLoader(ds, batch_size=a.bs, shuffle=True, collate_fn=collate,
                        drop_last=True, num_workers=0)
    opt = make_optimizer(mC, base_lr=a.lr, memory_lr_mult=5.0, warmup=0)
    print(f"    trainable={n_tr:,}  triplets={len(ds)}  hard_neg={a.n_hard_neg}",
          flush=True)

    mC.train()
    t0 = time.time()
    hist = []
    for ep in range(a.epochs):
        tot, n = 0.0, 0
        for b in loader:
            q, d = b["query"], b["positive"]
            qe = mC(q["input_ids"], q["attention_mask"], q["registry_slots"])
            de = mC(d["input_ids"], d["attention_mask"], d["registry_slots"])
            if "negative" in b and b["negative"]["input_ids"].shape[0] > 0:
                neg = b["negative"]
                ne = mC(neg["input_ids"], neg["attention_mask"], neg["registry_slots"])
                K = ne.shape[0] // qe.shape[0]
                ne = ne.view(qe.shape[0], K, -1)
                loss = info_nce_with_hard_negatives(qe, de, ne, 0.02)
            else:
                from de.losses import info_nce
                loss = info_nce(qe, de, 0.02)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mC.trainable_parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            n += 1
        hist.append({"epoch": ep, "loss": tot / max(n, 1),
                     "elapsed": time.time() - t0})
        print(f"    ep{ep:02d}  loss {hist[-1]['loss']:.4f}  "
              f"[{hist[-1]['elapsed']:.0f}s]", flush=True)
    mC.eval()
    c_met = report("C_trained", mC)

    # ---------------- D: + cold start, NO retraining ---------------- #
    print(f"\n=== D: C + cold-start writes (NO retraining) ===", flush=True)
    probe = queries[:200]
    print(f"    match rate BEFORE write: {100*match_rate(mC, tok, probe, a.query_len):.1f}%",
          flush=True)

    t1 = time.time()
    n_written = 0
    for r in cold_terms:
        if cold_start(mC, tok, r["term"], r.get("contexts", []),
                      refine_steps=a.refine_steps, ridge=a.ridge, device=device):
            n_written += 1
    mC.sync_matcher()
    print(f"    wrote {n_written}/{len(cold_terms)} terms in {time.time()-t1:.1f}s",
          flush=True)
    print(f"    match rate AFTER  write: "
          f"{100*match_rate(mC, tok, probe, a.query_len):.1f}%", flush=True)

    c_mean, c_min = cos_on_off(mC, tok, probe, a.query_len, a.doc_len, device)
    print(f"    cos(emb_ON, emb_OFF): mean={c_mean:.4f} min={c_min:.4f}  "
          f"(E-004 fp32/CPU reference: 0.8942)", flush=True)
    results["cos_on_off"] = {"mean": c_mean, "min": c_min}

    mC.eval()
    d_met = report("D_coldstart", mC)

    # ---------------- verdict ---------------- #
    print("\n" + "=" * 72)
    print("E-009 VERDICT -- cold-start effectiveness (D - C, unseen terms)")
    print("=" * 72)
    dn = d_met["ndcg@10"] - c_met["ndcg@10"]
    dr = d_met["recall@1"] - c_met["recall@1"]
    print(f"  D - C   nDCG {dn:+.2f}   R@1 {dr:+.2f}")
    if dn >= 2.0:
        verdict = "PASS  (>= +2 nDCG, the target set in docs/TODO.md)"
    elif dn > 0:
        verdict = "WEAK  (positive but below +2)"
    else:
        verdict = "FAIL  (no gain -- P0 is falsified on this platform)"
    print(f"  verdict: {verdict}")
    print(f"  cos(ON,OFF) mean={c_mean:.4f}  -> " +
          ("nudge (healthy)" if c_mean > 0.7 else
           "SUSPECT overwrite" if c_mean < 0.3 else "borderline"))
    print("\n  If FAIL, next checks (H1/H2/H3/H4 already excluded in E-004):")
    print("   1) H5 gate magnitude: sweep --refine-steps 0/5/20/50, is D-C monotone?")
    print("   2) sweep --ridge 1e-2/1e-3/1e-4 (larger ridge = smaller injection)")
    print("   3) confirm match rate is ~100% (printed above)")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"config": vars(a), "results": results, "hist": hist,
                       "verdict": verdict, "d_minus_c_ndcg": dn,
                       "d_minus_c_r1": dr}, f, indent=2, ensure_ascii=False)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
"""
run_abcd.py — the four-group control experiment that validates the design.

  A  baseline        : frozen backbone, memory OFF
  B  random memory   : frozen backbone, memory ON but UNTRAINED (noise control)
  C  trained memory  : frozen backbone, memory ON and trained with InfoNCE
  D  C + cold-start  : C, then write held-out terms into the registry with NO retraining

Two evaluation sets:
  * seen terms   (data/synth/eval.jsonl)    -> compares A / B / C
  * unseen terms (data/synth/eval_cold.jsonl)-> compares C / D  (the headline claim)

Runs fully offline with a tiny random BERT + a data-derived char tokenizer.
On the GB10 box, swap in the real backbone via --backbone.

Usage:
    python run_abcd.py --data data/synth --epochs 2
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
from de.data import TripletDataset, DECollator
from de.model import DomainEngramEncoder
from de.testing import (build_tiny_backbone, build_char_tokenizer,
                        load_rows, eval_rows)
from de.losses import info_nce
from de.train import make_optimizer


# --------------------------------------------------------------------------- #
def fresh_model(cfg, backbone, tokenizer, registry=None):
    return DomainEngramEncoder(cfg, registry=registry,
                               compressed_tokenizer=__import__("de.model", fromlist=["CompressedTokenizer"]).CompressedTokenizer(tokenizer),
                               backbone=backbone, tokenizer=tokenizer)


def train(model, tok, ds, epochs, bs, lr, max_len, device, seed=0):
    from de.train import set_seed
    set_seed(seed)
    collate = DECollator(tok, model.matcher, max_len=max_len,
                         lookup_table=model.compressed.lookup_table)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate,
                        drop_last=True, num_workers=0)
    opt = make_optimizer(model, base_lr=lr, memory_lr_mult=5.0, warmup=0)
    model.train()
    for ep in range(epochs):
        tot, n = 0.0, 0
        for b in loader:
            q, d = b["query"], b["positive"]
            qe = model(q["input_ids"], q["attention_mask"], q["registry_slots"])
            de = model(d["input_ids"], d["attention_mask"], d["registry_slots"])
            loss = info_nce(qe, de)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); n += 1
        print(f"    epoch {ep}: loss {tot/max(n,1):.4f}")
    return model


def report(name, metrics):
    r = "  ".join(f"{k}={v:.1f}" for k, v in metrics.items())
    print(f"  {name:28s} {r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--backbone", default=None, help="HF id; omit for offline tiny BERT")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-cold", action="store_true")
    a = ap.parse_args()

    device = torch.device(a.device)
    train_rows = load_rows(os.path.join(a.data, "train.jsonl"))
    eval_seen = load_rows(os.path.join(a.data, "eval.jsonl"))
    eval_cold = load_rows(os.path.join(a.data, "eval_cold.jsonl"))
    cold_terms = load_rows(os.path.join(a.data, "cold_start.jsonl"))
    print(f"train={len(train_rows)}  eval_seen={len(eval_seen)}  "
          f"eval_cold={len(eval_cold)}  cold_terms={len(cold_terms)}")

    # ---- offline tokenizer derived from the data (no [UNK] in-domain) ------ #
    corpus_text = [r["query"] for r in train_rows] + [r["positive"] for r in train_rows]
    corpus_text += [r["query"] for r in eval_seen] + [r["query"] for r in eval_cold]
    tok, vocab_size = build_char_tokenizer(corpus_text)
    print(f"offline char tokenizer: vocab={vocab_size}")

    if a.backbone:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(a.backbone)
        backbone = AutoModel.from_pretrained(a.backbone)
        hidden = backbone.config.hidden_size
    else:
        backbone = build_tiny_backbone(vocab_size)
        hidden = 64
    print(f"backbone hidden={hidden}  device={device}")

    cfg = default_config(
        backbone_name="tiny",
        layer_ids=[1, 3],
        max_registry_slots=2048,
        table_size_per_ngram=1024,
        d_head=8,
        d_registry=8,
        max_matches_per_pos=4,
        n_head_per_ngram=2,
        freeze_backbone=True,
        pooling="cls",
    )
    n_tr = sum(p.numel() for p in backbone.parameters())
    print(f"backbone params={n_tr:,}  EngramConfig={cfg}")

    results = {}

    # ================= GROUP A: baseline, memory OFF ======================= #
    print("\n=== A: baseline (memory OFF) ===")
    mA = DomainEngramEncoder(cfg, backbone=copy.deepcopy(backbone), tokenizer=tok)
    mA.memory_enabled = False
    mA.eval()
    results["A_baseline"] = eval_rows(mA, tok, eval_seen, a.max_len, 128, device)
    results["A_cold"] = eval_rows(mA, tok, eval_cold, a.max_len, 128, device)
    report("A baseline / seen", results["A_baseline"])
    report("A baseline / unseen", results["A_cold"])

    # ================= GROUP B: random memory, UNTRAINED ================== #
    print("\n=== B: memory ON, UNTRAINED (noise control) ===")
    mB = DomainEngramEncoder(cfg, backbone=copy.deepcopy(backbone), tokenizer=tok)
    # memory params are at init; do NOT train. This is the noise control.
    mB.eval()
    results["B_random"] = eval_rows(mB, tok, eval_seen, a.max_len, 128, device)
    report("B random memory / seen", results["B_random"])

    # ================= GROUP C: trained memory ============================ #
    print(f"\n=== C: trained memory ({a.epochs} epochs) ===")
    mC = DomainEngramEncoder(cfg, backbone=copy.deepcopy(backbone), tokenizer=tok)
    t0 = time.time()
    ds = TripletDataset(train_rows)
    train(mC, tok, ds, a.epochs, a.bs, a.lr, a.max_len, device, a.seed)
    print(f"    trained in {time.time()-t0:.1f}s")
    mC.eval()
    results["C_trained_seen"] = eval_rows(mC, tok, eval_seen, a.max_len, 128, device)
    results["C_trained_cold"] = eval_rows(mC, tok, eval_cold, a.max_len, 128, device)
    report("C trained / seen", results["C_trained_seen"])
    report("C trained / unseen", results["C_trained_cold"])

    # ================= GROUP D: C + cold start, NO retraining ============ #
    if not a.skip_cold:
        print("\n=== D: C + cold-start writes (NO retraining) ===")
        t0 = time.time()
        for r in cold_terms:
            # contexts: build from the term itself + a generic template, since the
            # synthetic cold terms come with contexts
            cold_start(mC, tok, r["term"], r.get("contexts", []),
                       refine_steps=0)
        mC.sync_matcher()
        print(f"    wrote {len(cold_terms)} terms in {time.time()-t0:.2f}s")
        mC.eval()
        results["D_coldstart_cold"] = eval_rows(mC, tok, eval_cold, a.max_len, 128, device)
        results["D_coldstart_seen"] = eval_rows(mC, tok, eval_seen, a.max_len, 128, device)
        report("D cold-start / unseen", results["D_coldstart_cold"])
        report("D cold-start / seen", results["D_coldstart_seen"])

    # ================= SUMMARY =========================================== #
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    keys_seen = ["A_baseline", "B_random", "C_trained_seen"]
    keys_cold = ["A_cold", "C_trained_cold", "D_coldstart_cold"]
    print("\nSEEN terms (does training the memory help?):")
    for k in keys_seen:
        if k in results:
            m = results[k]
            print(f"  {k:22s} R@1={m['recall@1']:5.1f}  R@5={m['recall@5']:5.1f}  "
                  f"nDCG={m['ndcg@10']:5.1f}")
    print("\nUNSEEN terms (does cold-start writing help?):")
    for k in keys_cold:
        if k in results:
            m = results[k]
            print(f"  {k:22s} R@1={m['recall@1']:5.1f}  R@5={m['recall@5']:5.1f}  "
                  f"nDCG={m['ndcg@10']:5.1f}")

    if "C_trained_seen" in results and "A_baseline" in results:
        d = results["C_trained_seen"]["ndcg@10"] - results["A_baseline"]["ndcg@10"]
        print(f"\nC - A (seen nDCG): {d:+.2f}  -> memory training "
              f"{'HELPS' if d > 0 else 'does not help'}")
    if "D_coldstart_cold" in results and "C_trained_cold" in results:
        d = results["D_coldstart_cold"]["ndcg@10"] - results["C_trained_cold"]["ndcg@10"]
        print(f"D - C (unseen nDCG): {d:+.2f} -> cold-start "
              f"{'HELPS' if d > 0 else 'does not help'}")

    out = os.path.join(a.data, "abcd_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
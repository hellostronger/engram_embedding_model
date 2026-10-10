"""
diagnose_cold.py — pin down WHY cold-start (group D) yields only +0.06.

Three competing hypotheses:

  H1  the write never lands   : PhraseMatcher does not fire on eval_cold text,
                                so reg_vecs stays zero and D is a literal no-op.
  H2  the gate suppresses it  : match fires, but alpha ~ 0 because W_K's
                                registry columns (and gate_bias) were never
                                trained while the registry stayed empty during C.
  H3  the value is unusable   : match fires and gate opens, but W_V's registry
                                columns are untrained, so alpha*v points nowhere.

Each is measured directly.  No guessing.

Usage:
    python diagnose_cold.py --data data/synth --epochs 2
"""
from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from de.config import default_config
from de.cold_start import cold_start
from de.data import TripletDataset, DECollator
from de.losses import info_nce
from de.model import DomainEngramEncoder
from de.train import make_optimizer, set_seed
from de.testing import (build_tiny_backbone, build_char_tokenizer,
                        load_rows, encode_texts)


def train(model, tok, ds, epochs, bs, lr, max_len, seed=0):
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


def branch_param_names(model):
    """Names of parameters belonging to the REGISTRY branch of the fusion."""
    # W_V and W_K are shared projections; the registry branch is reached through
    # the same modules, so we instead isolate by gradient coverage.
    names = []
    for n, p in model.named_parameters():
        if p.requires_grad and (".engram" in n or "registry" in n or "compressed" in n):
            names.append(n)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    train_rows = load_rows(os.path.join(a.data, "train.jsonl"))
    eval_cold = load_rows(os.path.join(a.data, "eval_cold.jsonl"))
    cold_terms = load_rows(os.path.join(a.data, "cold_start.jsonl"))

    corpus = [r["query"] for r in train_rows] + [r["positive"] for r in train_rows]
    corpus += [r["query"] for r in eval_cold]
    tok, vocab_size = build_char_tokenizer(corpus)
    backbone = build_tiny_backbone(vocab_size)

    cfg = default_config(backbone_name="tiny", layer_ids=[1, 3],
                         max_registry_slots=2048, table_size_per_ngram=1024,
                         d_head=8, d_registry=8, max_matches_per_pos=4,
                         n_head_per_ngram=2, freeze_backbone=True, pooling="cls")

    m = DomainEngramEncoder(cfg, backbone=copy.deepcopy(backbone), tokenizer=tok)
    print(f"\n[training C: {a.epochs} epochs]")
    train(m, tok, TripletDataset(train_rows), a.epochs, a.bs, a.lr, a.max_len, a.seed)
    m.eval()

    # ------------------------------------------------------------------ #
    # Snapshot the memory params BEFORE any cold-start write so we can tell
    # "trained during C" from "still at init".
    # ------------------------------------------------------------------ #
    pre_state = {n: p.detach().clone() for n, p in m.named_parameters()}

    # ---- H1: does the matcher fire at all on the cold eval text? ------ #
    eval_texts = [r["query"] for r in eval_cold]
    n_matched_before = 0
    for t in eval_texts:
        ids = tok(t, truncation=True, max_length=a.max_len)["input_ids"]
        canon = np.asarray([m.compressed.lookup_table[i] for i in ids], dtype=np.int64)
        slots = m.matcher.match(canon)
        if int(np.max(slots)) > 0:
            n_matched_before += 1

    print(f"\n[H1] matcher fires on eval_cold BEFORE write: "
          f"{n_matched_before}/{len(eval_texts)} "
          f"({100*n_matched_before/len(eval_texts):.1f}%)  <- expect 0")

    # ---- write the cold terms ---------------------------------------- #
    n_written = 0
    for r in cold_terms:
        if cold_start(m, tok, r["term"], r.get("contexts", []), refine_steps=0):
            n_written += 1
    m.sync_matcher()
    print(f"[D] wrote {n_written}/{len(cold_terms)} cold terms")

    n_matched_after = 0
    for t in eval_texts:
        ids = tok(t, truncation=True, max_length=a.max_len)["input_ids"]
        canon = np.asarray([m.compressed.lookup_table[i] for i in ids], dtype=np.int64)
        slots = m.matcher.match(canon)
        if int(np.max(slots)) > 0:
            n_matched_after += 1
    print(f"[H1] matcher fires on eval_cold AFTER  write: "
          f"{n_matched_after}/{len(eval_texts)} "
          f"({100*n_matched_after/len(eval_texts):.1f}%)")

    if n_matched_after == 0:
        print("\n>>> H1 CONFIRMED: write does not land. Diagnose registry/phrase forms.")
        return

    # ---- H2: what is the gate doing on those matched positions? ------- #
    # Encode with gate returned, memory ON.
    enc_on = encode_texts(m, tok, eval_texts, a.max_len, 128, torch.device("cpu"),
                          return_gate=True)
    emb_on, gates = enc_on
    all_alpha = []
    for idx, g in gates.items():
        all_alpha.append(g.reshape(-1).numpy())
    all_alpha = np.concatenate(all_alpha) if all_alpha else np.zeros(1)
    print(f"\n[H2] gate alpha over ALL positions: "
          f"mean={all_alpha.mean():.4f}  p50={np.percentile(all_alpha,50):.4f}  "
          f"p95={np.percentile(all_alpha,95):.4f}  max={all_alpha.max():.4f}")

    # alpha specifically at matched positions only
    matched_alpha = []
    for t in eval_texts:
        ids = tok(t, truncation=True, max_length=a.max_len)["input_ids"]
        canon = np.asarray([m.compressed.lookup_table[i] for i in ids], dtype=np.int64)
        slots = m.matcher.match(canon)
        pos = np.where(slots.max(axis=-1) > 0)[0]
        if len(pos):
            # gate tensor is (B,T,M); take layer 0, matched rows, max over branches
            g0 = list(gates.values())[0]
            for p_ in pos:
                if p_ < g0.shape[1]:
                    matched_alpha.append(float(g0[0, p_].max()))
    if matched_alpha:
        ma = np.array(matched_alpha)
        print(f"[H2] gate alpha at MATCHED positions (n={len(ma)}): "
              f"mean={ma.mean():.4f}  min={ma.min():.4f}  max={ma.max():.4f}")
        print(f"     -> {'SUPPRESSED' if ma.mean() < 0.05 else 'OPEN'} "
              f"(threshold 0.05)")
    else:
        print("[H2] no matched positions inside gate tensor -> alignment issue")

    # ---- H3: did W_V / W_K registry columns actually train? ----------- #
    print("\n[H3] parameter drift from init during C-training (engram/registry only):")
    drifted, frozen = [], []
    for n, p in m.named_parameters():
        if n not in pre_state:
            continue
        if not (".engram" in n or "registry" in n or "compressed" in n):
            continue
        d = (p.detach() - pre_state[n]).abs().max().item()
        (drifted if d > 1e-8 else frozen).append((n, d))
    for n, d in drifted[:12]:
        print(f"    DRIFT  {d:9.6f}  {n}")
    for n, d in frozen[:12]:
        print(f"    FROZEN {d:9.6f}  {n}")
    print(f"    -> {len(drifted)} drifted, {len(frozen)} frozen")

    # ---- embedding delta: is the write doing ANYTHING? ---------------- #
    m_off = copy.deepcopy(m)
    m_off.memory_enabled = False
    e_on = encode_texts(m, tok, eval_texts, a.max_len, 128, torch.device("cpu"))
    e_off = encode_texts(m_off, tok, eval_texts, a.max_len, 128, torch.device("cpu"))
    delta = (e_on - e_off).norm(dim=-1)
    print(f"\n[DELTA] ||emb(memory ON) - emb(OFF)|| on eval_cold: "
          f"mean={delta.mean():.5f}  max={delta.max():.5f}  "
          f"zero_frac={(delta<1e-8).float().mean():.2%}")
    print("        -> >0 means the write changes output at all")


if __name__ == "__main__":
    main()

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
    ap.add_argument("--backbone", default=None,
                    help="HF id; omit for the offline random tiny BERT")
    a = ap.parse_args()

    train_rows = load_rows(os.path.join(a.data, "train.jsonl"))
    eval_cold = load_rows(os.path.join(a.data, "eval_cold.jsonl"))
    cold_terms = load_rows(os.path.join(a.data, "cold_start.jsonl"))

    corpus = [r["query"] for r in train_rows] + [r["positive"] for r in train_rows]
    corpus += [r["query"] for r in eval_cold]
    if a.backbone:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.backbone)
        backbone = AutoModel.from_pretrained(a.backbone)
        print(f"backbone={a.backbone}  hidden={backbone.config.hidden_size}  "
              f"layers={backbone.config.num_hidden_layers}")
    else:
        tok, vocab_size = build_char_tokenizer(corpus)
        backbone = build_tiny_backbone(vocab_size)
        print("backbone=tiny random BERT (offline)")

    cfg = default_config(backbone_name=a.backbone or "tiny", layer_ids=[1, 3],
                         max_registry_slots=2048, table_size_per_ngram=1024,
                         d_head=8, d_registry=8, max_matches_per_pos=4,
                         n_head_per_ngram=2, freeze_backbone=True, pooling="cls")

    m = DomainEngramEncoder(cfg, backbone=copy.deepcopy(backbone), tokenizer=tok)

    # ------------------------------------------------------------------ #
    # Snapshot params AT INIT, so we can tell "trained during C" from
    # "still at random init". (Snapshotting after training would only measure
    # the cold-start writes -- that is the bug this replaces.)
    # ------------------------------------------------------------------ #
    init_state = {n: p.detach().clone() for n, p in m.named_parameters()}

    print(f"\n[training C: {a.epochs} epochs]")
    train(m, tok, TripletDataset(train_rows), a.epochs, a.bs, a.lr, a.max_len, a.seed)
    m.eval()
    post_c_state = {n: p.detach().clone() for n, p in m.named_parameters()}

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

    # alpha specifically at matched positions only (per-batch, correct alignment)
    B = 128
    matched_alpha = []
    layer0 = min(gates.keys()) if gates else None
    if layer0 is not None:
        g0_all = gates[layer0]                     # (N, T, M)
        pos_all = []
        for t in eval_texts:
            ids = tok(t, truncation=True, max_length=a.max_len)["input_ids"]
            canon = np.asarray([m.compressed.lookup_table[i] for i in ids], dtype=np.int64)
            slots = m.matcher.match(canon)
            pos = np.where(slots.max(axis=-1) > 0)[0]
            pos_all.append(pos)
        for bi in range(0, len(eval_texts), B):
            chunk = eval_texts[bi: bi + B]
            enc = tok(chunk, padding=True, truncation=True, max_length=a.max_len,
                      add_special_tokens=True, return_tensors="pt")
            slot_t = m.prepare_registry_slots(enc["input_ids"].tolist())
            for j in range(len(chunk)):
                g_j = g0_all[bi + j]                # (T, M)
                pos = pos_all[bi + j]
                for p_ in pos:
                    if p_ < g_j.shape[0]:
                        matched_alpha.append(float(g_j[p_].max()))
    if matched_alpha:
        ma = np.array(matched_alpha)
        print(f"[H2] gate alpha at MATCHED positions (n={len(ma)}): "
              f"mean={ma.mean():.4f}  min={ma.min():.4f}  max={ma.max():.4f}")
        print(f"     -> {'SUPPRESSED' if ma.mean() < 0.05 else 'OPEN'} "
              f"(threshold 0.05)")
    else:
        print("[H2] no matched positions inside gate tensor -> alignment issue")

    # ---- H3: did W_V / W_K registry columns actually train? ----------- #
    print("\n[H3] drift from INIT during C-training (engram/registry params):")

    def _drift(a_, b_):
        out = []
        for n, p in m.named_parameters():
            if n not in a_ or n not in b_:
                continue
            if not (n.startswith("engram_head_") or n.startswith("registry_table")):
                continue
            out.append((n, (b_[n] - a_[n]).abs().max().item()))
        return out

    c_drift = _drift(init_state, post_c_state)
    cold_drift = _drift(post_c_state, {n: p.detach() for n, p in m.named_parameters()})
    for n, d in sorted(c_drift, key=lambda r: -r[1]):
        tag = "TRAINED" if d > 1e-8 else "STILL-AT-INIT"
        print(f"    C-epoch {tag:14s} max|d|={d:9.6f}  {n}")
    n_untrained = sum(1 for _, d in c_drift if d <= 1e-8)
    print(f"    -> {len(c_drift)-n_untrained}/{len(c_drift)} engram/registry params "
          f"received gradient during C")
    print("    (a param that never moved has RANDOM W_V/W_K registry columns, "
          "so alpha*v is meaningless no matter how open the gate is)")
    for n, d in sorted(cold_drift, key=lambda r: -r[1])[:5]:
        if d > 1e-8:
            print(f"    cold-start wrote max|d|={d:9.6f}  {n}")

    # ---- embedding delta: is the write doing ANYTHING? ---------------- #
    m_off = copy.deepcopy(m)
    m_off.memory_enabled = False
    e_on = encode_texts(m, tok, eval_texts, a.max_len, 128, torch.device("cpu"))
    e_off = encode_texts(m_off, tok, eval_texts, a.max_len, 128, torch.device("cpu"))
    delta = (e_on - e_off).norm(dim=-1)
    cos = torch.nn.functional.cosine_similarity(e_on, e_off, dim=-1)
    print(f"\n[DELTA] ||emb(memory ON) - emb(OFF)|| on eval_cold: "
          f"mean={delta.mean():.5f}  max={delta.max():.5f}  "
          f"zero_frac={(delta<1e-8).float().mean():.2%}")
    print(f"        (embeddings are L2-normalised, so ||d|| maxes out at 2.0; "
          f"2.0 == opposite vectors)")
    print(f"[ORIENT] cos(emb_ON, emb_OFF) on eval_cold: "
          f"mean={cos.mean():.4f}  p05={cos.kthvalue(max(1,int(0.05*len(cos)))).values:.4f}  "
          f"min={cos.min():.4f}")
    print("        -> cos ~0 means memory wrote an ORTHOGONAL direction and")
    print("           OVERWROTE the backbone instead of nudging it (want ~0.9+)")


if __name__ == "__main__":
    main()

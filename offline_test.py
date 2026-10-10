"""
offline_test.py — validate the Domain Engram implementation WITHOUT network access.

Builds a tiny random-weight BERT + an offline WordPiece tokenizer, then runs the same
correctness battery as smoke_test.py. Use this in CI / sandboxes; use smoke_test.py on
the GB10 box with the real backbone.

Usage:  python offline_test.py
"""

from __future__ import annotations

import os
import tempfile
import sys

import numpy as np
import torch

from de.config import default_config
from de.model import DomainEngramEncoder


def build_tiny_backbone(vocab_size=256, hidden=64, layers=4, heads=4):
    """A randomly-initialised BERT (no download)."""
    from transformers import BertConfig, BertModel
    cfg = BertConfig(
        vocab_size=vocab_size, hidden_size=hidden, num_hidden_layers=layers,
        num_attention_heads=heads, intermediate_size=hidden * 4,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    return BertModel(cfg)


def build_offline_tokenizer(vocab_words, tmpdir):
    """A BertTokenizer built from a local vocab.txt (no network)."""
    from transformers import BertTokenizer
    vocab_path = os.path.join(tmpdir, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for w in vocab_words:
            f.write(w + "\n")
    return BertTokenizer(vocab_file=vocab_path, do_lower_case=True)


def main():
    tmpdir = tempfile.mkdtemp(prefix="de_test_")
    torch.manual_seed(1234)

    # --- vocab: specials + single chars + words + ##pieces ---------------- #
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    chars = list("的一是不了人我在有他这为之大来以个中上们")
    for c in chars:
        words.append(c)
    latin = ["diabetes", "neuro", "##pathy", "treat", "##ment", "##s", "##ing",
             "retina", "##l", "clinical", "trial", "cell", "##osis", "##itis",
             "bank", "loan", "plan", "the", "of", "and", "is", "a", "to"]
    words += latin
    for i in range(200):
        words.append(f"tok{i}")

    tokenizer = build_offline_tokenizer(words, tmpdir)
    print(f"tiny vocab: {len(tokenizer)}")

    cfg = default_config(
        backbone_name="tiny-bert",
        layer_ids=[1, 2],
        max_registry_slots=64,
        table_size_per_ngram=256,
        n_head_per_ngram=4,
        d_head=8,
        d_registry=8,
        max_ngram_size=3,
        max_matches_per_pos=4,
        freeze_backbone=True,
        pooling="cls",
    )
    model = DomainEngramEncoder(cfg, backbone=build_tiny_backbone(),
                                tokenizer=tokenizer)
    model.eval()

    n_train = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"params: trainable {n_train:,} / total {n_total:,} ({100*n_train/n_total:.1f}%)")
    ok = True

    # ---- 1. deterministic hashing -------------------------------------- #
    ids = torch.randint(5, 40, (2, 12))
    canon = model._canonical_ids(ids)
    h1 = model.hasher.hash_layer(canon, 1)
    h2 = model.hasher.hash_layer(canon, 1)
    det = torch.equal(h1, h2)
    # each head has its OWN prime-sized table, so validate index i against table i
    head = model.engram_head_1
    in_range = all(
        bool((h1[..., i] < head.tables[i].num_embeddings).all()
             and (h1[..., i] >= 0).all())
        for i in range(head.num_tables)
    )
    print(f"[1] hash deterministic={det} per_table_in_range={in_range} "
          f"shape={tuple(h1.shape)} n_tables={head.num_tables}")
    ok &= det and in_range

    # different layer -> different addresses (per-layer multipliers)
    h_layer2 = model.hasher.hash_layer(canon, 2)
    print(f"    layer1 vs layer2 differ: {not torch.equal(h1, h_layer2)}")

    # ---- 2. tokenizer compression ------------------------------------- #
    n_raw = len(tokenizer)
    n_comp = len(model.compressed)
    print(f"[2] compression: {n_raw} -> {n_comp} ({100*(1-n_comp/n_raw):.1f}% reduction)")
    ok &= n_comp <= n_raw

    # ---- 3. registry CRUD --------------------------------------------- #
    reg = model.registry
    p1 = tokenizer("neuropathy", add_special_tokens=False).input_ids
    p2 = tokenizer("retina", add_special_tokens=False).input_ids
    unk = tokenizer.unk_token_id
    # guard: these phrases must actually be tokenized, not collapsed to [UNK]
    assert unk not in p1, f"'neuropathy' collapsed to [UNK]; test vocab is incomplete"
    assert unk not in p2, f"'retina' collapsed to [UNK]; test vocab is incomplete"
    print(f"    phrase tokenization: neuropathy={p1} retina={p2}")
    c1 = tuple(model.compressed(torch.tensor([p1]))[0].tolist())
    c2 = tuple(model.compressed(torch.tensor([p2]))[0].tolist())
    s1 = reg.add(c1, "neuropathy")
    s2 = reg.add(c2, "retina")
    assert s1 != s2 and s1 != 0 and s2 != 0
    assert reg.get_slot(c1) == s1
    n_before = len(reg)
    deleted = reg.delete(c1)
    after_del = reg.get_slot(c1)
    s3 = reg.add(c1, "neuropathy")
    reused = (s3 == s1)  # free-list pops smallest free => reuses the tombstoned slot
    print(f"[3] CRUD: s1={s1} s2={s2} delete={deleted} after_del={after_del} "
          f"re-add s3={s3} (reused s1: {reused}) active={len(reg)}")
    ok &= deleted and after_del is None

    # ---- 3b. tuple-hash == rolling-hash (exact verification must work) ------ #
    reg2 = model.registry
    test_phrase = tuple([5, 17, 42, 99])
    th = model.matcher._hash_tuple(test_phrase)
    arr = np.zeros(64, dtype=np.int64)
    arr[10:14] = test_phrase
    rh = model.matcher._rolling_hash(arr, 4)[13]
    hash_agree = int(th) == int(rh)
    print(f"[3b] tuple-hash == rolling-hash: {hash_agree} "
          f"({int(th)} vs {int(rh)}) -> {'PASS' if hash_agree else 'FAIL'}")
    ok &= hash_agree

    # ---- 4. NULL inertness (critical correctness guarantee) ----------- #
    text_ids = [tokenizer("diabetes neuropathy", add_special_tokens=False).input_ids] * 2
    padded = torch.zeros(2, 8, dtype=torch.long)
    mask = torch.zeros(2, 8, dtype=torch.long)
    for i, t in enumerate(text_ids):
        t = t[:8]
        padded[i, :len(t)] = torch.tensor(t)
        mask[i, :len(t)] = 1

    slots_all = model.prepare_registry_slots(padded.tolist())
    reg_vecs = model._registry_vecs(slots_all)
    hit_slots = int((slots_all > 0).sum())
    nonzero_vecs = int((reg_vecs.abs().sum(dim=-1) > 0).any(dim=-1).sum())
    print(f"[4] registry path: matched slots={hit_slots} positions_with_vec={nonzero_vecs}")
    ok &= hit_slots > 0 and nonzero_vecs > 0

    # empty registry -> registry vectors exactly zero
    saved = dict(reg.entries)
    reg.entries.clear()
    reg._free = list(range(reg.max_slots - 1, 0, -1))
    reg.version += 1
    model.sync_matcher()
    slots_empty = model.prepare_registry_slots(padded.tolist())
    vec_empty = model._registry_vecs(slots_empty)
    inert = bool(vec_empty.abs().sum() == 0) and bool((slots_empty == 0).all())
    print(f"[5] NULL inertness (empty registry -> zero vectors, zero slots): "
          f"sum={float(vec_empty.abs().sum())} all_zero_slots={bool((slots_empty==0).all())} "
          f"-> {'PASS' if inert else 'FAIL'}")
    ok &= inert
    reg.entries.update(saved)
    reg.version += 1
    model.sync_matcher()

    # ---- 6. end-to-end forward + gate --------------------------------- #
    model.memory_enabled = True
    e, gates = model(padded, mask, model.prepare_registry_slots(padded.tolist()),
                     return_gate=True)
    g = gates[1]
    gate_ok = 0.0 <= float(g.min()) and float(g.max()) <= 1.0
    print(f"[6] forward emb={tuple(e.shape)} gate[1] mean={float(g.mean()):.4f} "
          f"range=[{float(g.min()):.4f},{float(g.max()):.4f}] -> "
          f"{'PASS' if gate_ok else 'FAIL'}")
    ok &= gate_ok

    # ---- 7. memory suppression (paper Sec 6.3 ablation switch) -------- #
    model.memory_enabled = False
    e_nomem = model(padded, mask, model.prepare_registry_slots(padded.tolist()))
    model.memory_enabled = True
    diff = float((e - e_nomem).abs().mean())
    print(f"[7] memory on vs off mean|delta|={diff:.6f} (should be > 0 after memory is warm)")
    ok &= diff > 0

    # ---- 8. cold start (closed-form write) --------------------------- #
    from de.cold_start import cold_start
    contexts = [
        "neuropathy treatment",
        "the neuropathy treat plan",
        "neuropathy of the retina",
    ]
    slot = cold_start(model, tokenizer, "neuropathy", contexts)
    vec = model.registry_table.emb.weight[slot].detach()
    nz = float(vec.abs().sum())
    print(f"[8] cold-start slot={slot} |e|={float(vec.norm()):.5f} sum|e|={nz:.5f} "
          f"-> {'PASS' if nz > 1e-9 else 'FAIL'}")
    ok &= nz > 1e-9

    # ---- 9. training step reduces loss -------------------------------- #
    from de.data import Triplet, TripletDataset, DECollator
    from de.losses import info_nce
    from de.train import make_optimizer
    from torch.utils.data import DataLoader

    def enc(s):
        return tokenizer(s, return_tensors="pt", padding=True, truncation=True,
                         max_length=16).input_ids[0].tolist()

    triples = [
        Triplet("neuropathy", "neuropathy treatment", ["retina trial"]),
        Triplet("diabetes", "diabetes treat", ["bank loan"]),
    ]
    ds = TripletDataset(triples)
    collate = DECollator(tokenizer, model.matcher, max_len=16,
                          lookup_table=model.compressed.lookup_table)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)
    opt = make_optimizer(model, base_lr=1e-2, memory_lr_mult=5.0, warmup=0)
    model.train()
    losses = []
    for _ in range(30):
        for batch in loader:
            q, d = batch["query"], batch["positive"]
            qe = model(q["input_ids"], q["attention_mask"], q["registry_slots"])
            de_ = model(d["input_ids"], d["attention_mask"], d["registry_slots"])
            loss = info_nce(qe, de_)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
    print(f"[9] train loss {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"-> {'PASS' if losses[-1] < losses[0] else 'WARN'}")
    ok &= losses[-1] < losses[0]

    print("\n" + ("=" * 60))
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
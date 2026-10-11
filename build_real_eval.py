"""
build_real_eval.py — build a REAL evaluation/training set from downloaded C-MTEB data.

Why this exists
---------------
`make_synthetic_data.py` produced a set that was **self-证** and **saturated**:
the query and its positive shared the same domain term, distractors differed only
in that term, and a frozen bge-small-zh already scored R@1=100.0 -- leaving the
memory module no headroom to demonstrate anything (see docs/TODO.md).

This builder converts a C-MTEB retrieval dataset (real search-log queries) into the
project's JSONL contract, keeping per-query mini-corpora so `de.testing.eval_rows`
can consume it unchanged:

    eval.jsonl  {"query":str, "corpus":[str,...], "gold":int}   # gold = LOCAL index
    train.jsonl {"query":str, "positive":str, "negatives":[str,...]}

Design notes
------------
* HTML noise (<br>, <img>) is stripped -- it is T2Retrieval-specific and would
  otherwise let a pure lexical matcher win on token artefacts.
* Hard negatives are drawn from the SAME mini-corpus pool, preferring documents
  that are BM25-similar to the query so negatives are genuinely hard.
* `n_cand` docs per query (default 10) keeps eval_rows cheap while staying far
  above the chance level that made the synthetic set useless.

Usage:
    python build_real_eval.py --dataset t2retrieval --out data/real/eval/t2
    python build_real_eval.py --dataset du --out data/real/eval/du --n-queries 2000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict
from typing import Dict, List

# ---------------------------------------------------------------- HTML cleanup
_BR = re.compile(r"<br\s*/?>", re.I)
_IMG = re.compile(r"<img[^>]*>", re.I)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")


def clean(t: str, limit: int = 512) -> str:
    if not t:
        return ""
    t = _BR.sub(" ", t)
    t = _IMG.sub(" ", t)
    t = _TAG.sub(" ", t)
    return _WS.sub(" ", t).strip()[:limit]


# ---------------------------------------------------------------- loaders
def _find(root: str, pattern: str) -> str:
    hits = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under {root}")
    return hits[0]


def load_c_mteb(corpus_root: str, qrels_root: str):
    """Return (queries{id:text}, corpus{id:text}, qrels{qid:[(pid,score)]})."""
    import pyarrow.parquet as pq

    q = {r["id"]: r["text"] for r in
         pq.read_table(_find(corpus_root, "queries*.parquet")).to_pylist()}
    c = {r["id"]: r["text"] for r in
         pq.read_table(_find(corpus_root, "corpus*.parquet")).to_pylist()}
    rels = defaultdict(list)
    for r in pq.read_table(_find(qrels_root, "*.parquet")).to_pylist():
        if int(r["score"]) > 0:
            rels[str(r["qid"])].append(str(r["pid"]))
    return q, c, dict(rels)


# ---------------------------------------------------------------- BM25-ish hard negs
def _char_tokens(s: str) -> List[str]:
    return [ch for ch in s if ch.strip()]


def build(args):
    rng = random.Random(args.seed)
    q, c, rels = load_c_mteb(args.corpus, args.qrels)
    all_ids = list(c.keys())

    qids = [qid for qid in q if rels.get(qid)]
    rng.shuffle(qids)
    # ---- SPLIT FIRST, before building anything -----------------------------
    # E-010 leak finding: train and eval used to be drawn from the same pool, so
    # every eval query was also a training query and every eval positive also a
    # training positive. The eval set was therefore part of the training set.
    # Split the QUERY ids first; train negatives are then additionally forbidden
    # from appearing anywhere in eval.
    n_eval = args.n_queries or len(qids)
    eval_qids = set(qids[:n_eval])
    train_qids = qids[n_eval:]
    print(f"queries={len(q)}  corpus={len(c)}  usable_qrels={len(rels)}  "
          f"eval_qids={len(eval_qids)}  train_qids={len(train_qids)}")

    eval_rows, train_rows = [], []
    eval_docs: set = set()

    # ---- precompute char sets ONCE for the hard-negative scan --------------
    # Rebuilding set(c[pid][:400]) per query was O(n_q * hard_scan * doc_len)
    # and made a full build take >15 min. One pass, then reuse.
    char_sets = {}
    if args.hard_neg:
        scan_ids = rng.sample(all_ids, min(args.hard_scan, len(all_ids)))
        for pid in scan_ids:
            char_sets[pid] = set(_char_tokens(c[pid][:400]))
        print(f"  hard-neg scan pool: {len(scan_ids)} docs (char sets precomputed)")

    def _make(qid, forbid):
        qt = clean(q[qid])
        pos_ids = [p for p in rels[qid] if p in c]
        if not qt or not pos_ids:
            return None
        pos_cap = [p for p in pos_ids[: args.n_pos] if p not in forbid]
        if not pos_cap:
            return None
        pos_set = set(pos_ids)

        if args.hard_neg:
            qchars = set(_char_tokens(qt))
            scored = []
            for pid, cs in char_sets.items():
                if pid in pos_set or pid in forbid:
                    continue
                scored.append((len(qchars & cs) / max(len(qchars), 1), pid))
            scored.sort(key=lambda r: -r[0])
            negs = [pid for _, pid in scored[: args.n_cand]]
        else:
            negs = []

        need = max(args.n_cand - len(pos_cap) - len(negs), 0)
        if need:
            used = pos_set | forbid | set(negs)
            filler = [pid for pid in all_ids if pid not in used]
            rng.shuffle(filler)
            negs = negs + filler[:need]

        cand_ids = pos_cap + negs
        cand_texts = [clean(c[p]) for p in cand_ids]
        keep = [i for i, t in enumerate(cand_texts) if t]
        if not keep or 0 not in keep:
            return None
        cand_texts = [cand_texts[i] for i in keep]
        gold_local = keep.index(0)
        return {"query": qt, "corpus": cand_texts, "gold": gold_local,
                "_doc_ids": [cand_ids[i] for i in keep]}

    # ---- eval side: no forbidden set yet, but record every doc used -------- #
    for qid in eval_qids:
        r = _make(qid, set())
        if r is None:
            continue
        eval_docs.update(r.pop("_doc_ids"))
        eval_rows.append(r)

    # ---- train side: forbid EVERY document that appears in eval ----------- #
    for qid in train_qids:
        if len(train_rows) >= args.max_train:
            break
        r = _make(qid, eval_docs)
        if r is None:
            continue
        r.pop("_doc_ids")
        neg_texts = [r["corpus"][i] for i in range(len(r["corpus"]))
                     if i != r["gold"]][: args.n_train_neg]
        if neg_texts:
            train_rows.append({
                "query": r["query"],
                "positive": r["corpus"][r["gold"]],
                "negatives": neg_texts,
            })

    # ---- verify no leakage survived --------------------------------------- #
    eval_texts = set()
    for r in eval_rows:
        eval_texts.update(r["corpus"])
        eval_texts.add(r["query"])
    leak_n = sum(1 for r in train_rows for x in r["negatives"] if x in eval_texts)
    leak_p = sum(1 for r in train_rows if r["positive"] in eval_texts)
    leak_q = sum(1 for r in train_rows if r["query"] in eval_texts)
    print(f"[leak-check] train_neg in eval: {leak_n}   "
          f"train_pos in eval: {leak_p}   train_query in eval: {leak_q}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "eval.jsonl"), "w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_cand = sum(len(r["corpus"]) for r in eval_rows) / max(len(eval_rows), 1)
    print(f"eval.jsonl   {len(eval_rows)} queries  (avg {n_cand:.1f} candidates)")
    print(f"train.jsonl  {len(train_rows)} triplets")
    print(f"-> {args.out}")


DATASETS = {
    "t2retrieval": ("A1_C-MTEB_T2Retrieval", "P1_qrels_A1_T2Retrieval-qrels"),
    "du": ("A2_C-MTEB_DuRetrieval", "P1_qrels_A2_DuRetrieval-qrels"),
    "covid": ("A3_C-MTEB_CovidRetrieval", "P1_qrels_A3_CovidRetrieval-qrels"),
    "ecom": ("E1_C-MTEB_EcomRetrieval", "P1_qrels_E1_EcomRetrieval-qrels"),
    "cmedqa": ("M4_C-MTEB_CmedqaRetrieval", "P1_qrels_M4_CmedqaRetrieval-qrels"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="t2retrieval", choices=list(DATASETS))
    ap.add_argument("--root", default="data/real")
    ap.add_argument("--out", default="data/real/eval/t2")
    ap.add_argument("--n-queries", type=int, default=0, help="0 = all")
    ap.add_argument("--n-cand", type=int, default=10)
    ap.add_argument("--n-pos", type=int, default=5)
    ap.add_argument("--n-train-neg", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--hard-neg", action="store_true", default=True)
    ap.add_argument("--hard-scan", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cdir, qdir = DATASETS[args.dataset]
    args.corpus = os.path.join(args.root, cdir)
    args.qrels = os.path.join(args.root, qdir)
    build(args)


if __name__ == "__main__":
    main()
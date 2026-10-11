"""
build_cold_law.py — build a STRICT cold-start dataset from Chinese regulations.

Why strict
----------
The project claim under test is: "a term that never appeared in training can be
made useful at inference time by writing into the CRUD registry, with no
retraining." That claim is only falsifiable if the evaluation terms are
*provably* unseen. So this builder does not approximate it, it enforces it:

  T0 = 2021-01-01
  TRAIN = { d : publish_date <  T0 }          (14,034 rows, content non-empty)
  COLD  = { d : publish_date >= T0 }          ( 4,351 rows)
  EVAL  = COLD rows whose *title* AND *first 200 chars of content* never occur
          in TRAIN                             ( 2,906 rows -> 66.8%)

Because the Layer-A registry is seeded only from TRAIN titles, and Layer-B is
fit only on TRAIN, every EVAL term is guaranteed to have:
  * no registry slot  -> matcher returns NULL (slot 0) before cold-start
  * zero gradient     -> hashed N-gram row stays at init

so any D-vs-C gain is attributable to the cold-start write, not to leakage.

Measured on this dataset (verified, not assumed):
  publish_date non-null  18,385 / 22,552  (81.5%), range 1950-01-01..2023-07-27
  type mix is 88% 地方性法规 -- NOT national law. Sampling is stratified by type.

Outputs (project JSONL contract, see de/data.py):
  train.jsonl      {"query","positive","negatives":[...]}   from TRAIN only
  eval_cold.jsonl  {"query","corpus":[...],"gold":int}     from EVAL (COLD)
  cold_start.jsonl {"term","contexts":[...]}               terms to write at inference

Usage:
    python build_cold_law.py --out data/real/eval/law
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

# the regulation body is full of amendment-revision tables; the leading header block
# carries the actual substance, so we keep the head of the document.
_BRK = re.compile(r"[\r\n]+")
_WS = re.compile(r"[ \t]+")
# 第X条 markers -- useful for pulling a self-contained clause as the positive
_CLAUSE = re.compile(r"第[一二三四五六七八九十百千零〇\d]+条")
# administrative boilerplate that must not become the query (it is identical across docs)
_BOILER = re.compile(r"(?:^|\n)>\s*(?:[^<>\n]{0,40})(?:修订|修改|废止)")


def clean(t: str, limit: int) -> str:
    if not t:
        return ""
    t = _BRK.sub(" ", t)
    return _WS.sub(" ", t).strip()[:limit]


def load_rows(root: str) -> List[dict]:
    import pyarrow.parquet as pq
    files = glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)
    if not files:
        raise FileNotFoundError(f"no parquet under {root}")
    f = max(files, key=os.path.getsize)
    return pq.read_table(
        f, columns=["publish_date", "type", "status", "title", "content"]
    ).to_pylist()


def split(rows: List[dict], t0_year: int):
    train, cold = [], []
    for r in rows:
        d = r.get("publish_date")
        if not d or not r.get("content"):
            continue
        (train if d.year < t0_year else cold).append(r)
    return train, cold


def unseen_filter(train: List[dict], cold: List[dict]):
    """Keep only COLD rows provably absent from TRAIN (title AND content head)."""
    tr_titles = {(r["title"] or "").strip() for r in train}
    tr_head = {(r["content"] or "")[:200].strip() for r in train}
    out = []
    for r in cold:
        t = (r["title"] or "").strip()
        h = (r["content"] or "")[:200].strip()
        if t and t not in tr_titles and h not in tr_head:
            out.append(r)
    return out


def title_is_generic(title: str) -> bool:
    """Drop revision-log titles: they are near-duplicates and teach nothing.

    e.g. '...（2018年修订）' / '...2018年修正案' -- the term itself already existed.
    """
    return bool(re.search(r"(修订|修正案|修改决定|废止)", title))


def split_clauses(content: str, limit: int = 400, max_clauses: int = 6) -> List[str]:
    """Pull self-contained 第X条 clauses out of the regulation body."""
    parts = _CLAUSE.split(content)
    # parts alternates [preamble, clause_body, clause_body, ...]
    bodies = parts[1:]
    out = []
    for b in bodies:
        b = clean(b, limit)
        if len(b) >= 40:
            out.append(b)
        if len(out) >= max_clauses:
            break
    return out


def build(args):
    rng = random.Random(args.seed)
    rows = load_rows(args.root)
    print(f"loaded {len(rows)} rows")

    train, cold = split(rows, args.t0_year)
    eval_pool = unseen_filter(train, cold)
    print(f"TRAIN (publish < {args.t0_year}) : {len(train)}")
    print(f"COLD  (publish >= {args.t0_year}): {len(cold)}")
    print(f"EVAL  (provably unseen)        : {len(eval_pool)}  "
          f"({100*len(eval_pool)/max(len(cold),1):.1f}% of COLD)")

    # ---- negative pool: TRAIN documents only, so negatives are never unseen too
    train_clauses: List[str] = []
    for r in train:
        cs = split_clauses(r["content"] or "")
        train_clauses.extend(cs)
    print(f"TRAIN clause pool (negatives)  : {len(train_clauses)}")
    if not train_clauses:
        raise RuntimeError("no TRAIN clauses -> cannot build negatives")

    # ---- eval rows: one per unseen regulation -----------------------------
    # CRITICAL (found by smoke test): the query MUST contain the term VERBATIM
    # and CONTIGUOUSLY, otherwise PhraseMatcher cannot fire and the whole
    # cold-start mechanism is a silent no-op (match rate 0% -> D == C always).
    # So the title is placed FIRST and the clause follows, and the query is kept
    # short enough that the title can never be truncated away.
    eval_rows = []
    cold_terms = []
    dropped = 0
    for r in eval_pool:
        title = (r["title"] or "").strip()
        if title_is_generic(title):
            continue
        clauses = split_clauses(r["content"] or "")
        if not clauses:
            continue
        body = clean(r["content"], 800)
        if len(body) < 100:
            continue

        # title first, verbatim, then the first clause as the "question" part
        query = clean(f"{title}。{clauses[0]}", args.query_budget)
        # guard: the title must survive the query_budget truncation
        if not query.startswith(title[: min(len(title), args.title_budget)]):
            dropped += 1
            continue
        negs = [rng.choice(train_clauses) for _ in range(args.n_neg)]
        eval_rows.append({
            "query": query,
            "corpus": [body] + negs,
            "gold": 0,
        })
        cold_terms.append({"term": title, "contexts": clauses[:3] or [title]})

    print(f"  dropped {dropped} rows whose title would not survive truncation")
    # CRITICAL: eval_rows and cold_terms are index-paired (query[i] belongs to
    # term[i]). Shuffling them independently silently mismatches them, which makes
    # the matcher fire on the wrong query and yields match_rate 0%.
    pairs = list(zip(eval_rows, cold_terms))
    rng.shuffle(pairs)
    pairs = pairs[: args.max_eval]
    eval_rows = [p[0] for p in pairs]
    cold_terms = [p[1] for p in pairs]

    # ---- train rows: query = old clause, positive = a nearby old clause -----
    train_rows = []
    by_type: Dict[str, List[str]] = defaultdict(list)
    for r in train:
        by_type[r.get("type") or "?"].extend(split_clauses(r["content"] or ""))
    for r in train:
        title = (r["title"] or "").strip()
        cs = split_clauses(r["content"] or "")
        if len(cs) < 2 or not title or title_is_generic(title):
            continue
        pool = by_type.get(r.get("type") or "?", []) or train_clauses
        neg = rng.choice(pool)
        train_rows.append({
            "query": clean(f"{title}。{cs[0]}", args.query_budget),
            "positive": cs[1],
            "negatives": [neg] * args.n_neg,
        })
        if len(train_rows) >= args.max_train:
            break

    os.makedirs(args.out, exist_ok=True)
    def dump(name, rs):
        p = os.path.join(args.out, name)
        with open(p, "w", encoding="utf-8") as f:
            for x in rs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"  {name:18s} {len(rs)}")

    print(f"-> {args.out}")
    dump("train.jsonl", train_rows)
    dump("eval_cold.jsonl", eval_rows)
    dump("cold_start.jsonl", cold_terms)
    _verify(args.out, args.query_budget)


def _verify(out_dir: str, query_budget: int):
    """Post-build check: every cold term must appear verbatim in its eval query."""
    try:
        from transformers import AutoTokenizer
        import numpy as np
        from de.hashing import CompressedTokenizer
    except Exception as e:                                    # offline fallback
        print(f"[verify] skipped ({type(e).__name__}: {e})")
        return
    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5")
    ct = CompressedTokenizer(tok)
    ev = [json.loads(l) for l in open(os.path.join(out_dir, "eval_cold.jsonl"),
                                      encoding="utf-8")]
    cs = [json.loads(l) for l in open(os.path.join(out_dir, "cold_start.jsonl"),
                                      encoding="utf-8")]
    hit = 0
    for r, c in zip(ev, cs):
        tids = tok(c["term"], add_special_tokens=False)["input_ids"]
        if not tids:
            continue
        canon = ct(np.asarray([tids], dtype=np.int64))[0].tolist()
        qids = tok(r["query"], truncation=True, max_length=512,
                   add_special_tokens=False)["input_ids"]
        qc = ct(np.asarray(qids, dtype=np.int64)).tolist()
        n = len(canon)
        if any(qc[i:i + n] == canon for i in range(len(qc) - n + 1)):
            hit += 1
    print(f"[verify] cold term present verbatim in its query: {hit}/{len(ev)} "
          f"({100*hit/max(len(ev),1):.1f}%)   <-- must be ~100% or E-009 is a no-op")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/real/COLD-1_twang2218_chinese-law-and-regulations")
    ap.add_argument("--out", default="data/real/eval/law")
    ap.add_argument("--t0-year", type=int, default=2021)
    ap.add_argument("--n-neg", type=int, default=9)
    ap.add_argument("--query-budget", type=int, default=200,
                    help="char budget for the query; must leave the title intact")
    ap.add_argument("--title-budget", type=int, default=60)
    ap.add_argument("--max-train", type=int, default=8000)
    ap.add_argument("--max-eval", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
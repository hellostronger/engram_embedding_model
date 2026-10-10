"""
prepare_data.py — build training/eval JSONL from the verified public datasets in
docs/DATA.md.

IMPORTANT: every field name below was verified against the HuggingFace
`datasets-server/first-rows` endpoint (NOT the README), per docs/DATA.md. Several of
these differ from the README -- e.g. coastalcph/lex_glue case_hold actually emits
`endings`, not `holdings`. Datasets whose schema could not be verified are DISABLED and
marked accordingly; re-enable only after dumping one row.

Known traps reproduced here (from docs/DATA.md):
  * Shitao/bge-m3-data is ONE 22.9 GB tarball -> cannot be streamed per-source.
  * C-MTEB/*-qrels live in SEPARATE repos; without them you get no negatives.
  * Alibaba-NLP/AliMe, C-MTEB/OnlineShopping, C-MTEB/JDReview return HTTP 401.
  * mteb/scifact and pile-of-law are NC -> EVAL ONLY, never training.

Usage:
    python prepare_data.py --dataset med_qa --out data/medical
    python prepare_data.py --dataset all --out data/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Callable, Dict, Iterator, List, Optional

MAX_TRIALS = 500_000


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def write_jsonl(path: str, rows: Iterator[dict]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_hf(path: str, name: Optional[str] = None, split: str = "train",
            streaming: bool = True):
    from datasets import load_dataset
    return load_dataset(path, name=name, split=split, streaming=streaming)


def take(it: Iterator, n: Optional[int] = None) -> Iterator:
    for i, r in enumerate(it):
        if n is not None and i >= n:
            break
        yield r


# --------------------------------------------------------------------------- #
# text cleanup
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")


def clean(t: Optional[str], limit: int = 512) -> str:
    if not t:
        return ""
    return _WS.sub(" ", str(t)).strip()[:limit]


# --------------------------------------------------------------------------- #
# dataset builders -> yield {"query","positive","negatives":[...]}
# --------------------------------------------------------------------------- #
def build_med_qa(limit: int, rng: random.Random) -> Iterator[dict]:
    """bigbio/med_qa (has a zh config). Fields: question / answer / options[{key,value}].

    The 3 non-gold options are *ideal* hard negatives: same question, wrong answers.
    """
    ds = load_hf("bigbio/med_qa", name="zh", split="train")
    for r in take(ds, limit):
        opts = r.get("options") or []
        if len(opts) < 2:
            continue
        gold_key = r.get("answer_idx", 0)
        gold = opts[gold_key].get("value") if isinstance(gold_key, int) and gold_key < len(opts) else None
        negs = [o.get("value") for i, o in enumerate(opts)
                if i != gold_key and o.get("value")]
        q = clean(r.get("question"))
        g = clean(gold)
        if not q or not g or not negs:
            continue
        yield {"query": q, "positive": g, "negatives": negs[:2]}


def build_pubmed_qa(limit: int, rng: random.Random) -> Iterator[dict]:
    """qiaojin/PubMedQA (MIT). Fields: question / context{contexts,labels,meshes}.

    Uses the `meshes` field to mine MeSH terms -> registry seeding candidates.
    """
    ds = load_hf("qiaojin/PubMedQA", name="pqa_labeled", split="train")
    for r in take(ds, limit):
        q = clean(r.get("question"))
        ctx = r.get("context") or {}
        ctxs = ctx.get("contexts") or []
        if not q or not ctxs:
            continue
        pos = clean(ctxs[0], 400)
        negs = [clean(c, 400) for c in ctxs[1:3]]
        yield {"query": q, "positive": pos, "negatives": negs,
               "_terms": [clean(m) for m in (ctx.get("meshes") or [])][:5]}


def build_case_hold(limit: int, rng: random.Random) -> Iterator[dict]:
    """coastalcph/lex_glue case_hold. Fields: context / endings[5] / label (0..4).

    NOTE: README says `holdings`; the loader actually emits `endings`.
    """
    ds = load_hf("coastalcph/lex_glue", name="case_hold", split="train")
    for r in take(ds, limit):
        ctx = clean(r.get("context"), 400)
        ends = [clean(e, 200) for e in (r.get("endings") or [])]
        if len(ends) < 2 or not ctx:
            continue
        lab = r.get("label", 0)
        if not isinstance(lab, int) or lab >= len(ends):
            continue
        yield {"query": ctx, "positive": ends[lab],
               "negatives": [e for i, e in enumerate(ends) if i != lab][:2]}


def build_disc_law(limit: int, rng: random.Random) -> Iterator[dict]:
    """ShengbinYue/DISC-Law-SFT (Apache-2.0). Fields: input / output."""
    ds = load_hf("ShengbinYue/DISC-Law-SFT", split="train")
    buf: List[str] = []
    for r in take(ds, limit):
        q, p = clean(r.get("input"), 400), clean(r.get("output"), 400)
        if not q or not p:
            continue
        buf.append(p)
        if len(buf) > 4:
            neg = rng.choice(buf[:-1])
            yield {"query": q, "positive": p, "negatives": [neg]}
            buf.pop(0)


def build_sec_filings(limit: int, rng: random.Random) -> Iterator[dict]:
    """PleIAs/SEC (CC0). Fields include `year` (STRING). Used for time-split cold start."""
    ds = load_hf("PleIAs/SEC", split="train")
    by_year: Dict[str, List[str]] = {}
    for r in take(ds, limit):
        y = str(r.get("year", ""))
        t = clean(r.get("text"), 400)
        if t and y:
            by_year.setdefault(y, []).append(t)
    for y, texts in by_year.items():
        for i, t in enumerate(texts):
            neg = texts[(i + 1) % len(texts)] if len(texts) > 1 else t
            yield {"query": clean(t, 200), "positive": t, "negatives": [neg], "_year": y}


BUILDERS: Dict[str, Callable] = {
    "med_qa": build_med_qa,
    "pubmed_qa": build_pubmed_qa,
    "case_hold": build_case_hold,
    "disc_law": build_disc_law,
    "sec_filings": build_sec_filings,
}

# Disabled until schema is verified against a live dump (see docs/DATA.md).
DISABLED = {
    "bge_m3": "single 22.9GB tarball, license undeclared",
    "finqa": "dead loading script (ibm-research/finqa)",
    "tat_qa": "schema extraction failed; dump one row first",
    "cn_finance": "no open Chinese finance triplet set; must self-mine",
    "cn_ecom_dict": "no open Chinese e-commerce category dictionary",
    "scifact": "CC-BY-NC -> EVAL ONLY",
    "pile_of_law": "NC license -> EVAL ONLY",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="med_qa", choices=list(BUILDERS))
    ap.add_argument("--out", default="data/med_qa.jsonl")
    ap.add_argument("--limit", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n = write_jsonl(args.out, BUILDERS[args.dataset](args.limit, rng))
    print(f"wrote {n} triplets -> {args.out}")
    if n == 0:
        print("WARNING: 0 rows. Check network / dataset gating / schema.")


if __name__ == "__main__":
    main()
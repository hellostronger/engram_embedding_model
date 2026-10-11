"""
download_data.py — reproduce every dataset listed in docs/EXPERIMENTS.md E-003f.

WHY THIS EXISTS
---------------
`data/` is gitignored (3.85 GB). Rather than committing binaries, this script
re-downloads exactly the same files from HuggingFace, deterministically, into
the same directory layout. Verified against the local copy on 2026-10-11
(3.73 GB, 24 datasets, ~1.05M parquet rows).

NETWORK
-------
No proxy is configured by default -- a normal machine with direct egress to
huggingface.co just works. If your host needs one, export HTTP_PROXY /
HTTPS_PROXY yourself before running (this repo deliberately hardcodes nothing).

Usage:
    python download_data.py --tier p0            # ~0.65 GB, what experiments need
    python download_data.py --tier p0 p1         # + fiqa / DISC-Law / Medical
    python download_data.py --tier p0 p1 p2      # + SEC / Amazon (~3.7 GB)
    python download_data.py --tier all --verify  # check an existing copy
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# (local_dir, repo_id, allow_patterns or None for whole repo, approx MB, why)
P0: List[Tuple[str, str, object, int, str]] = [
    ("COLD-1_twang2218_chinese-law-and-regulations",
     "twang2218/chinese-law-and-regulations", None, 153,
     "中文冷启动主战场，publish_date 时间戳，Apache-2.0  [E-008]"),
    ("A1_C-MTEB_T2Retrieval", "C-MTEB/T2Retrieval", None, 150,
     "中文通用检索，训练+评测主平台  [E-005/E-007/E-010]"),
    ("A2_C-MTEB_DuRetrieval", "C-MTEB/DuRetrieval", None, 62, "中文通用检索"),
    ("A3_C-MTEB_CovidRetrieval", "C-MTEB/CovidRetrieval", None, 62, "中文新闻"),
    ("E1_C-MTEB_EcomRetrieval", "C-MTEB/EcomRetrieval", None, 8, "中文电商"),
    ("M4_C-MTEB_CmedqaRetrieval", "C-MTEB/CmedqaRetrieval", None, 59, "中文医疗"),
    ("L2_mteb_LeCaRDv2", "mteb/LeCaRDv2", None, 150,
     "中文法律检索（实测仅 159q/3795cases，勿按论文规模规划）"),
    ("M2_bigbio_med_qa", "bigbio/med_qa", ["med_qa_zh_source/*"], 5, "中文医疗，硬负例来自 options"),
    ("P1_qrels_A1_T2Retrieval-qrels", "C-MTEB/T2Retrieval-qrels", None, 1, "负例来源，独立仓库"),
    ("P1_qrels_A2_DuRetrieval-qrels", "C-MTEB/DuRetrieval-qrels", None, 1, "负例来源"),
    ("P1_qrels_E1_EcomRetrieval-qrels", "C-MTEB/EcomRetrieval-qrels", None, 1, "负例来源"),
    ("P1_qrels_M4_CmedqaRetrieval-qrels", "C-MTEB/CmedqaRetrieval-qrels", None, 1, "负例来源"),
    ("P1_qrels_A3_CovidRetrieval-qrels", "C-MTEB/CovidRetrieval-qrels", None, 1, "负例来源"),
]

P1: List[Tuple[str, str, object, int, str]] = [
    ("P1_BeIR_fiqa", "BeIR/fiqa", None, 27, "英文金融检索"),
    ("P1_BeIR_fiqa-qrels", "BeIR/fiqa-qrels", None, 1, "格式为 query-id/corpus-id/score"),
    ("P1_ShengbinYue_DISC-Law-SFT", "ShengbinYue/DISC-Law-SFT", None, 552,
     "中文法律 Pair 166758 + Triplet 16000"),
    ("P1_C-MTEB_MedicalRetrieval", "C-MTEB/MedicalRetrieval", None, 24, "中文医疗检索"),
    ("P1_qrels_P1_MedicalRetrieval-qrels", "C-MTEB/MedicalRetrieval-qrels", None, 1, "负例来源"),
]

P2: List[Tuple[str, str, object, int, str]] = [
    ("P2_PleIAs_SEC", "PleIAs/SEC", ["2022.parquet", "2023.parquet", "2024.parquet"], 2200,
     "按 year 切分冷启动，CC0。只下三年，不要全量 20GB"),
    ("P2_McAuley-Lab_Amazon-Reviews-2023", "McAuley-Lab/Amazon-Reviews-2023",
     ["raw/review_categories/All_Beauty.jsonl", "raw/meta_categories/meta_All_Beauty.jsonl"],
     386, "英文电商冷启动，timestamp 毫秒"),
]

TIERS = {"p0": P0, "p1": P1, "p2": P2}

# Also needed for every run but cached by HF separately:
BACKBONE = "BAAI/bge-small-zh-v1.5"


def download_one(local_dir, repo_id, patterns, root):
    from huggingface_hub import snapshot_download
    dst = os.path.join(root, local_dir)
    t0 = time.time()
    p = snapshot_download(
        repo_id=repo_id, repo_type="dataset", local_dir=dst,
        allow_patterns=patterns,
    )
    n = sum(len(f) for _, _, f in os.walk(p))
    mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(p) for f in fs
    ) / 1e6
    print(f"    OK  {repo_id:46s} {mb:7.1f} MB  {n:3d} files  {time.time()-t0:5.1f}s",
          flush=True)
    return mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/real")
    ap.add_argument("--tier", nargs="+", default=["p0"], choices=[*TIERS, "all"])
    ap.add_argument("--backbone", action="store_true", default=True,
                    help="also fetch the backbone weights (recommended)")
    ap.add_argument("--verify", action="store_true",
                    help="only report what is present, download nothing")
    a = ap.parse_args()

    jobs = []
    for t in a.tier:
        jobs.extend(TIERS["all" if t == "all" else t])

    print(f"root={a.root}   datasets={len(jobs)}   "
          f"approx={sum(j[3] for j in jobs)/1000:.2f} GB", flush=True)

    if a.verify:
        missing = 0
        for local_dir, repo_id, _, mb, why in jobs:
            d = os.path.join(a.root, local_dir)
            ok = os.path.isdir(d) and any(
                len(fs) for _, _, fs in os.walk(d)
            )
            print(f"  {'OK  ' if ok else 'MISS'} {local_dir:46s} {why}")
            missing += (not ok)
        print(f"\nmissing {missing}/{len(jobs)}")
        return

    failed = []
    total = 0.0
    for local_dir, repo_id, patterns, mb, why in jobs:
        print(f"  -> {repo_id}  ({why})", flush=True)
        try:
            total += download_one(local_dir, repo_id, patterns, a.root)
        except Exception as e:
            # gated / 404 / no such config: report and continue, never block
            print(f"    FAIL {repo_id}: {type(e).__name__}: {str(e)[:160]}",
                  flush=True)
            failed.append(repo_id)

    if a.backbone:
        print("  -> backbone weights", flush=True)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=BACKBONE,
                              local_dir=os.path.join(a.root, "_backbone",
                                                     BACKBONE.replace("/", "__")))
            print(f"    OK  {BACKBONE}")
        except Exception as e:
            print(f"    FAIL {BACKBONE}: {type(e).__name__}: {str(e)[:160]}")
            failed.append(BACKBONE)

    print(f"\ndownloaded {total/1000:.2f} GB into {a.root}/")
    if failed:
        print(f"failed ({len(failed)}): {failed}")
        print("non-fatal: continue, the affected experiments are marked in "
              "docs/EXPERIMENTS.md")
    else:
        print("all datasets OK")


if __name__ == "__main__":
    sys.exit(main())
"""
run_gb10.py — single entry point for the GB10 (DGX Spark class) box.

Verified platform notes (see docs/COMPUTE.md for sources):
  * GB10 = Blackwell sm_121, ARM64 aarch64, 128 GB unified LPDDR5x, ~273 GB/s.
  * Use NGC PyTorch container (DGX-Spark tag) or torch built with CUDA >= 12.8
    (sm_120 binaries are binary-compatible with sm_121).
  * DO NOT use flash-attn: no aarch64/sm_121 wheels; a GB10 measurement showed
    SDPA is as fast or faster. Use PyTorch SDPA.
  * pin_memory is pointless (unified memory); leave it off to avoid H2D pathologies.

This script:
  1. sets seeds and prints a resolved config + parameter accounting,
  2. optionally runs a GPU micro-benchmark to calibrate throughput,
  3. runs Stage-1 training, Stage-2 cold-start, then the A/B/C/D evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from de.config import default_config
from de.model import DomainEngramEncoder
from de.train import set_seed


# --------------------------------------------------------------------------- #
def describe(model) -> dict:
    mem_rows = 0
    for name, head in model.named_modules():
        if name.startswith("engram_head_") and hasattr(head, "tables"):
            mem_rows += sum(t.num_embeddings for t in head.tables)
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    to = sum(p.numel() for p in model.parameters())
    bb = sum(p.numel() for p in model.backbone.parameters())
    return {
        "backbone_params": bb,
        "trainable_params": tr,
        "total_params": to,
        "hashed_table_rows": mem_rows,
        "registry_slots": model.registry.max_slots,
        "trainable_pct_of_total": round(100 * tr / to, 2),
        "trainable_over_backbone": round(tr / max(bb, 1), 2),
        "estimated_resident_gb_fp32": round(tr * 16 / 1e9, 3),  # param+grad+m+v
    }


PRESETS = {
    # tiny: fastest kill-shot, fits anywhere
    "tiny": dict(table_size_per_ngram=4096, d_head=16, n_head_per_ngram=4,
                 max_registry_slots=5_000, d_registry=32, layer_ids=[1, 3]),
    # small: balanced first real run
    "small": dict(table_size_per_ngram=16384, d_head=32, n_head_per_ngram=8,
                  max_registry_slots=50_000, d_registry=64, layer_ids=[1, 3]),
    # ref: reference config (matches the experiments plan's DE-Ref-S scale)
    "ref": dict(table_size_per_ngram=32768, d_head=32, n_head_per_ngram=8,
                max_registry_slots=200_000, d_registry=64, layer_ids=[1, 3]),
    # medium: recommended GB10 config
    "medium": dict(table_size_per_ngram=65536, d_head=32, n_head_per_ngram=8,
                   max_registry_slots=200_000, d_registry=64, layer_ids=[1, 3]),
    # large: scale-up once results are positive
    "large": dict(table_size_per_ngram=131072, d_head=64, n_head_per_ngram=8,
                  max_registry_slots=500_000, d_registry=128, layer_ids=[1, 3]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="medium", choices=list(PRESETS))
    ap.add_argument("--backbone", default="BAAI/bge-small-zh-v1.5")
    ap.add_argument("--stage", default="all",
                    choices=["describe", "bench", "train", "coldstart", "eval", "all"])
    ap.add_argument("--train", default=None, help="train.jsonl")
    ap.add_argument("--eval", default=None, help="eval.jsonl")
    ap.add_argument("--new-terms", default=None, help="jsonl of unseen terms to cold-start")
    ap.add_argument("--out", default="runs/de")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pooling", default="cls", choices=["cls", "mean", "last"])
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"  capability={torch.cuda.get_device_capability()} "
              f"name={torch.cuda.get_device_name(0)}")

    cfg = default_config(backbone_name=args.backbone, pooling=args.pooling,
                         freeze_backbone=True, **PRESETS[args.preset])
    model = DomainEngramEncoder(cfg).to(device)
    info = describe(model)
    print("\n=== config ===")
    print(json.dumps(info, indent=2))
    print(f"resolved: {cfg}")

    if args.stage == "describe":
        return

    # ---------------- GPU micro-benchmark (calibrate before long runs) ------ #
    if args.stage in ("bench", "all"):
        model.train()
        ids = torch.randint(100, 1000, (args.batch_size, args.max_len), device=device)
        mask = torch.ones_like(ids)
        slots = torch.zeros(args.batch_size, args.max_len,
                            cfg.max_matches_per_pos, dtype=torch.long, device=device)
        from de.train import make_optimizer
        opt = make_optimizer(model, args.lr, warmup=0)
        for i in range(3):  # warmup
            loss = model(ids, mask, slots).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()
        N = 10
        for i in range(N):
            loss = model(ids, mask, slots).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize() if device.type == "cuda" else None
        dt = (time.time() - t0) / N
        tps = args.batch_size * args.max_len / dt
        print(f"\n=== benchmark ===\n{args.batch_size}x{args.max_len} "
              f"{dt*1000:.1f} ms/step -> {tps:,.0f} tok/s (incl. memory-table Adam)")
        if device.type == "cuda":
            print(f"peak mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # ---------------- Stage 1 --------------------------------------------- #
    if args.stage in ("train", "all"):
        if not args.train:
            print("\n[stage1] skipped (no --train)")
        else:
            from transformers import AutoTokenizer
            from de.data import TripletDataset
            from de.train import train_stage1, save_checkpoint
            tok = AutoTokenizer.from_pretrained(args.backbone)
            ds = TripletDataset.from_jsonl(args.train)
            print(f"\n[stage1] training on {len(ds)} triplets")
            train_stage1(model, tok, ds, epochs=args.epochs,
                         batch_size=args.batch_size, lr=args.lr,
                         max_len=args.max_len, device=device, seed=args.seed)
            save_checkpoint(model, tok, os.path.join(args.out, "stage1.pt"), cfg)

    # ---------------- Stage 2: cold start on unseen terms ------------------- #
    if args.stage in ("coldstart", "all"):
        if not args.new_terms:
            print("\n[stage2] skipped (no --new-terms)")
        else:
            from transformers import AutoTokenizer
            from de.cold_start import cold_start
            tok = AutoTokenizer.from_pretrained(args.backbone)
            rows = [json.loads(l) for l in open(args.new_terms, encoding="utf-8") if l.strip()]
            t0 = time.time()
            for r in rows:
                cold_start(model, tok, r["term"], r.get("contexts", []),
                           refine_steps=r.get("refine_steps", 0))
            model.sync_matcher()
            dt = time.time() - t0
            print(f"\n[stage2] wrote {len(rows)} terms in {dt:.2f}s "
                  f"({dt/max(len(rows),1)*1000:.2f} ms/term)")
            torch.save({"registry": model.registry.to_json(),
                        "table": model.registry_table.state_dict()},
                       os.path.join(args.out, "registry.pt"))

    print("\n[done]")


if __name__ == "__main__":
    main()
"""
train.py — Stage 1 (memory warm-up) and Stage 2 (cold-start) training loops.

Key hyperparameters follow the Engram paper Sec 4.1:
  * memory-table params use Adam, lr scaled by MEMORY_LR_MULT, weight_decay = 0
  * conv params are zero-initialised (identity at init)
  * backbone is frozen by default (the "low-cost domain adaptation" claim)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch. Required for reproducible experiment comparison."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from .config import EngramConfig, default_config
from .data import DECollator, TripletDataset
from .evaluation import evaluate_retrieval
from .losses import info_nce, info_nce_with_hard_negatives
from .model import DomainEngramEncoder


def build_param_groups(model, base_lr: float, memory_lr_mult: float = 5.0, weight_decay: float = 0.01):
    """Split params: memory tables (Adam, no WD, 5x lr) vs fusion (AdamW)."""
    mem, fusion = [], []
    for name, p in model.engram_parameters():
        if not p.requires_grad:
            continue
        if ".conv." in name or "norm_" in name or "gate_bias" in name or ".bias" in name:
            fusion.append(p)
        else:
            mem.append(p)
    groups = [
        {"params": mem, "lr": base_lr * memory_lr_mult, "weight_decay": 0.0, "opt": "adam"},
        {"params": fusion, "lr": base_lr, "weight_decay": weight_decay, "opt": "adamw"},
    ]
    # backbone (if unfrozen)
    bb = [p for p in model.backbone.parameters() if p.requires_grad]
    if bb:
        groups.append({"params": bb, "lr": base_lr, "weight_decay": weight_decay, "opt": "adamw"})
    return [g for g in groups if len(g["params"]) > 0]


def make_optimizer(model, base_lr: float, memory_lr_mult: float = 5.0,
                   weight_decay: float = 0.01, warmup: int = 100):
    """Build one optimizer per param group with a linear warmup.

    Memory-table group : Adam,   lr = base_lr * memory_lr_mult, wd = 0  (paper Sec 4.1)
    Fusion group       : AdamW,  lr = base_lr,                     wd = weight_decay
    Backbone (optional): AdamW,  lr = base_lr,                     wd = weight_decay
    """
    groups = build_param_groups(model, base_lr, memory_lr_mult, weight_decay)
    opts = []
    for g in groups:
        cls = torch.optim.Adam if g["opt"] == "adam" else torch.optim.AdamW
        opts.append(cls(g["params"], lr=g["lr"], weight_decay=g["weight_decay"]))
    if not opts:
        raise RuntimeError("no trainable parameters found")
    return _MultiOpt(opts, warmup)


class _MultiOpt:
    """Thin wrapper: step()/zero_grad() across several optimizers + linear warmup."""

    def __init__(self, opts, warmup: int = 100):
        self.opts = opts
        self.base_lrs = [o.param_groups[0]["lr"] for o in opts]
        self.warmup = max(int(warmup), 0)
        self._step = 0

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()
        self._step += 1
        if self.warmup > 0 and self._step <= self.warmup:
            f = 0.1 + 0.9 * (self._step / self.warmup)
            for o, base in zip(self.opts, self.base_lrs):
                for pg in o.param_groups:
                    pg["lr"] = base * f

    @property
    def param_groups(self):
        return [pg for o in self.opts for pg in o.param_groups]


def train_stage1(
    model: DomainEngramEncoder,
    tokenizer,
    dataset: TripletDataset,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 2e-5,
    memory_lr_mult: float = 5.0,
    temperature: float = 0.02,
    max_len: int = 128,
    device: Optional[torch.device] = None,
    warmup: int = 100,
    log_every: int = 50,
    grad_clip: float = 1.0,
    num_workers: int = 4,
    seed: int = 0,
):
    device = device or model.device
    model.to(device)
    model.train()
    set_seed(seed)

    collate = DECollator(tokenizer, model.matcher, max_len=max_len,
                          lookup_table=model.compressed.lookup_table)
    # num_workers>0 matters: PhraseMatcher runs on CPU (numpy), so with 0 workers it
    # serializes against the GPU. Each worker gets a copy of the matcher; because we
    # rebuild on version change and workers fork AFTER training starts, all workers see
    # the same registry state.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate, drop_last=True, num_workers=num_workers,
                        persistent_workers=num_workers > 0,
                        pin_memory=torch.cuda.is_available())

    opt = make_optimizer(model, lr, memory_lr_mult, warmup=warmup)

    step = 0
    for ep in range(epochs):
        for batch in loader:
            q = batch["query"]
            d = batch["positive"]
            q_emb = model(q["input_ids"].to(device), q["attention_mask"].to(device),
                          q["registry_slots"].to(device))
            d_emb = model(d["input_ids"].to(device), d["attention_mask"].to(device),
                          d["registry_slots"].to(device))

            if "negative" in batch and batch["negative"]["input_ids"].shape[0] > 0:
                n = batch["negative"]
                n_emb = model(n["input_ids"].to(device), n["attention_mask"].to(device),
                              n["registry_slots"].to(device))
                K = n_emb.shape[0] // q_emb.shape[0]
                n_emb = n_emb.view(q_emb.shape[0], K, -1)
                loss = info_nce_with_hard_negatives(q_emb, d_emb, n_emb, temperature)
            else:
                loss = info_nce(q_emb, d_emb, temperature)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
            opt.step()
            step += 1
            if step % log_every == 0:
                print(f"[stage1] ep={ep} step={step} loss={loss.item():.4f}")
    return model


def save_checkpoint(model, tokenizer, path: str, config: EngramConfig):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "model": model.state_dict(),
        "registry": model.registry.to_json(),
        "config": asdict(config),
    }
    torch.save(state, path)
    print(f"saved checkpoint -> {path}")


def load_checkpoint(model, path: str, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    print(f"loaded checkpoint <- {path}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="BAAI/bge-small-zh-v1.5")
    ap.add_argument("--train", required=True, help="path to train.jsonl")
    ap.add_argument("--out", default="runs/de_stage1.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    cfg = default_config(
        backbone_name=args.backbone, layer_ids=args.layers,
        freeze_backbone=True, pooling="cls",
    )
    model = DomainEngramEncoder(cfg)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    ds = TripletDataset.from_jsonl(args.train)
    print(f"train triplets: {len(ds)}  device: {device}")
    train_stage1(model, tokenizer, ds, epochs=args.epochs, batch_size=args.batch_size,
                 lr=args.lr, max_len=args.max_len, device=device)
    save_checkpoint(model, tokenizer, args.out, cfg)


if __name__ == "__main__":
    main()
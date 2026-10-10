"""
config.py — configuration for Domain Engram (DE)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EngramConfig:
    """Hyper-parameters for the Engram memory + fusion module.

    Defaults target a small/medium bi-encoder (e.g. BGE-small, d_model=384). The ratios
    follow the paper's Engram-27B setting where d_mem=1280 for d_model=2560 (~1/2).
    """

    # --- injection topology ---
    layer_ids: List[int] = field(default_factory=lambda: [1, 3])  # 0-indexed encoder blocks
    n_branches: int = 1  # 1 = plain encoder; >1 only meaningful with a multi-branch backbone

    # --- hashed N-gram memory (Layer-B) ---
    max_ngram_size: int = 3  # orders 2..max_ngram_size
    n_head_per_ngram: int = 8
    table_size_per_ngram: int = 65536  # base size; each head gets a distinct prime >= this
    d_head: int = 32  # per (order, head) embedding dim
    hash_seed: int = 0

    # --- CRUD registry (Layer-A) ---
    max_registry_slots: int = 200_000
    d_registry: int = 64
    max_phrase_len: int = 32
    max_matches_per_pos: int = 4

    # --- fusion ---
    conv_kernel_size: int = 4
    gate_bias_init: float = -2.0
    gate_sqrt: bool = False

    # --- backbone ---
    backbone_name: str = "BAAI/bge-small-zh-v1.5"
    pooling: str = "cls"  # cls | mean | last
    normalize: bool = True
    freeze_backbone: bool = True
    torch_dtype: str = "float32"


def default_config(**overrides) -> EngramConfig:
    cfg = EngramConfig()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"unknown EngramConfig field: {k}")
        setattr(cfg, k, v)
    return cfg
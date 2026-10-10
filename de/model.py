"""
model.py — Domain Engram encoder: a frozen/finetuned bi-encoder backbone with Engram
conditional-memory injection + a CRUD registry, producing a single pooled embedding.

Pipeline (per forward):
  input_ids
    -> canonical ids (CompressedTokenizer)                       [deterministic]
    -> hashed N-gram indices (NgramHashMapping)                  [O(1), device-side]
    -> registry slot ids per position (PhraseMatcher, done in the data/collate step)
    -> backbone embeddings -> for each injection layer: h = h + Engram(h, mem)
    -> pooling -> normalize -> embedding
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .config import EngramConfig
from .engram import EngramTableHead
from .hashing import CompressedTokenizer, NgramHashMapping
from .registry import PhraseMatcher, PhraseRegistry, RegistryTable


def attention_bias(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert a 0/1 mask to the additive float mask transformers' blocks expect.

    Recent transformers (>=4.4x) hand the mask straight to SDPA, which requires
    bool or float, not int64. We therefore always build the additive form:

        bias = (1 - mask) * finfo(dtype).min
    """
    m = attention_mask[:, None, None, :].to(dtype)
    return (1.0 - m) * torch.finfo(dtype).min


def _get_blocks(backbone):
    """Return the list of transformer blocks for BERT-like / ModernBERT backbones."""
    for path in (
        lambda m: getattr(m, "encoder", None) and getattr(m.encoder, "layer", None),
        lambda m: getattr(m, "layers", None),          # ModernBERT
        lambda m: getattr(m, "model", None) and getattr(m.model, "layers", None),
    ):
        blocks = path(backbone)
        if blocks is not None:
            return blocks
    raise ValueError("could not locate transformer blocks in backbone")


class DomainEngramEncoder(nn.Module):
    def __init__(
        self,
        cfg: EngramConfig,
        registry: Optional[PhraseRegistry] = None,
        compressed_tokenizer: Optional[CompressedTokenizer] = None,
        backbone=None,
        tokenizer=None,
    ):
        super().__init__()
        self.cfg = cfg
        dtype = getattr(torch, cfg.torch_dtype)

        # ---- backbone ----
        if backbone is None:
            backbone = AutoModel.from_pretrained(cfg.backbone_name, torch_dtype=dtype)
        self.backbone = backbone
        self.hidden_size = backbone.config.hidden_size
        self.blocks = _get_blocks(self.backbone)
        self.num_layers = len(self.blocks)
        self.layer_ids = [l for l in cfg.layer_ids if l < self.num_layers]

        # ---- tokenizer compression + hashing ----
        if compressed_tokenizer is None:
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.backbone_name)
            compressed_tokenizer = CompressedTokenizer(tokenizer)
        self.compressed = compressed_tokenizer
        self.register_buffer(
            "canonical_lookup",
            torch.from_numpy(compressed_tokenizer.lookup_table),
            persistent=False,
        )
        self.hasher = NgramHashMapping(
            canonical_vocab_size=compressed_tokenizer.canonical_vocab_size,
            max_ngram_size=cfg.max_ngram_size,
            table_size_per_ngram=cfg.table_size_per_ngram,
            n_head_per_ngram=cfg.n_head_per_ngram,
            layer_ids=self.layer_ids,
            pad_id=compressed_tokenizer.pad_id,
            seed=cfg.hash_seed,
        )

        # ---- CRUD registry (Layer-A) ----
        self.registry = registry if registry is not None else PhraseRegistry(cfg.max_registry_slots)
        self.registry_table = RegistryTable(
            self.registry.max_slots, cfg.d_registry
        ).to(dtype)
        self.matcher = PhraseMatcher(
            self.registry,
            max_per_pos=cfg.max_matches_per_pos,
            max_phrase_len=cfg.max_phrase_len,
        )

        # ---- Engram injection heads (Layer-B tables + shared fusion) ----
        dilation = cfg.max_ngram_size
        for lid in self.layer_ids:
            head = EngramTableHead(
                table_sizes=self.hasher.vocab_size_across_layers[lid],
                d_head=cfg.d_head,
                d_model=self.hidden_size,
                # the registry block is the concatenation of up to max_matches_per_pos
                # slot rows, so its width is R * d_registry (see _registry_vecs)
                registry_dim=cfg.d_registry * cfg.max_matches_per_pos,
                n_branches=cfg.n_branches,
                kernel_size=cfg.conv_kernel_size,
                dilation=dilation,
                gate_bias_init=cfg.gate_bias_init,
                gate_sqrt=cfg.gate_sqrt,
            ).to(dtype)
            self.add_module(f"engram_head_{lid}", head)

        # ---- memory-suppression switch (paper Sec 6.3 sensitivity ablation) ----
        self.memory_enabled = True

        if cfg.freeze_backbone:
            self.freeze_backbone()

    # ------------------------------------------------------------------ utils
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.freeze_backbone:
            self.backbone.eval()   # keep frozen BN/LN/dropout off
        return self

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def engram_parameters(self):
        """Memory tables + fusion params (the DE-specific trainable set)."""
        out = []
        for name, p in self.named_parameters():
            if name.startswith("engram_head_") or name.startswith("registry_table"):
                out.append((name, p))
        return out

    def memory_parameters(self):
        return [(n, p) for n, p in self.engram_parameters() if ".conv." not in n]

    # ------------------------------------------------------------------ memory ops
    @torch.no_grad()
    def add_term(self, phrase_ids: Sequence[int], surface: str = "", **meta) -> int:
        """Operator write path: allocate a slot and zero it (ready for cold-start)."""
        self.matcher.maybe_rebuild()
        slot = self.registry.add(phrase_ids, surface=surface, **meta)
        if self.registry_table.emb.weight.device.type == "cuda":
            self.registry_table.reset_slot(slot)
        else:
            self.registry_table.reset_slot(slot)
        return slot

    @torch.no_grad()
    def delete_term(self, phrase_ids: Sequence[int]) -> bool:
        self.registry.delete(phrase_ids)
        self.matcher.maybe_rebuild()
        return True

    def sync_matcher(self):
        self.matcher.maybe_rebuild()

    # ------------------------------------------------------------------ forward
    def _pool(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.cfg.pooling == "cls":
            return h[:, 0]
        if self.cfg.pooling == "mean":
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            return (h * m).sum(1) / m.sum(1).clamp_min(1e-6)
        if self.cfg.pooling == "last":
            lens = attention_mask.sum(1).clamp_min(1) - 1
            return h[torch.arange(h.shape[0], device=h.device), lens]
        raise ValueError(f"unknown pooling {self.cfg.pooling}")

    def _canonical_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        lut = self.canonical_lookup.to(input_ids.device)
        safe = input_ids.clamp_min(0)
        return lut[safe]

    def _registry_vecs(self, registry_slots: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if registry_slots is None:
            return None
        B, T, R = registry_slots.shape
        vecs = self.registry_table(registry_slots.reshape(B, T * R))   # (B, T*R, d_reg)
        return vecs.reshape(B, T, R * self.cfg.d_registry)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        registry_slots: Optional[torch.Tensor] = None,
        return_gate: bool = False,
    ):
        use_memory = self.memory_enabled and bool(self.layer_ids)
        gates = {}

        canon = self._canonical_ids(input_ids) if use_memory else None
        reg_vecs = self._registry_vecs(registry_slots) if use_memory else None

        # token embeddings (never touch input/output embedding per Engram design)
        h = self.backbone.embeddings(input_ids=input_ids)
        attn_bias = attention_bias(attention_mask, h.dtype)

        for idx, block in enumerate(self.blocks):
            if use_memory and idx in self.layer_ids:
                head = getattr(self, f"engram_head_{idx}")
                hash_ids = self.hasher.hash_layer(canon, idx)
                y, alpha = head(h, hash_ids, reg_vecs)
                h = h + y
                if return_gate:
                    gates[idx] = alpha.detach()
            out = block(h, attention_mask=attn_bias)
            h = out[0] if isinstance(out, (tuple, list)) else out

        emb = self._pool(h, attention_mask)
        if self.cfg.normalize:
            emb = F.normalize(emb, dim=-1)
        if return_gate:
            return emb, gates
        return emb

    # ------------------------------------------------------------------ convenience
    def prepare_registry_slots(self, input_ids_list: List[List[int]]) -> torch.Tensor:
        """CPU-side: canonical ids -> (B, T, R) slot tensor for the forward pass."""
        arr = np.asarray(input_ids_list, dtype=np.int64)
        canon = self.compressed(arr)                       # (B,T)
        slots = self.matcher.match_batch(canon)             # (B,T,R)
        return torch.from_numpy(slots)

    @torch.no_grad()
    def encode(
        self,
        input_ids_list: List[List[int]],
        batch_size: int = 64,
        max_len: int = 256,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Simple CPU-driven encoding loop (matcher runs on CPU)."""
        device = device or self.device
        outs = []
        self.eval()
        for i in range(0, len(input_ids_list), batch_size):
            chunk = input_ids_list[i: i + batch_size]
            L = min(max(len(c) for c in chunk), max_len)
            ids = torch.zeros(len(chunk), L, dtype=torch.long)
            mask = torch.zeros(len(chunk), L, dtype=torch.long)
            for j, c in enumerate(chunk):
                c = c[:L]
                ids[j, : len(c)] = torch.tensor(c, dtype=torch.long)
                mask[j, : len(c)] = 1
            slots = self.prepare_registry_slots(chunk).to(device)
            emb = self(ids.to(device), mask.to(device), slots)
            outs.append(emb.float().cpu())
        return torch.cat(outs, dim=0)
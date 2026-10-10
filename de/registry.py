"""
registry.py — Layer-A: the CRUD-addressable domain memory  (our increment over Engram)

Why this exists
---------------
Engram's table is addressed purely by hash, so an operator cannot ask "what row holds
the term 糖尿病视网膜病变?". That blocks the product requirement of *dynamically
adding / updating / removing domain terms without retraining*.

We therefore split memory in two:

  Layer-A  PhraseRegistry  -- explicit, append-only slot table + an inverse index
                             (phrase -> slot). O(1) create / read / update / delete.
                             This is the *writable* surface used by operators.
  Layer-B  Hashed N-grams   -- trained statistical memory (see hashing.py), read-only.

Both are merged into one vector per token position before the Engram fusion module, so
the rest of the architecture is unchanged.

Guarantees
----------
* Delete never moves a slot id (tombstone + free list) => no index rebuild, no retrain.
* Slot 0 is a permanently-zero NULL row (padding_idx=0) => "no phrase matched" is an
  exact no-op, so an empty registry can never perturb the backbone output.
* All mutation metadata (slot, active flag, hit counts) is persisted separately from the
  vector table, so the vectors can live in GPU memory while the index lives on disk/DB.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

Phrase = Tuple[int, ...]


@dataclass
class RegistryEntry:
    slot: int
    phrase: Phrase
    surface: str
    active: bool = True
    n_hits: int = 0
    meta: dict = field(default_factory=dict)


class PhraseRegistry:
    """Append-only slot allocator + inverse index over canonical token-id phrases."""

    def __init__(self, max_slots: int):
        if max_slots < 2:
            raise ValueError("max_slots must be >= 2 (slot 0 is reserved NULL)")
        self.max_slots = int(max_slots)
        self.entries: Dict[Phrase, RegistryEntry] = {}
        self._free: List[int] = list(range(self.max_slots - 1, 0, -1))  # pop() -> smallest
        self.version = 0

    # ------------------------------------------------------------------ CRUD
    def add(self, phrase: Sequence[int], surface: str = "", **meta) -> int:
        """Create (or return the slot of) a phrase. O(1)."""
        p = tuple(int(x) for x in phrase)
        if not p:
            raise ValueError("empty phrase")
        if p in self.entries:
            e = self.entries[p]
            if not e.active:  # resurrect a tombstoned entry
                e.active = True
                self.version += 1
            return e.slot
        if not self._free:
            raise RuntimeError(
                f"phrase registry exhausted ({self.max_slots} slots); "
                "raise EngramConfig.max_registry_slots or compact the table"
            )
        slot = self._free.pop()
        self.entries[p] = RegistryEntry(slot=slot, phrase=p, surface=surface, meta=dict(meta))
        self.version += 1
        return slot

    def get_slot(self, phrase: Sequence[int]) -> Optional[int]:
        e = self.entries.get(tuple(int(x) for x in phrase))
        return e.slot if (e is not None and e.active) else None

    def delete(self, phrase: Sequence[int]) -> bool:
        """Tombstone a phrase and return its slot to the free list. O(1)."""
        p = tuple(int(x) for x in phrase)
        e = self.entries.get(p)
        if e is None or not e.active:
            return False
        e.active = False
        self.entries.pop(p, None)  # drop inverse-index entry; slot stays stable
        self._free.append(e.slot)
        self.version += 1
        return True

    def mark_hit(self, phrase: Sequence[int], n: int = 1) -> None:
        e = self.entries.get(tuple(int(x) for x in phrase))
        if e is not None:
            e.n_hits += n

    # ------------------------------------------------------------------ query
    def __len__(self) -> int:
        return sum(1 for e in self.entries.values() if e.active)

    @property
    def n_used_slots(self) -> int:
        return self.max_slots - len(self._free)

    def phrases(self) -> Iterable[RegistryEntry]:
        return list(self.entries.values())

    def active_phrases(self) -> List[RegistryEntry]:
        return [e for e in self.entries.values() if e.active]

    # ------------------------------------------------------------------ io
    def to_json(self) -> str:
        return json.dumps(
            {
                "max_slots": self.max_slots,
                "entries": [
                    {"phrase": list(e.phrase), "surface": e.surface, "slot": e.slot,
                     "n_hits": e.n_hits, "meta": e.meta}
                    for e in self.entries.values()
                ],
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, blob: str) -> "PhraseRegistry":
        d = json.loads(blob)
        reg = cls(max_slots=d["max_slots"])
        used = set()
        for item in d["entries"]:
            p = tuple(item["phrase"])
            e = RegistryEntry(
                slot=item["slot"], phrase=p, surface=item.get("surface", ""),
                n_hits=item.get("n_hits", 0), meta=item.get("meta", {}),
            )
            reg.entries[p] = e
            used.add(e.slot)
        reg._free = [s for s in range(reg.max_slots - 1, 0, -1) if s not in used]
        reg.version += 1
        return reg


class RegistryTable(nn.Module):
    """[max_slots, d] embedding table for Layer-A. Slot 0 is NULL (frozen at zero)."""

    def __init__(self, max_slots: int, dim: int, init_std: float = 0.02):
        super().__init__()
        self.emb = nn.Embedding(max_slots, dim, padding_idx=0)
        nn.init.normal_(self.emb.weight, mean=0.0, std=init_std)
        with torch.no_grad():
            self.emb.weight[0].zero_()

    @torch.no_grad()
    def reset_slot(self, slot: int) -> None:
        """Reset a slot to a small random vector (used before cold-start writes)."""
        if slot != 0:
            nn.init.normal_(self.emb.weight[slot], mean=0.0, std=self.emb.weight.std().item())

    def forward(self, slot_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(slot_ids)


# --------------------------------------------------------------------------- #
# Phrase matching: canonical token id sequence -> per-position slot ids
# --------------------------------------------------------------------------- #
class PhraseMatcher:
    """Rolling-hash Aho-Corasick-lite matcher over canonical token ids.

    For a batch of canonical sequences we emit, for every position t, the registry
    slots of every phrase that *ends* at t (matching Engram's suffix-N-gram convention),
    capped at ``max_per_pos`` and ordered longest-phrase-first.

    Implementation: group phrases by length L, compute a vectorised polynomial
    rolling hash for every length-L window, then `searchsorted` into a sorted hash
    array. Only surviving candidates are verified by exact tuple comparison (which
    removes hash collisions entirely), so this is O(T * |lengths|) numpy + O(#hits).
    """

    def __init__(
        self,
        registry: PhraseRegistry,
        max_per_pos: int = 4,
        max_phrase_len: int = 32,
        seed: int = 0xC0FFEE,
    ):
        self.registry = registry
        self.max_per_pos = int(max_per_pos)
        self.max_phrase_len = int(max_phrase_len)
        # odd base, kept in [1, 2^64) so that both the tuple hash and the rolling hash
        # can be computed in pure mod-2^64 integer arithmetic.
        self.base = int(((seed * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)) | 1)
        self._built_version = -1
        self._sorted_hashes: Dict[int, np.ndarray] = {}
        self._buckets: Dict[int, Dict[np.uint64, List[Phrase]]] = {}
        self._by_len: Dict[int, List[Phrase]] = {}
        self._slot_of: Dict[Phrase, int] = {}
        self._rebuild(registry)

    # ------------------------------------------------------------------ build
    def _rebuild(self, registry: PhraseRegistry) -> None:
        by_len: Dict[int, List[Phrase]] = {}
        for e in registry.active_phrases():
            if len(e.phrase) <= self.max_phrase_len:
                by_len.setdefault(len(e.phrase), []).append(e.phrase)
        self._by_len = by_len
        self._slot_of = {e.phrase: e.slot for e in registry.active_phrases()}

        sorted_hashes: Dict[int, np.ndarray] = {}
        buckets: Dict[int, Dict[np.uint64, List[Phrase]]] = {}
        for L, phrases in by_len.items():
            bk: Dict[np.uint64, List[Phrase]] = {}
            for p in phrases:
                bk.setdefault(self._hash_tuple(p), []).append(p)
            hashes = np.array(sorted(bk.keys()), dtype=np.uint64)
            sorted_hashes[L] = hashes
            buckets[L] = bk
        self._sorted_hashes = sorted_hashes
        self._buckets = buckets
        self._built_version = registry.version

    def maybe_rebuild(self) -> None:
        if self._built_version != self.registry.version:
            self._rebuild(self.registry)

    # ------------------------------------------------------------------ hash
    _MASK64 = (1 << 64) - 1

    def _hash_tuple(self, phrase: Sequence[int]) -> np.uint64:
        """Polynomial hash mod 2^64, computed in python ints (no numpy overflow).

        NOTE: iterates the phrase in REVERSE. The rolling hash accumulates the window
        as x[p]*B^(L-1) + x[p-1]*B^(L-2) + ... + x[p-L+1], i.e. it weights the most
        recent token highest. Reversing here makes the two agree exactly, which is what
        makes the searchsorted candidate lookup able to VERIFY (and thus be collision
        free). If these two disagree, no candidate ever verifies and the registry
        silently matches nothing.
        """
        h = 0
        base = int(self.base)
        for tok in reversed(phrase):
            h = (h * base + int(tok) + 1) & self._MASK64
        return np.uint64(h)

    def _rolling_hash(self, x: np.ndarray, L: int) -> np.ndarray:
        """Rolling version of _hash_tuple over all windows; all arithmetic mod 2^64.

        Uses uint64 numpy ops with explicit wrapping (numpy may warn on scalar
        overflow, so we cast to a 0-d array to keep it in integer arithmetic).
        """
        T = x.shape[0]
        h = np.zeros(T, dtype=np.uint64)
        xu = (x.astype(np.int64) + 1).astype(np.uint64)
        base = np.uint64(int(self.base))
        for k in range(L):
            shifted = np.zeros(T, dtype=np.uint64)
            if T > k:
                shifted[k:] = xu[: T - k]
            # modular multiply-add (uint64 wraps by definition, but silence the warning)
            with np.errstate(over="ignore"):
                h = np.multiply(h, base, dtype=np.uint64)
                h = np.add(h, shifted, dtype=np.uint64)
        return h

    # ------------------------------------------------------------------ match
    def match(self, canon_ids: np.ndarray) -> np.ndarray:
        """canon_ids: (T,) -> (T, max_per_pos) int64 slot ids (0 == NULL)."""
        self.maybe_rebuild()
        T = int(canon_ids.shape[0])
        out = np.zeros((T, self.max_per_pos), dtype=np.int64)
        if T == 0 or not self._by_len:
            return out

        hits: List[Tuple[int, int, int]] = []  # (end_pos, -len, slot)
        pos_idx = np.arange(T, dtype=np.int64)
        for L, hashes in self._sorted_hashes.items():
            if L > T:
                continue
            h = self._rolling_hash(canon_ids, L)
            idx = np.searchsorted(hashes, h)
            idx_c = np.clip(idx, 0, len(hashes) - 1)
            hit = hashes[idx_c] == h
            cand = pos_idx[hit & (pos_idx >= L - 1)]
            if cand.size == 0:
                continue
            for p in cand:
                window = tuple(int(x) for x in canon_ids[p - L + 1: p + 1])
                bucket = self._buckets[L].get(h[p], ())
                for phrase in bucket:            # exact verification kills collisions
                    if phrase == window:
                        slot = self._slot_of.get(phrase)
                        if slot:
                            hits.append((int(p), -L, int(slot)))
                        break

        if not hits:
            return out
        hits.sort(key=lambda r: (r[0], r[1]))    # position asc, longest phrase first
        count: Dict[int, int] = {}
        for p, _, slot in hits:
            c = count.get(p, 0)
            if c < self.max_per_pos:
                out[p, c] = slot
                count[p] = c + 1
        return out

    def match_batch(self, canon_ids: np.ndarray) -> np.ndarray:
        """canon_ids: (B, T) -> (B, T, max_per_pos)."""
        self.maybe_rebuild()
        B = canon_ids.shape[0]
        return np.stack([self.match(canon_ids[b]) for b in range(B)], axis=0)

    # ------------------------------------------------------------------ tokenize helper
    def phrase_from_text(self, text: str, compressed_tokenizer) -> Tuple[Phrase, str]:
        """Canonical token-id tuple for a raw surface term (uses the compressed ids)."""
        ids = compressed_tokenizer.tokenizer(text, add_special_tokens=False).input_ids
        canon = compressed_tokenizer(np.asarray(ids, dtype=np.int64))
        return tuple(int(x) for x in canon), text
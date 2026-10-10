"""
hashing.py — O(1) deterministic addressing (adapted from DeepSeek Engram, Apache-2.0)

Implements:
  * CompressedTokenizer  : surjective V -> V' canonical-id projection (NFKC/NFD/lowercase/...)
  * NgramHashMapping     : multi-head multiplicative-XOR hashing of suffix N-grams into
                           prime-sized embedding tables (Section 2.2 of the Engram paper).

The hash is a *pure function of token ids* (deterministic, no runtime state), which is
exactly what makes the memory table CRUD-able and prefetchable -- the property we rely
on for "storage / compute decoupling".
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

try:
    from sympy import isprime
except Exception:  # pragma: no cover - fallback Miller-Rabin
    def isprime(n: int) -> bool:  # noqa
        if n < 2:
            return False
        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if n % p == 0:
                return n == p
        d, r = n - 1, 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            x = pow(a, d, n)
            if x in (1, n - 1):
                continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True


# --------------------------------------------------------------------------- #
# Tokenizer compression  (Engram Sec 2.2 "Tokenizer Compression")
# --------------------------------------------------------------------------- #
class CompressedTokenizer:
    """Project raw subword ids to canonical ids via surface-form normalization.

    Two raw tokens that decode to the same normalized string (e.g. "Apple" vs
    "␣apple", full-width vs half-width) collapse to ONE canonical id. This raises
    semantic density of N-grams (the paper reports a 23% vocab reduction on a 128k
    tokenizer) and is essential for a *domain* memory table where the same term
    appears with inconsistent casing/spacing.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.lookup_table, self.num_new_token = self._build_lookup_table()

    def __len__(self) -> int:
        return self.num_new_token

    @property
    def canonical_vocab_size(self) -> int:
        return self.num_new_token

    def _normalizer(self):
        from tokenizers import normalizers, Regex

        SENTINEL = "\ue000"
        return normalizers.Sequence(
            [
                normalizers.NFKC(),
                normalizers.NFD(),
                normalizers.StripAccents(),
                normalizers.Lowercase(),
                normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                normalizers.Replace(Regex(r"^ $"), SENTINEL),
                normalizers.Strip(),
                normalizers.Replace(SENTINEL, " "),
            ]
        )

    def _build_lookup_table(self):
        self.norm = self._normalizer()
        old2new: dict[int, int] = {}
        key2new: dict[str, int] = {}
        new_tokens: List[str] = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            if "\ufffd" in text:  # partial byte-token -> fall back to raw token string
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.norm.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid

        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]
        return lookup, len(new_tokens)

    @property
    def pad_id(self) -> int:
        return int(self.lookup_table[self.tokenizer.pad_token_id or 0])

    def to_torch(self, device=None) -> torch.Tensor:
        t = torch.from_numpy(self.lookup_table)
        return t.to(device) if device is not None else t

    def __call__(self, input_ids):
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        if pos_mask.any():
            out[pos_mask] = self.lookup_table[arr[pos_mask]]
        return out


# --------------------------------------------------------------------------- #
# N-gram hashing  (Engram Sec 2.2 "Multi-Head Hashing")
# --------------------------------------------------------------------------- #
def find_next_prime(start: int, seen_primes: set[int]) -> int:
    candidate = int(start) + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


class NgramHashMapping:
    """Multi-head hashed suffix N-gram addressing.

    For each N-gram order n in [2, N] and each of K heads we compute

        z_{t,n,k} = phi_{n,k}(g_{t,n}) = (XOR_i x_{t-i} * m_i) mod M_{n,k}

    and look up row ``z`` in table ``E_{n,k}`` (size = a distinct prime, so that the
    multiplicative hash is well distributed and collisions are minimized).

    ``z`` depends ONLY on the token-id sequence => fully deterministic, precomputable,
    and therefore *offline-addressable* (the basis for CRUD + prefetch).
    """

    def __init__(
        self,
        canonical_vocab_size: int,
        max_ngram_size: int = 3,
        table_size_per_ngram: Sequence[int] | int = 65536,
        n_head_per_ngram: int = 8,
        layer_ids: Sequence[int] = (1, 3),
        pad_id: int = 0,
        seed: int = 0,
    ):
        self.canonical_vocab_size = int(canonical_vocab_size)
        self.max_ngram_size = int(max_ngram_size)
        self.n_head_per_ngram = int(n_head_per_ngram)
        self.pad_id = int(pad_id)
        self.layer_ids = list(layer_ids)

        if isinstance(table_size_per_ngram, int):
            table_size_per_ngram = [table_size_per_ngram] * (self.max_ngram_size - 1)
        if len(table_size_per_ngram) != self.max_ngram_size - 1:
            raise ValueError("table_size_per_ngram must have max_ngram_size-1 entries")
        self.table_size_per_ngram = list(table_size_per_ngram)

        # ---- per-layer odd multipliers (avoids degenerate XOR) ---------------- #
        max_long = np.iinfo(np.int64).max
        m_max = int(max_long // max(self.canonical_vocab_size, 1))
        half_bound = max(1, m_max // 2)
        self.layer_multipliers: dict[int, np.ndarray] = {}
        for layer_id in self.layer_ids:
            rng = np.random.default_rng(seed + 10007 * int(layer_id))
            r = rng.integers(low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64)
            self.layer_multipliers[layer_id] = r * 2 + 1  # force odd

        # ---- distinct prime table sizes (shared across layers, globally unique) - #
        self.vocab_size_across_layers = self._calc_vocab_sizes()

    def _calc_vocab_sizes(self) -> dict[int, List[List[int]]]:
        seen: set[int] = set()
        out: dict[int, List[List[int]]] = {}
        for layer_id in self.layer_ids:
            per_ngram: List[List[int]] = []
            for n_idx in range(self.max_ngram_size - 1):
                base = self.table_size_per_ngram[n_idx]
                sizes: List[int] = []
                start = base - 1
                for _ in range(self.n_head_per_ngram):
                    p = find_next_prime(start, seen)
                    seen.add(p)
                    sizes.append(p)
                    start = p
                per_ngram.append(sizes)
            out[layer_id] = per_ngram
        return out

    # ------------------------------------------------------------------ #
    @property
    def n_tables_per_layer(self) -> int:
        return (self.max_ngram_size - 1) * self.n_head_per_ngram

    def total_slots_per_layer(self, layer_id: int) -> int:
        return sum(s for ns in self.vocab_size_across_layers[layer_id] for s in ns)

    # ------------------------------------------------------------------ #
    def _shift(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """x_{t-k} with left padding."""
        if k == 0:
            return x
        pad = torch.full((x.shape[0], k), self.pad_id, dtype=x.dtype, device=x.device)
        return torch.cat([pad, x[:, : x.shape[1] - k]], dim=1)

    def hash_layer(self, canon_ids: torch.Tensor, layer_id: int) -> torch.Tensor:
        """canon_ids: (B, T) int64 -> (B, T, n_tables_per_layer) int64 row indices."""
        mult = torch.as_tensor(self.layer_multipliers[layer_id], device=canon_ids.device)
        shifts = [self._shift(canon_ids, k) for k in range(self.max_ngram_size)]

        out = []
        for n in range(2, self.max_ngram_size + 1):
            mix = shifts[0] * mult[0]
            for k in range(1, n):
                mix = torch.bitwise_xor(mix, shifts[k] * mult[k])
            mix = mix.to(torch.int64)
            for p in self.vocab_size_across_layers[layer_id][n - 2]:
                out.append(torch.remainder(mix, p))
        return torch.stack(out, dim=2)

    def hash(self, canon_ids: torch.Tensor, layer_ids: Optional[Sequence[int]] = None):
        layer_ids = self.layer_ids if layer_ids is None else layer_ids
        return {lid: self.hash_layer(canon_ids, lid) for lid in layer_ids}
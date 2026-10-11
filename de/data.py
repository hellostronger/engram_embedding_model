"""
data.py — domain triplet dataset + collate that also produces registry slot tensors.

A sample is (query, positive, [hard negatives]). The collate function tokenizes,
masks, computes canonical ids and *runs the PhraseMatcher on CPU* so the GPU only
performs gather + fusion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Triplet:
    query: str
    positive: str
    negatives: List[str]


class TripletDataset(Dataset):
    def __init__(self, triplets: Sequence[Triplet]):
        self.triplets = list(triplets)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, i):
        return self.triplets[i]

    @classmethod
    def from_jsonl(cls, path: str, max_negatives: int = 1) -> "TripletDataset":
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                negs = d.get("negatives", [])
                if isinstance(negs, str):
                    negs = [negs]
                out.append(
                    Triplet(
                        query=d["query"],
                        positive=d.get("positive", d.get("doc", "")),
                        negatives=negs[:max_negatives],
                    )
                )
        return cls(out)


class DECollator:
    """Builds (input_ids, attention_mask, registry_slots) for q / pos / neg streams.

    registry_slots for each stream is a (B, T, R) int tensor produced by the model's
    PhraseMatcher on CANONICAL ids. The stream is flattened into one batch to keep it
    simple.
    """

    def __init__(self, tokenizer, matcher, max_len: int = 128,
                 add_special_tokens: bool = True, lookup_table=None,
                 query_max_len: Optional[int] = None):
        self.tok = tokenizer
        self.matcher = matcher
        self.max_len = max_len
        self.query_max_len = query_max_len or max_len
        self.add_special_tokens = add_special_tokens
        if lookup_table is None:
            raise ValueError(
                "DECollator needs the CompressedTokenizer.lookup_table so that phrase "
                "matching runs on canonical ids, matching the model at forward time. "
                "Pass lookup_table=model.compressed.lookup_table."
            )
        self.lookup_table = lookup_table

    def _encode(self, texts: List[str], max_len: Optional[int] = None) -> Dict[str, torch.Tensor]:
        ml = max_len or self.max_len
        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=ml,
            add_special_tokens=self.add_special_tokens,
            return_tensors=None,
        )
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        # PhraseMatcher needs a rectangular (B, T) int array; pad rows are filled with
        # the tokenizer's pad id, which we point at the canonical pad id so a padded
        # position can never spuriously match a registry phrase.
        T = max(len(x) for x in ids)
        pad_id = getattr(self.tok, "pad_token_id", 0) or 0
        ids_arr = np.full((len(ids), T), pad_id, dtype=np.int64)
        mask_arr = np.zeros((len(ids), T), dtype=np.int64)
        for i, (row, mrow) in enumerate(zip(ids, mask)):
            ids_arr[i, : len(row)] = row
            mask_arr[i, : len(mrow)] = mrow

        # The matcher must run on CANONICAL ids, exactly as the model does at forward
        # time (model._canonical_ids). Matching on raw ids would silently miss every
        # phrase whose surface form collapses under compression.
        canon = np.asarray([self.lookup_table[i] for i in ids_arr], dtype=np.int64)
        slots = self.matcher.match_batch(canon)
        return {
            "input_ids": torch.from_numpy(ids_arr),
            "attention_mask": torch.from_numpy(mask_arr),
            "registry_slots": torch.from_numpy(slots).long(),
        }

    @staticmethod
    def _field(item, key: str, default=None):
        """Read a field from either a plain dict or a Triplet dataclass.

        TripletDataset is a transparent container: offline_test.py feeds it Triplet
        objects while run_abcd.py feeds it dicts parsed straight from jsonl. Both
        are legitimate, so the collator accepts either instead of forcing one.
        """
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    def __call__(self, batch: Sequence[Triplet]) -> Dict[str, Dict[str, torch.Tensor]]:
        queries = [self._field(b, "query") for b in batch]
        pos = [self._field(b, "positive") for b in batch]
        # _field defaults to None on a missing key; surface that loudly rather than
        # letting a None reach the tokenizer as an un-debuggable TypeError.
        for i, (q, p) in enumerate(zip(queries, pos)):
            if not isinstance(q, str) or not q:
                raise ValueError(f"batch[{i}] has no usable 'query': {q!r}")
            if not isinstance(p, str) or not p:
                raise ValueError(f"batch[{i}] has no usable 'positive': {p!r}")
        negs: List[str] = []
        for b in batch:
            n = self._field(b, "negatives", []) or []
            if isinstance(n, str):
                n = [n]
            negs.extend(n)

        out = {
            "query": self._encode(queries, self.query_max_len),
            "positive": self._encode(pos),
        }
        if negs:
            out["negative"] = self._encode(negs)
        return out
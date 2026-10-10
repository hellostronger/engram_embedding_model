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
                 add_special_tokens: bool = True, lookup_table=None):
        self.tok = tokenizer
        self.matcher = matcher
        self.max_len = max_len
        self.add_special_tokens = add_special_tokens
        if lookup_table is None:
            raise ValueError(
                "DECollator needs the CompressedTokenizer.lookup_table so that phrase "
                "matching runs on canonical ids, matching the model at forward time. "
                "Pass lookup_table=model.compressed.lookup_table."
            )
        self.lookup_table = lookup_table

    def _encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
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

    def __call__(self, batch: Sequence[Triplet]) -> Dict[str, Dict[str, torch.Tensor]]:
        queries = [b.query for b in batch]
        pos = [b.positive for b in batch]
        negs: List[str] = []
        for b in batch:
            negs.extend(b.negatives)

        out = {
            "query": self._encode(queries),
            "positive": self._encode(pos),
        }
        if negs:
            out["negative"] = self._encode(negs)
        return out
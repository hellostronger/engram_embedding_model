"""
de — Domain Engram: an Engram-style conditional-memory embedding model with a
dynamically CRUD-addressable domain memory.

Adapted from DeepSeek Engram (Conditional Memory via Scalable Lookup, arXiv:2601.07372,
Apache-2.0 https://github.com/deepseek-ai/Engram).
"""

from .config import EngramConfig, default_config
from .hashing import CompressedTokenizer, NgramHashMapping
from .registry import PhraseMatcher, PhraseRegistry, RegistryTable
from .engram import EngramFusion, EngramTableHead, DepthwiseCausalConv
from .model import DomainEngramEncoder
from .losses import info_nce, info_nce_with_hard_negatives
from .data import DECollator, Triplet, TripletDataset
from .evaluation import evaluate_retrieval, retrieval_metrics
from .cold_start import cold_start, solve_slot_closed_form, refine_slot

__all__ = [
    "EngramConfig", "default_config",
    "CompressedTokenizer", "NgramHashMapping",
    "PhraseRegistry", "RegistryTable", "PhraseMatcher",
    "EngramFusion", "EngramTableHead", "DepthwiseCausalConv",
    "DomainEngramEncoder",
    "info_nce", "info_nce_with_hard_negatives",
    "Triplet", "TripletDataset", "DECollator",
    "evaluate_retrieval", "retrieval_metrics",
    "cold_start", "solve_slot_closed_form", "refine_slot",
]

__version__ = "0.1.0"
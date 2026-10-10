"""
cold_start.py — instant, training-free initialization of a NEW registry term.

Motivation
----------
Engram's hashed table cannot be addressed, so a new domain term has no meaningful
vector: a randomly initialized slot will (at best) be ignored by the context gate.
This module makes the *new* slot useful immediately, in three escalating steps:

  Step 1 (aggregate)   Collect the frozen backbone's representation of the term over a
                       small support corpus -> target semantic state `s*`.

  Step 2 (closed form) Solve for the slot row `e` such that the Engram Value projection
                       injects exactly the correction needed to move the plain hidden
                       state at the term's position to `s*`:

                           min_e || W_V [e_hash ; e] - (s* - h_plain) ||^2 + λ||e||^2

                       This is a ridge-regularized least-squares with the *known* hash
                       part folded into the residual, hence a closed-form solution:

                           e = (W_Vr^T W_Vr + λI)^-1 W_Vr^T (target - W_Vh e_hash)

                       No gradient steps, no retraining: microseconds per new term.

  Step 3 (refine)      Optionally run a handful of gradient steps on ONLY that slot row
                       (LoRA-style single-row fine-tune) to sharpen it. Backbone, fusion
                       and all other rows stay frozen.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from .model import attention_bias

Phrase = Tuple[int, ...]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> Optional[int]:
    n, m = len(haystack), len(needle)
    if m == 0 or n < m:
        return None
    for i in range(n - m + 1):
        if list(haystack[i: i + m]) == list(needle):
            return i
    return None


def _last_occurrence(token_ids: List[int], target: List[int]) -> Optional[int]:
    n, m = len(token_ids), len(target)
    if m == 0 or n < m:
        return None
    for i in range(n - m, -1, -1):
        if list(token_ids[i: i + m]) == list(target):
            return i
    return None


# --------------------------------------------------------------------------- #
# forward variants
# --------------------------------------------------------------------------- #
def _embeddings(model, input_ids):
    return model.backbone.embeddings(input_ids=input_ids)


def _forward_backbone(model, h, attention_mask, start_layer: int = 0):
    bias = attention_bias(attention_mask, h.dtype)
    for idx in range(start_layer, len(model.blocks)):
        out = model.blocks[idx](h, attention_mask=bias)
        h = out[0] if isinstance(out, (tuple, list)) else out
    return h


@torch.no_grad()
def backbone_state_at(model, tokenizer, term: str, contexts: List[str],
                      max_len: int = 192, device=None):
    """Frozen-backbone hidden state at the last token of `term`, averaged over contexts.

    Memory is suppressed so the estimate is purely generic semantics (step 1).
    """
    device = device or model.device
    prev = model.memory_enabled
    model.memory_enabled = False
    model.eval()

    term_ids = tokenizer(term, add_special_tokens=False).input_ids
    vecs = []
    for ctx in contexts:
        enc = tokenizer(ctx, return_tensors="pt", truncation=True, max_length=max_len)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        end = _last_occurrence(ids[0].tolist(), term_ids)
        if end is None or end >= ids.shape[1]:
            continue
        h = _forward_backbone(model, _embeddings(model, ids), mask, start_layer=0)
        vecs.append(h[0, end].float())

    model.memory_enabled = prev
    if not vecs:
        return None
    return torch.stack(vecs).mean(dim=0)


@torch.no_grad()
def _engram_parts(model, token_ids: List[int], layer_id: int, device,
                  return_slots: bool = False):
    """Return (e_hash, e_reg) at the last position of `token_ids` for one layer.

    If ``return_slots`` is set, also return the (R,) slot-id array so callers know
    which chunk of the registry block corresponds to which phrase.
    """
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    canon = model._canonical_ids(ids)
    head = getattr(model, f"engram_head_{layer_id}")
    hash_ids = model.hasher.hash_layer(canon, layer_id)
    e_hash = head.lookup(hash_ids)[0, -1]                       # (d_mem - d_registry,)
    # registry contribution: match at the final position only
    canon_np = canon[0].cpu().numpy()
    slots_np = model.matcher.match(canon_np)[-1]                # (R,)
    slots_t = torch.from_numpy(slots_np).to(device).view(1, 1, -1)
    e_reg = model._registry_vecs(slots_t)[0, -1]                # (d_registry * R,)
    if return_slots:
        return e_hash, e_reg, slots_t.view(-1)
    return e_hash, e_reg


@torch.no_grad()
def _engram_hash_and_slots(model, token_ids: List[int], layer_id: int, device):
    """Return (e_hash, slots) for one context. e_reg is intentionally NOT returned:
    the solve reconstructs the known registry contribution itself."""
    e_hash, _e_reg, slots = _engram_parts(
        model, token_ids, layer_id, device, return_slots=True)
    return e_hash, slots


# --------------------------------------------------------------------------- #
# Step 2: closed-form ridge solve
# --------------------------------------------------------------------------- #
@torch.no_grad()
def solve_slot_closed_form(
    model,
    tokenizer,
    term: str,
    contexts: List[str],
    slot: int,
    layer_id: Optional[int] = None,
    ridge: float = 1e-3,
    max_len: int = 192,
    device=None,
) -> bool:
    """Write a usable vector into `slot` with zero gradient steps.

    Formulation
    -----------
    The memory must give the term a *stable* representation, so we define the target as
    the **cross-context consensus** of the term's final-token state:

        s* = mean_ctx  h_plain(ctx)[end]        (memory suppressed => pure backbone)

    For each support context we then want the injected Value output to move the state at
    the injection layer toward that consensus:

        W_V [e_hash(ctx) ; e]  ~=  s* - h_pre(ctx)[end]

    where the hashed block ``e_hash(ctx)`` is *known* and folded into the residual,
    leaving a linear least-squares in the registry block only:

        e = (W_Vr^T W_Vr + λI)^-1 W_Vr^T ( s* - h_pre - W_Vh e_hash )

    NOTE (important): the target is computed with memory *suppressed* and h_pre likewise,
    so the two are directly comparable; using the consensus rather than the per-context
    h_plain is what makes the residual non-degenerate -- a per-context target would equal
    h_pre plus backbone drift and yield a near-zero (uninformative) delta.

    Returns True if at least one context contributed.
    """
    device = device or model.device
    layer_id = model.layer_ids[0] if layer_id is None else layer_id
    head = getattr(model, f"engram_head_{layer_id}")
    fusion = head.fusion
    n_hash = head.num_tables * model.cfg.d_head
    d_reg = model.cfg.d_registry
    R = model.cfg.max_matches_per_pos

    W = fusion.value_proj.weight.float()                       # (d_model, d_mem)
    W_h = W[:, :n_hash]                                        # hash block
    W_r = W[:, n_hash:]                                        # registry block

    term_ids = tokenizer(term, add_special_tokens=False).input_ids
    prev = model.memory_enabled
    model.memory_enabled = False
    model.eval()

    # ---- pass 1: collect (h_pre_at_end, e_hash) per context --------------- #
    records = []
    consensus: List[torch.Tensor] = []
    for ctx in contexts:
        enc = tokenizer(ctx, return_tensors="pt", truncation=True, max_length=max_len)
        raw_ids = enc["input_ids"][0].tolist()
        end = _last_occurrence(raw_ids, term_ids)
        if end is None or end >= len(raw_ids):
            continue
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        # state at the injection layer (this is what memory is added to)
        h_pre = _embeddings(model, ids)
        bias = attention_bias(mask, h_pre.dtype)
        for idx in range(layer_id):
            out = model.blocks[idx](h_pre, attention_mask=bias)
            h_pre = out[0] if isinstance(out, (tuple, list)) else out

        # full-backbone state at the term's final token -> contributes to consensus
        h_full = _forward_backbone(model, _embeddings(model, ids), mask, start_layer=0)

        e_hash, slots = _engram_hash_and_slots(model, raw_ids, layer_id, device)
        where = (slots == slot).nonzero(as_tuple=False)
        r_star = int(where[0]) if where.numel() else 0
        records.append((h_pre[0, end].float(), e_hash.float(), slots, r_star))
        consensus.append(h_full[0, end].float())

    if not records:
        model.memory_enabled = prev
        return False

    # ---- target = cross-context consensus -------------------------------- #
    s_star = torch.stack(consensus).mean(dim=0)

    # ---- pass 2: solve a d_reg-sized system per context, then average ------ #
    # Only the chunk of e_reg holding OUR slot is a free variable; every other chunk
    # belongs to a different table row and is held at its current value.
    solutions: List[torch.Tensor] = []
    for h_pre_end, e_hash, slots, r_star in records:
        W_self = W_r[:, r_star * d_reg: (r_star + 1) * d_reg]
        A = W_self.t() @ W_self + ridge * torch.eye(
            d_reg, device=W_self.device, dtype=W_self.dtype)
        known = torch.zeros_like(h_pre_end)
        for r in range(R):
            if r == r_star:
                continue
            s = int(slots[r]) if slots.numel() > r else 0
            if s == 0:
                continue
            row = model.registry_table.emb.weight[s].float()
            known = known + W_r[:, r * d_reg: (r + 1) * d_reg] @ row
        residual = (s_star - h_pre_end) - (W_h @ e_hash) - known
        solutions.append(torch.linalg.solve(A, W_self.t() @ residual))

    model.memory_enabled = prev
    if not solutions:
        return False

    sol = torch.stack(solutions).mean(dim=0)
    w = model.registry_table.emb.weight
    w[slot] = sol.to(w.dtype).to(w.device)
    return True


# --------------------------------------------------------------------------- #
# Step 3: targeted refinement (optional)
# --------------------------------------------------------------------------- #
def refine_slot(
    model,
    tokenizer,
    term: str,
    slot: int,
    contexts: List[str],
    steps: int = 20,
    lr: float = 0.05,
    max_len: int = 192,
    device=None,
) -> None:
    """Train ONLY the new slot row so the term matches its aggregated target.

    Loss = 1 - cos(hidden_at_term_with_memory, target). Everything else stays frozen.
    """
    device = device or model.device
    layer_id = model.layer_ids[0]
    term_ids = tokenizer(term, add_special_tokens=False).input_ids

    prev_mem = model.memory_enabled
    model.memory_enabled = False
    with torch.no_grad():
        items = []  # (ids, mask, end, target)
        for ctx in contexts:
            enc = tokenizer(ctx, return_tensors="pt", truncation=True, max_length=max_len)
            raw = enc["input_ids"][0].tolist()
            end = _last_occurrence(raw, term_ids)
            if end is None or end >= len(raw):
                continue
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            h = _forward_backbone(model, _embeddings(model, ids), mask, 0)
            items.append((ids, mask, end, h[0, end].float()))
    model.memory_enabled = prev_mem
    if not items:
        return

    w = model.registry_table.emb.weight
    w.requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    was_training = model.training
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        total = 0.0
        for ids, mask, end, tgt in items:
            h = _forward_backbone_with_memory(model, ids, mask, layer_id)
            total = total + (1 - torch.nn.functional.cosine_similarity(
                h[0, end].unsqueeze(0), tgt.unsqueeze(0), dim=-1)).mean()
        loss = total / len(items)
        loss.backward()
        opt.step()
    w.requires_grad_(False)
    model.train(was_training)


def _forward_backbone_with_memory(model, ids, mask, layer_id: int):
    h = _embeddings(model, ids)
    canon = model._canonical_ids(ids)
    bias = attention_bias(mask, h.dtype)
    for idx, block in enumerate(model.blocks):
        if idx == layer_id:
            head = getattr(model, f"engram_head_{idx}")
            hash_ids = model.hasher.hash_layer(canon, idx)
            slots = model.prepare_registry_slots([ids[0].tolist()]).to(ids.device)
            y, _ = head(h, hash_ids, model._registry_vecs(slots))
            h = h + y
        out = block(h, attention_mask=bias)
        h = out[0] if isinstance(out, (tuple, list)) else out
    return h


def _slots_for(self, ids, term, layer_id):
    return self.prepare_registry_slots([ids[0].tolist()]).to(ids.device)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def cold_start(
    model,
    tokenizer,
    term: str,
    contexts: List[str],
    refine_steps: int = 0,
    refine_lr: float = 0.05,
    ridge: float = 1e-3,
    device=None,
) -> int:
    """Allocate a slot for `term` and write a usable vector into it (no retraining).

    Returns the slot id. Idempotent: re-adding an existing term reuses its slot.
    """
    device = device or model.device
    model.sync_matcher()

    ids = tokenizer(term, add_special_tokens=False).input_ids
    canon = model.compressed(torch.tensor([ids], dtype=torch.long))[0].tolist()
    slot = model.registry.add(canon, surface=term)
    model.registry_table.reset_slot(slot)
    model.sync_matcher()

    ok = solve_slot_closed_form(
        model, tokenizer, term, contexts, slot,
        layer_id=model.layer_ids[0] if model.layer_ids else None,
        ridge=ridge, device=device,
    )

    if refine_steps > 0 and ok:
        refine_slot(
            model, tokenizer, term, slot, contexts,
            steps=refine_steps, lr=refine_lr, device=device,
        )
    return slot
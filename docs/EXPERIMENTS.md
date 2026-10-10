# EXPERIMENTS.md — Domain Engram: Comparison & Ablation Experiment Matrix

> Companion to `docs/PLAN.md`. PLAN.md says *what* to build; this document says **how we will try to
> prove it is wrong**. Every experiment below is specified so that it can *falsify* the Domain Engram
> (DE) hypothesis. Nothing here is a "confirmatory" test by construction.
>
> Author: experiment-design subagent · Date: 2026-10-10 · Plan revision target: PLAN.md §5
>
> **Sandbox note:** the development sandbox has no GPU and blocks HuggingFace downloads, so *nothing
> in this document was executed*. All numbers marked `[REF]` are copied from published sources and
> were verified by web search; all timings are `[EST]` engineering estimates that **must be replaced
> by the M0 calibration measurement** (§7 Phase 0) before they are quoted.
>
> Items I could not verify are explicitly tagged `UNVERIFIED`. No identifier in this document is
> invented; where an id could not be confirmed it is left out rather than guessed.

---

## 0. TL;DR — the thesis, and how we will kill it

**Thesis.** A frozen bi-encoder plus a *writable* Engram-style memory (hashed N-gram tables + a CRUD
term registry) can be domain-adapted for retrieval at (a) lower trainable-parameter cost than LoRA at
matched budget, and (b) **zero retraining cost** when the domain vocabulary changes after deployment.

Three claims, three falsifiers:

| # | Claim | Falsifier |
|---|-------|-----------|
| **C1** | Memory improves domain retrieval over the frozen backbone | `A` ≈ `C` on DomainTier nDCG@10 |
| **C2** | The memory — not just extra capacity — is the mechanism | a matched-capacity **LoRA** or **CPT** on the same data reaches ≥ `C` |
| **C3** | The **writable registry** lets you fix new terms after deployment | post-hoc writes (`D`) do not beat `C` on terms unseen in training |

If **C2** fails, the honest paper is "a cheap-to-update but not better" method, and only C3 can
rescue it. If **C3** fails, DE collapses into plain Engram-for-embeddings and the CRUD registry —
the paper's *only* claimed novelty over the Engram paper — is dead. We pre-commit to reporting that
outcome as a negative result (§6).

---

## 1. Reference configuration `DE-Ref`

Everything is stated relative to one frozen reference. No experiment deviates from this without a
logged reason.

### 1.1 Backbone (tier S — the fast iteration model)

| Property | Value | Source |
|---|---|---|
| Model id | `BAAI/bge-small-zh-v1.5` | `[REF]` model card |
| Params | 24 M total; ≈12.58 M non-embedding | `[REF]` card + arithmetic |
| Layers | **4** | `[REF]` model config |
| Hidden / heads / FFN | 512 / 8 / 2048 | `[REF]` model config |
| Vocab | 21 128 · max seq 512 · emb dim 512 | `[REF]` model config |
| Pooling | CLS + L2 norm (BGE convention) | `[REF]` BGE docs |

> **Correction to PLAN.md.** PLAN.md §4 states "BGE-small-zh-v1.5 / bge-base-zh-v1.5（中文，384/768 维）"
> and "BGE-small-zh（33M 级）". Both are wrong: `bge-small-zh-v1.5` is **512-dim, 24 M params, 4
> layers** (33 M / 384-dim is `bge-small-en-v1.5`, an English model). This matters a great deal — a
> 4-layer encoder is far more hostile to "inject at layer 2 and let 10 layers refine" (§4.4).

**Published C-MTEB anchors for group A** (`[REF]`, FlagEmbedding / C-MTEB 1.1.1 leaderboard,
100k sampled candidates per query):

| Metric | `bge-small-zh-v1.5` | `bge-base-zh-v1.5` | `bge-large-zh-v1.5` |
|---|---|---|---|
| Avg | 57.82 | 63.13 | 64.53 |
| **Retrieval** | **61.77** | **69.49** | **70.46** |
| STS | 49.11 | 53.72 | 56.25 |
| Classification | 63.96 | 68.07 | 69.13 |
| Clustering | 44.18 | 47.53 | 48.99 |
| Embedding dim | 512 | 768 | 1024 |

`A` must land within **±0.5 nDCG@10** of 61.77 on C-MTEB Retrieval or the whole harness is invalid (§7
Phase 0). Per-task C-MTEB numbers for the `-v1.5` checkpoints are **not published** (`UNVERIFIED`);
only `bge-small-zh` (v1) per-task values exist (Ecom 58.17 / Medical 49.90 / Cmedqa 35.18 /
Covid 68.95 / Video 69.33, retrieval avg 63.07) — use these as an order-of-magnitude sanity band only.

### 1.2 Tier-B backbone (scale-up, run only if Phase 4 passes)

`BAAI/bge-base-zh-v1.5` — 12 layers, hidden 768, 768-dim embeddings, C-MTEB Avg 63.13 /
Retrieval 69.49 `[REF]`. 12 layers is the configuration in which Engram's "layer 2 is optimal"
finding was measured, so the layer sweep must be repeated here before any layer claim is published
(§3.4).

### 1.3 DE reference hyperparameters `DE-Ref-S`

| Knob | Value | Rationale |
|---|---|---|
| `layer_ids` | `[1]` (0-indexed) | 4-layer model; ≈ layer 2 of 4 |
| `max_ngram_size` | 3 (orders {2,3}) | Engram reference config uses {2,3} `[REF]` Sec 6.2 |
| `n_head_per_ngram` | 8 | Engram reference config uses K=8 `[REF]` Sec 4.1 |
| `table_size_per_ngram` | 16 384 (distinct primes ≥) | chosen to hit budget S (§1.4) |
| `d_head` | 16 | ditto |
| Registry slots / dim | 50 000 × 32 | ditto |
| `max_matches_per_pos` | 4 | matches current `EngramConfig` |
| `conv_kernel_size` / dilation | 4 / 3 | Engram Eq. 5 `[REF]` |
| `gate_bias_init` | −2.0 | current code; keeps α ≈ 0.12 at init |
| `pooling` | CLS | BGE convention — **ablated**, see OAT-12 |
| InfoNCE temp | 0.02 | current code; BGE-family standard |
| fusion opt | AdamW, lr 2e-5, wd 0.01, 10 % linear warmup → cosine | |
| memory opt | Adam, lr ×5, **wd 0** | Engram Sec 4.1 `[REF]` |
| conv init | zeros (identity at step 0) | Engram Sec 4.1 `[REF]` |
| train `max_len` / eval `max_len` | 128 / 512 | held constant across *all* arms (§5.4) |

### 1.4 Parameter budget arithmetic (the iso-parameter anchor)

```
hashed N-gram tables  : 2 orders × 8 heads = 16 tables × 16 384 × 16 d_head = 4 194 304
registry table        : 50 000 slots × 32                                       = 1 600 000
fusion (1 site)       : d_mem = 16·16 + 32·4 = 384
                        W_V 384×512 = 196 608 ; W_K 384×512 = 196 608
                        RMSNorm_q/k/v 3×512 = 1 536 ; gate_bias 1 ; conv 512×4 = 2 048
                                                                       total =  396 800
-------------------------------------------------------------------------------------
DE-Ref-S trainable                                            ≈ 6 191 104  (6.19 M)
backbone non-embedding params                                           12 582 912  (12.58 M)
backbone total                                                          ≈ 24 M
```

**DE-Ref-S trains 49 % of the backbone's own transformer parameters.** PLAN.md's current default
config (`d_head=32`, `table_size=65536`, `200 000 × 64` registry) yields
`16·65 536·32 + 200 000·64 + 0.79 M ≈ 33.55 + 12.80 + 0.79 = 47.1 M` trainable — **~2× the entire
backbone**. That config cannot be used for any iso-parameter claim; `DE-Ref-S` replaces it. Four
budgets are used throughout:

| Budget | Hashed tables | Registry | d_head | Trainable |
|---|---|---|---|---|
| **XS** (cheapest kill-shot) | 16 × 4 096 | 10 k × 16 | 8 | **≈ 0.88 M** |
| **S (reference)** | 16 × 16 384 | 50 k × 32 | 16 | **≈ 6.19 M** |
| **L** | 16 × 16 384 | 100 k × 64 | 32 | **≈ 15.58 M** |
| **XL** | 16 × 65 536 | 200 k × 64 | 32 | **≈ 47.1 M** |

(fusion adds 0.20 / 0.40 / 0.79 / 0.79 M for XS / S / L / XL respectively)

### 1.5 Training-FLOP arithmetic (the iso-compute anchor)

Per-token MACs, `bge-small-zh-v1.5`, non-embedding `P = 12.58 M`, attention term negligible:

| Arm | fwd | bwd | step cost |
|---|---|---|---|
| Full fine-tune | 2P | 4P | **6P** |
| LoRA (all layers) | 2P | 4P (activation grads still needed everywhere) | **≈6P** |
| **DE, frozen, inject @ layer 1 of 4** | 2P | 4P·(3/4) | **5P** |
| DE, frozen, inject @ layer 3 of 4 | 2P | 4P·(1/4) | 3P |

DE module's own MACs/token: `W_V + W_K = 2·384·512 = 393 216` → **+3.1 %** over the backbone.
Adam state cost ≈ 6 FLOP/param/step ≈ 37 MFLOP per *step* → **+0.01 %/token** at batch 64 × 128
(8 192 tokens/step). Embedding gather + its backward on touched rows adds ≤ +0.2 %/token. **Measure
this in P0; do not quote it from this table.**

> **Honest consequence.** On a *4-layer* encoder, freezing the backbone and injecting at layer 1
> saves only **17 %** of training FLOPs. The compute advantage of DE is **not** the story. DE's real
> advantages are (i) *trainable-parameter count* and (ii) *post-deployment update latency*. Any write-up
> that claims a compute win is wrong at our scale and must not be made.

**Compute-matched protocol.** With DE at batch `B`, the compute-matched baseline batch is
`B' = round(6P / 5P · B) = 1.20·B`. At `B = 64` → `B' = 77`. Because a larger InfoNCE batch also gives
the baseline *more in-batch negatives*, we run **both** arms and report separately:
* `matched-batch` (B = 64 everywhere; matched negatives, DE gets more FLOPs per step), and
* `matched-FLOPs` (baseline B = 77; matched compute, baseline gets more negatives).
Reporting only one of these is a fatal protocol error.

### 1.6 Eval harness tiers

| Tier | Contents | Cost | Used for |
|---|---|---|---|
| **Tier-0 (smoke)** | 200 queries × 1 000-candidate subsample from EcomRetrieval + MedicalRetrieval | ~1 min | smoke tests, kill-shot |
| **Tier-D (domain)** | C-MTEB `EcomRetrieval`(1000 q), `MedicalRetrieval`(1000), `CmedqaRetrieval`(3999), `VideoRetrieval`(1000), `CovidRetrieval`(949) — nDCG@10 over 100 000 sampled candidates `[REF]` | ~20 min/arm | **primary metric** |
| **Tier-G (general guard)** | C-MTEB `T2Retrieval`(24 832), `MMarcoRetrieval`(7 437), `DuRetrieval`(4 000) | ~90 min/arm | forgetting guard |
| **Tier-S (STS guard)** | `ATEC, BQ, LCQMC, PAWSX, STSB, AFQMC, QBQTC` (Spearman) | ~15 min/arm | forgetting guard |
| **Tier-X (full)** | all of C-MTEB + MTEB(cmn,v1) | ~4 h/arm | final report only |
| **Tier-CS (cold-start)** | custom held-out term split, §3.5 | ~10 min | **headline metric** |

All dataset names/sizes above are `[REF]` from the C-MTEB 1.1.1 / FlagEmbedding task table. Retrieval
protocol follows C-MTEB: **100 000 candidates sampled per query including the gold document**.

---

## 2. Control groups A / B / C / D (+ the controls PLAN.md conflates)

PLAN.md §5 lists four groups but describes B as "memory present but random / no gate", which are
**two different controls**. They are separated here, and two controls PLAN.md omits entirely are added
(`A1` and `F`), because without them the central claim "the memory, not the extra capacity, is what
helps" is unfalsifiable.

| ID | Config | Trained | Frozen | Hypothesised result & reasoning |
|---|---|---|---|---|
| **A0** | `bge-small-zh-v1.5`, no memory, **zero trainable params** | nothing | everything | Reproduces the published 61.77 C-MTEB Retrieval ±0.5 `[REF]`. This is the *reference point*, not a competitor. |
| **A1** | frozen backbone + **6.19 M trainable projection/gating head** (2-layer MLP `512→1024→512`, randomly init, trained with the same InfoNCE data/steps) | the head | backbone | **≈ A0, maybe +0.5–1.5.** This is the *capacity control*: any gain of `C` over `A0` that `A1` also gets is not a memory effect. **This is the most important control in the matrix.** |
| **B1** | memory present, **randomly initialised, never trained**, gate active, residual injection on | nothing | everything | **≈ A0 ± 0.5.** Engram's design says random `e_t` gives an essentially arbitrary `v_t`; the gate cannot suppress it because `α` is computed from a random key. Expect a *small* random degradation, not a collapse, because conv is zero-init and `W_V` is small at init. Tests "does the module just add noise". |
| **B2** | **memory trained, gate removed** (`α ≡ 1`) | tables, W_V, conv | backbone, W_K, gate params | **Worse than C by the largest single-component margin.** Engram Sec 6.2 ranks context-aware gating #2 of 3 components `[REF]`. If B2 ≈ C, the gating claim does not transfer to bi-encoders → **falsifier F4**. |
| **B3** | **memory trained, `W_V := 0`** (lookup happens, value path dead) | tables, W_K, gate, conv | backbone, W_V | **≈ A0.** Isolates "gradients flow into the tables" from "the injection path carries information". A cheap sanity control: if B3 ≠ A0 the harness is leaking. |
| **C** | **full DE**, `DE-Ref-S`, trained Stage-1 | tables + registry + fusion | backbone | **> A0 on Tier-D.** Reasoning: Engram Sec 6.3 shows that suppressing the memory at inference collapses *factual-knowledge* tasks to 29–44 % of score while reading comprehension retains 81–93 % `[REF]`. Retrieval of domain terms is fact-lookup-like, not context-grounded reading comprehension, so we predict DE >> 0 sensitivity. This is the paper-grounded expectation we are testing. |
| **D** | **`C` + post-hoc registry writes**, zero retraining, on terms unseen in training | **only the new slot rows** (closed-form ridge solve; optional single-row refine) | everything else, incl. hashed tables and backbone | **D > C on Tier-CS, ≈ C on Tier-D/G** (writes touch only unseen terms). If D also improves Tier-D, that means the written terms are not actually unseen → contamination, not a win. |
| **E** | `C` **+ retrain after the same writes** (full Stage-1 replay including the new documents) | everything | backbone | **Oracle upper bound** for "what you'd get if you could afford to retrain". If `D ≥ E`, the no-retraining claim is a *strict* win. If `E >> D`, we must say so. |
| **F** | `C` retrained **with the cold-start documents included from the start** (oracle: model saw the terms) | everything | backbone | **Oracle ceiling for the terms themselves.** If `D ≈ F`, the post-hoc write fully recovers what training-on-those-terms would give → the killer result. If `F >> D`, the cold-start claim is much weaker than "no retraining needed". |
| **G** | `C` + poisoned registry entries (deliberately wrong vectors for N common terms), then deleted | nothing | everything | **Recover to ≥ 98 % of the pre-poison score** (tombstone + gate safety). Gate is the safety mechanism (PLAN §1.3); this tests it. |

**Minimum claim set:** the paper may claim *anything* only if `C > A0` **and** `C ≥ A1` **and**
`C ≥ best iso-parameter baseline` **and** `D > C` on Tier-CS. All four, or no claim.

---

## 3. External baselines

### 3.1 Published encoders — **context, not competitors**

These are *different backbones*. They tell us the ceiling, not whether our mechanism works. They are
reported in the final table and never used to justify or refute C1/C2.

| Model id | Params | Emb dim | C-MTEB Avg | C-MTEB Retrieval | Source |
|---|---|---|---|---|---|
| `BAAI/bge-large-zh-v1.5` | 325 M | 1024 | 64.53 | 70.46 | `[REF]` C-MTEB 1.1.1 |
| `BAAI/bge-base-zh-v1.5` | 102 M | 768 | 63.13 | 69.49 | `[REF]` |
| `BAAI/bge-small-zh-v1.5` | 24 M | 512 | 57.82 | 61.77 | `[REF]` |
| `intfloat/multilingual-e5-small` | 118 M | 384 | 55.38 | 59.95 | `[REF]` (12 layers, 512 ctx, mean-pool) |
| `BAAI/bge-m3` | 568 M | 1024 | — (`UNVERIFIED` per-task C-MTEB) | — | `[REF]` size; multilingual dense+sparse |
| `Qwen/Qwen3-Embedding-0.6B` | 0.6 B | 1024 (28 layers) | 66.33 | 71.03 | `[REF]` Qwen3-Embedding card |
| `Qwen/Qwen3-Embedding-4B` | 4 B | 2560 (36 layers) | 72.27 | 77.03 | `[REF]` |
| `Qwen/Qwen3-Embedding-8B` | 8 B | 4096 (36 layers) | 73.84 | 78.21 | `[REF]` |
| `gte-Qwen2-1.5B-instruct` | 1.5 B | — | 67.12 | 71.86 | `[REF]` Qwen3-Embedding card comparison table |
| `gte-Qwen2-7B-instruct` | 7.6 B | — | 71.62 | 75.70 | `[REF]` |

**Live leaderboard position is a moving target.** A snapshot taken during this design showed
`MTEB(cmn, v1)` with 60 models / 31 tasks, and `ritrieve_zh_v1` (0.3 B, C-MTEB 72.71 / retrieval
76.97) and `KaLM-Embedding-Gemma3-12B-2511` near the top of the English/multilingual boards.
Their HF ids and current C-MTEB ranks are **`UNVERIFIED`** — take a dated snapshot of
`mteb-leaderboard.hf.space` before writing any "SOTA comparison" sentence.

**Implication:** `bge-small-zh-v1.5` (61.77) sits **~15 nDCG points** below `Qwen3-Embedding-8B`
(78.21). Any paper claiming DE closes that gap from a 24 M frozen backbone is not credible. DE must
be framed as *a cheap adaptation mechanism for a fixed small backbone*, never as a general SOTA
embedding model.

### 3.2 The closest competitor: static vocabulary / embedding expansion

**`arXiv:2509.26124` — "Vocabulary Customization for Efficient Domain-Specific LLM Deployment"**
(Herold, Kozielski, Santavas, Versley, Khadivi — eBay; NEURIPS 2025 CCFM Workshop; submitted
2025-09-30). `[REF]` — title, authors, venue and abstract verified on arXiv.

**Do not confuse it with a second paper of the same name:** Nakash et al., *"AdaptiVocab: Enhancing
LLM Efficiency in Focused Domains through Lightweight Vocabulary Adaptation"*, **`arXiv:2503.19693`**
`[REF]`. Herold et al. cite it and note that "at the time of writing, they have not yet released their
code" `[REF]`. Both are LLM (autoregressive) tokenizer-extension methods; neither is an embedding
method.

**Precisely how it works** `[REF]`: (1) train an in-domain BPE tokenizer; (2) append its
non-overlapping tokens **and their merge rules** to the base tokenizer, chosen by an algorithm that
*guarantees* token count never increases for any input; (3) initialise the new embedding/projection
rows as the mean of the constituent tokens (following Yao et al. 2021); (4) **continue training the
whole model** — 10 000 iterations on 480 H100s, < 24 h — so the model learns to use the new rows;
(5) evaluate.

**Measured outcome** `[REF]`: on e-commerce, up to **20 % shorter sequences**, ~**8 % mean speedup**
(up to 20 %) at 30 k added tokens, +1 % forward-pass time for the larger embedding matrix, and —
critically — **quality unchanged-to-slightly-worse**: NLU 60.5 → 60.1, E-commerce NLU 42.4 → 42.9,
E-commerce Lead gen 60.5 → 60.1, Lead 47.9 → 47.6. And the model *prefers* the new tokenisation only
~98 % of the time on sequences > 15 words.

| Dimension | Herold et al. 2509.26124 / AdaptiVocab 2503.19693 | **Domain Engram** |
|---|---|---|
| What changes | **tokenizer + input embedding table + LM head projection** | **nothing** — tokenizer, embeddings and backbone weights are byte-identical |
| Objective | **inference efficiency** (token fertility, RPS) | **retrieval quality** + update latency |
| Requires retraining | **Yes** — 10 k steps, 480× H100, < 24 h `[REF]` | **No** for a new term (closed-form ridge solve on one row) |
| Per-token FLOP growth | **+1 %** for 30 k tokens, growing with vocab `[REF]` | ≈ +3 % constant, independent of table size |
| Adding a term later | requires extending merges, re-tokenising the corpus, **re-indexing the collection**, and re-training the new rows | allocate slot + write one row + (amortised) matcher rebuild — **no re-indexing of the vector store** |
| Effect on quality in their own eval | **≈ 0 (marginally negative)** `[REF]` | unknown — this is our experiment |
| Bi-encoder / retrieval | not evaluated | this is our target |

**The comparison that isolates our contribution** (this is the single most important external
experiment):

> **V-EXP (vocabulary-expansion arm).** Take `bge-small-zh-v1.5`, run Herold-style vocabulary
> extension on our domain corpus: train a domain BPE on the e-commerce corpus, add the top-K
> non-overlapping tokens (K swept over {1 k, 4 k, 16 k}) with merge rules, mean-initialise the new
> embedding rows, resize the embedding matrix, and **fine-tune with the same InfoNCE data/steps** as
> `C`. Evaluate on Tier-D. Report nDCG@10, trainable params, wall-clock, and index size.
>
> **What it isolates:** whether *any* form of "give the model a dedicated representation for a domain
> term" beats the *same* information delivered through a side-channel. Three outcomes:
> - V-EXP ≥ C → the gain is about **having** a term representation, not about **how** it is delivered.
>   Our story weakens to "we are a cheaper-to-update way of getting vocabulary expansion", which is
>   still a defensible paper but a much smaller one.
> - C > V-EXP by ≥ 1.5 nDCG@10 → our residual injection + gating genuinely beats input-layer
>   expansion for retrieval. This is the strong version of the claim.
> - Both ≈ A0 → the whole domain-memory premise is wrong for a 24 M encoder, and we should say so.

Note the parameter asymmetry we must control for: V-EXP adds `K × 512` embedding rows
(1 k → 0.5 M, 16 k → 8.4 M), which is **not** free at training time (the embedding+logits matmul
grows), whereas DE's hashed tables are pure gathers. Report both param counts and both wall-clocks.

### 3.3 Continued contrastive pretraining (CPT) — the key "is it just domain data?" control

**Definition.** Same backbone, same domain corpus, **all** (or a matched subset of) backbone weights
trainable, standard contrastive objective. This is the baseline that says "the model just needed to
see e-commerce text".

- **Cost:** full fine-tune of `bge-small-zh-v1.5` is 24 M trainable and 6P FLOPs/token. On one GB10
  this is `[EST]` hours for the same token budget as `C`; `C` itself is ~17 % cheaper (§1.5).
- **What it does about dynamic updates:** **nothing, gracefully.** Incorporating a new term requires
  re-running CPT over a corpus containing it. There is no per-term write path.
- **GB10:** yes, trivially — a 24 M model fits in unified memory many times over.
- **Two variants, both mandatory:**
  - **CPT-full**: all backbone params trainable. Upper bound on "domain data helps".
  - **CPT-matched**: trainable params matched to `DE-Ref-S` (6.19 M) by training only the top-K
    layers + FFN, or by a rank-512 LoRA — the fair version.
- **Interpretation:** if `CPT-matched ≥ C` (within 0.5 nDCG@10), the memory buys nothing that a
  param-matched adapter on the same data would not buy → **falsifier F3/F6, stop.**

### 3.4 LoRA / adapter at matched parameter count

- **Cost:** rank grid `r ∈ {4, 8, 16, 32, 64, 128, 256}` on module subsets `{Q,V}`, `{Q,K,V,O}`,
  `{Q,K,V,O,FFN}`; step FLOPs ≈ same as full FT (§1.5) because activation gradients are still needed.
- **Matching rule:** for `DE-Ref-S` = 6.19 M on a 4-layer/512-dim encoder, LoRA param count is
  `Σ 2·r·(d_in + d_out)`. Over `{Q,K,V,O}` all layers this is `16 384·r`, so 6.19 M ⇒ **r ≈ 378**;
  over `{Q,K,V,O,FFN}` it is `36 864·r` ⇒ **r ≈ 168**. Both are large enough that "LoRA" is close to
  "dense adapter" — an honest observation we must state. Report the **matched-rank point** *and* the
  best point on the full rank curve, plus an interpolation of DE's budget onto the curve.
- **Dynamic updates:** LoRA can be incrementally updated per term (add a per-term delta) but every
  write changes the model globally and requires re-running inference over the corpus. It has no
  O(1) write and no delete. This is DE's genuine differentiator — but see §6 `G4`, which is where we
  will find out whether it matters in practice.
- **GB10:** yes; `peft` supports LoRA on BERT.

### 3.5 Add the terms to the tokenizer and retrain — "the obvious way"

Covered by V-EXP (§3.2) but listed separately because PLAN.md §3.2's "add terms to Registry" is easy
to confuse with it. Key extra numbers to report for this arm: corpus re-index time after tokenisation
changes, and the k-vec/ONNX rebuild cost. On a 1 M-document e-commerce index this is hours; the
Registry write is milliseconds. That asymmetry *is* the product claim — and it must be measured, not
asserted.

### 3.6 Other method families that must appear in the related-work / baseline table

| Family | Representative | Why relevant | Verified |
|---|---|---|---|
| **N-gram embedding injection** | **OverEncoding** — "Over-Tokenized Transformer: Vocabulary is Generally Worth Expanding", ICML 2025 (PMLR v267, `huang25bb`) | Engram names it as the closest n-gram-embedding prior art and reports it "fails to yield meaningful improvements on sparse MoE backbones even under a non-isoparametric setting" `[REF]` Sec 7. Direct prior art for our Layer-B. | `[REF]` title/venue; arXiv id `UNVERIFIED` |
| **N-gram / superword embedding scaling** | **SCONE** — "Scaling Embedding Layers in Language Models", **`arXiv:2502.01637`** (Google) | n-gram embeddings offloaded + contextually encoded. Engram notes SCONE is inference-focused and *adds training FLOPs*, hence incompatible with iso-compute `[REF]`. | `[REF]` id + repo `github.com/llmsresearch/scone` |
| **Superword tokenisation** | **SuperBPE** (Liu et al., 2025) | merges multi-word expressions into single tokens; same motivation as vocabulary expansion | `[REF]` name only; arXiv id `UNVERIFIED` |
| **Tokenizer transfer / expansion** | Yamaguchi et al. 2024 (`arXiv:2406.11477`), Csaki et al. 2023 (`arXiv:2311.05741`), Nguyen et al. 2023 (`arXiv:2312.00738`), Dagan et al. ICML 2024, Moisin et al. *Artif. Intell.* 2023, Sachidananda et al. **EMNLP 2021 Findings** (`2021.sustainlp.1.16`), Gee et al. **EMNLP 2022 Industry** (`2022.emnlp-industry.41`), Tai et al. **EMNLP 2020 Findings** (`2020.findings-emnlp.129`), Liu et al. **EMNLP 2024** (`2024.emnlp-main.424`, "Gold panning in vocabulary") | the entire vocabulary-adaptation literature, incl. for **encoder-only** models (which is our setting) | `[REF]` ids |
| **Efficient tokenizer adaptation** | "Teaching Old Tokenizers New Words: Efficient Tokenizer Adaptation for Pre-trained Models", **`arXiv:2512.03989`** — continued-BPE vocabulary extension, reports that naive "add tokens from an independently trained domain tokenizer" yields unreachable tokens | strongest *recent* methodological rival; should be implemented as the V-EXP variant instead of the naive one | `[REF]` id + claim |
| **Model-Aware Tokenizer Transfer** | OpenReview `IyV1QEc95F` | transfers attention inter-token communication patterns into the new tokenizer | `[REF]` forum id |
| **External parametric memory / RAG** | RETRO (`arXiv:2112.08726` `UNVERIFIED`), REALM, PlugLM, Memory+ (`ICML 2025`, openreview `ATqGm1WyDj`) | Engram's own framing of "decoupled, *editable* knowledge store" `[REF]` Sec 7 — this is the intellectual parent of our CRUD registry and must be cited | `[REF]` names; REALM/PlugLM ids `UNVERIFIED` |
| **Knowledge editing (no retraining)** | ROME (`arXiv:2202.05262` `UNVERIFIED`), MEMIT, AlphaEdit, WISE (NeurIPS 2024) | **the real dynamic-update competitor.** Editing is the field's answer to "update without retraining". It is per-fact, not per-term, but it is the baseline a reviewer will cite. | `[REF]` names; arXiv ids `UNVERIFIED` |
| **Lifelong editing limits** | "Understanding the Limits of Lifelong Knowledge Editing in LLMs", **`arXiv:2503.05683`**: MEMIT **collapses after ~250 sequential edits**; LoRA-Merge > LoRA-FT but both far below RAG | **this is the strongest ammunition for DE.** Post-hoc parametric editing does *not* scale to thousands of updates. Our D-group writes thousands of rows. | `[REF]` id + collapse figure |
| **Domain retrieval benchmark (zh medical)** | CMedTEB / CARE, **`arXiv:2604.10937`** (ACL 2026), repo `github.com/PhilipGAQ/CARE` | a curated Chinese-medical embedding benchmark; a good *second* domain and an independent judge | `[REF]` id + repo |
| **Domain-adapted e-commerce LLM** | Herold et al., "Domain adaptation of foundation LLMs for e-commerce", **ACL 2025 Industry** (`2025.acl-industry.74`) | the CPT recipe that 2509.26124 starts from | `[REF]` |

### 3.7 Baseline feasibility matrix (one GB10, GB10/DGX Spark class)

Hardware `[REF]`: GB10 Grace Blackwell, **sm_121**, 128 GB coherent unified LPDDR5x (64 GB variant
exists), **273 GB/s** memory bandwidth, 20-core Arm (10× Cortex-X925 + 10× A725), up to
**1 PFLOP FP4 sparse ≈ 500 TFLOPS dense FP4**, 240 W, 4 TB NVMe.

| Baseline | Fits in 128 GB? | Train time on one GB10 | Per-term dynamic update? | Re-index needed? |
|---|---|---|---|---|
| A0/A1/B1–B3/C/D/E/F/G | trivially (< 1 GB) | minutes | — | — |
| V-EXP (vocab extension) | trivially | `[EST]` 1–3 h (embed resize + same steps) | **No** (needs merge table + retrain) | **Yes, full re-index** |
| CPT-full (24 M) | trivially | `[EST]` 2–4 h | No | Yes |
| CPT-matched | trivially | `[EST]` 2–4 h | No | Yes |
| LoRA grid (7 ranks × 3 subsets) | trivially | `[EST]` 3–8 h total | No (global delta) | Yes |
| OverEncoding-style input-embedding injection | trivially | `[EST]` same as C | No | Yes |
| Knowledge editing (ROME/MEMIT) on a bi-encoder | trivially, but **no off-the-shelf bi-encoder editing impl** (`UNVERIFIED`) | n/a | Yes, per-fact | No |
| `Qwen3-Embedding-8B` (context only) | yes (16 GB bf16) | inference only, `[EST]` ~5× slower than small | No | — |

**Platform prerequisites before any of this runs** (all `[REF]` community reports, all a real risk):
GB10 is `sm_121` and **aarch64**. Stock x86 PyTorch wheels fail with "unsupported compute
capability" or "sm_121"; you need a CUDA ≥ 12.8 build (or a nightly / custom sm_121 wheel), and some
ecosystem packages (e.g. `bitsandbytes`) have no aarch64 wheels. **Run `torch.cuda.is_available()`
and a 100-step matmul benchmark as the very first item of Phase 0** — a silent CPU fallback would
invalidate every timing in this document.

---

## 4. Ablation grid

### 4.1 Design

**One-at-a-time (OAT) removal from `DE-Ref-S`.** For each variant: disable exactly one component,
keep everything else (data, steps, LR schedule, seeds, eval) identical. Report Δ vs `C` on Tier-D and
Tier-G.

Two reporting modes per ablation, because removal usually frees parameters:

* **`free`** — the variant simply has fewer parameters (what the module would cost in practice).
* **`padded`** — the freed parameters are re-spent (e.g. on table size) so the budget is held
  constant, matching Engram's "we keep the Engram parameter budget fixed" protocol `[REF]` Sec 6.2.
  **The `padded` column is the one that supports a causal claim about the component.** The `free`
  column is what an engineer would actually deploy. We report both; we draw conclusions from
  `padded`.

### 4.2 OAT table

Engram's Sec 6.2 importance order is **(i) multi-branch fusion, (ii) context-aware gating,
(iii) tokenizer compression**, then **(iv) depthwise conv (small)**, with **4-grams slightly hurting
at fixed budget** `[REF]`. Ours, mapped to a plain BERT encoder:

| ID | Disabled | Expected Δ (Tier-D nDCG@10) | Paper basis / our expectation |
|---|---|---|---|
| **OAT-1** | **Multi-branch fusion** → synthetic `M=2` split of the residual stream at the injection layer with separate `W_K^(m)`, shared table + `W_V` | **≈ 0 (null)** | **We predict this does NOT transfer.** Engram's branch fusion relies on mHC (M=4) hyper-connections `[REF]`; a BERT encoder has a single residual stream, so there is nothing to branch. Reporting a null here is a *feature*: it tests whether the paper's #1 importance item is architecture-specific. |
| **OAT-2** | **Context-aware gating** (`α ≡ 1`, drop `W_K` and the Q/K norms) | **largest negative** | Paper rank #2 `[REF]`. Also the mechanism the whole safety story rests on. Falsifier `F4` if null. |
| **OAT-3** | **Tokenizer compression** (`P = identity`, use raw token ids) | **large negative** | Paper rank #3 `[REF]`. On Chinese, NFKC + lowercase collapse 全/半角 and case variants; without it the same term occupies several hash slots and the table is spent. **This is where we expect the biggest transferable hit.** |
| **OAT-4** | **Multi-head hashing** (`K = 1` per order) | **moderate negative** | Paper does not rank this separately; collisions rise. |
| **OAT-5** | **Depthwise causal conv** | **small negative (0 to −0.5)** | Paper: "only marginally degrades" `[REF]`. |
| **OAT-6** | **Orders {2,3} → {2} only** | **moderate negative** | Removes the most frequent high-signal n-grams. |
| **OAT-7** | **Orders {2,3} → {2,3,4}** at **fixed budget** | **≈ 0 to small negative** | Paper: 4-grams are "slightly suboptimal under a fixed 1.6B budget — likely because it dilutes capacity from the more frequent 2/3-gram patterns" `[REF]`. We expect the same dilution here and it is a directly falsifiable prediction. |
| **OAT-8** | **Layer-A registry entirely** (hash memory only) | **Tier-CS: large negative; Tier-D: small** | This is the CRUD contribution. If Tier-D barely moves, DE is just "n-gram embeddings" and the whole paper is thin. |
| **OAT-9** | **Layer-B hashed memory** (registry only) | **moderate negative on Tier-D** | Isolates pattern memory from term memory. |
| **OAT-10** | **Residual injection** (compute `y`, discard it) | **must equal A0 to within fp noise** | Harness-integrity assertion, not an ablation. If not, stop. |
| **OAT-11** | **Gate bias** (`gate_bias_init` → none, pure content-driven α) | **small negative** | Controls whether the gate is doing semantic matching or just acting as a learned global scale. |
| **OAT-12** | **Pooling**: CLS (ref) vs **mean** vs **last** | **mean ≥ CLS expected** | PLAN §2.1 predicts mean-pooling benefits most because it sees more injection points. Directly testable, cheap. |
| **OAT-13** | **Gated sqrt variance reduction** (`gate_sqrt`) | ≈ 0 | The official demo applies a signed-√ to the logit; the paper Eq. 4 does not. |
| **OAT-14** | **Memory LR multiplier** (5× → 1×) | **large negative if 5× matters** | Paper Sec 4.1 `[REF]`. If null, our optimiser recipe differs from theirs and must be reported as such. |

### 4.3 Capacity / budget sweep (Engram Sec 3.2 analogue)

`DE-Ref-XS → S → L → XL`, sweeping total memory slots at a fixed `d_head`, then a fixed-slot
`d_head` sweep. Report nDCG@10 vs `log(slots)`.

**Pre-registered expectation:** Engram reports a "strict power law (linear in log-space)" for
validation loss vs number of slots, up to 1e7 slots / ≈13 B params `[REF]` Sec 3.2. We cannot test
that at 6 M params. **If our curve is flat above 4 M slots, we must explicitly state that the
"scalable memory axis" claim is unverified at our scale and we have no evidence it survives** — this
is gate `G6`.

### 4.4 Injection-layer placement sweep

Engram Sec 6.2 `[REF]`, 12-layer 3B MoE, fixed 1.6 B memory: single-module sweep over layers 1–12,
**layer 2 best (1.770)** vs MoE baseline 1.808; **two half-size modules at layers 2 and 6 → 1.768**,
slightly better than the best single point. Stated mechanism: early injection offloads local
pattern reconstruction, but early `h_t` is poorly contextualised for gating — a **trade-off between
offload depth and gating quality**.

**Our sweep — Tier S (`bge-small-zh-v1.5`, 4 layers, 0-indexed), budget held constant:**

| Sweep | Points | Prediction |
|---|---|---|
| **S-sweep-1** (single module) | `{0}, {1}, {2}, {3}` | best ≈ `{1}`; degrades monotonically toward `{3}`; `{0}` worse than `{1}` (no attention round has run, so the gate query is uninformative — Engram says "one round of attention is already sufficient" `[REF]`) |
| **S-sweep-2** (two half modules) | `{0,2}, {1,2}, {1,3}, {0,3}`, each with **half the per-site budget** | `{1,2}` best, matching `{1,2}` ≳ best single point |
| **S-sweep-3** (control) | `{}` (no injection) | must equal A0 |

**Our sweep — Tier B (`bge-base-zh-v1.5`, 12 layers) — required before any layer claim is published:**

| Sweep | Points | Cost |
|---|---|---|
| **B-sweep-1** | `{0} … {11}` single module at reduced budget | 12 runs |
| **B-sweep-2** | best-2 split from B-sweep-1, e.g. `{2,6}` vs `{1,4}` vs `{2,8}`, half budget each | 3–4 runs |

Engram's layer-2 result was measured on a **12-layer** backbone; `bge-small-zh-v1.5` has **4**.
Extrapolating "layer 2" to a 4-layer model is an *assumption*, so the sweep is mandatory and its
outcome (including "the best layer is not 1") is reported whatever it is.

### 4.5 Gating visualisation (replicates Engram Sec 6.5 `[REF]`)

Plot mean `α_t` at positions whose suffix n-gram contains a domain term vs. random positions, for
(a) trained terms, (b) post-hoc written terms, (c) non-matching positions. Paper finds strong α on
multi-token entities and formulaic phrases, and reports Chinese examples (四大发明, 张仲景) `[REF]`.
This is **qualitative corroboration only** and is never used as evidence for C1/C2/C3.

### 4.6 Memory-suppression sensitivity (replicates Engram Sec 6.3 `[REF]`)

Evaluate `C` with `memory_enabled=False` at inference (train/inference mismatch, exactly as the
paper does). Prediction: Tier-D collapses hard (the analogue of factual-knowledge retention dropping
to 29–44 %), Tier-G degrades less (the analogue of reading-comprehension retaining 81–93 %) `[REF]`.
This is the **sharpest available test of whether retrieval-with-domain-terms is on the
"fact-lookup" side of Engram's dichotomy** — and therefore of whether migrating Engram to embeddings
was a sensible idea at all. A null here (memory contributes equally to general retrieval) would be a
substantive negative finding about the migration.

---

## 5. Fair-comparison protocol

### 5.1 Iso-parameter

Report for **every arm**: trainable params, total params (incl. memory tables), optimiser-state
bytes, trainable-bytes, checkpoint bytes, and index bytes.

Engram's Sparsity Allocation framing `[REF]` Sec 3.1 fixes `P_tot` and `P_act` and varies how the
*inactive* budget is split. Encoders have no MoE, so the analogue is:

```
P_de    = backbone_active_params + memory_slots × d_head   (all counted, nothing is free)
P_act   = backbone_active_params + 2 · d_mem · d_model      (per-token activated, independent of table size)
```

**Rule.** Every headline comparison is made at **matched trainable params (≤ ±2 %)**; if a baseline
cannot hit the budget, report the bracketing points and interpolate. `A0` (0 trainable) is *not*
iso-parameter with anything and is reported separately as the zero-shot reference.

### 5.2 Iso-compute

As derived in §1.5: `C_DE/C_base = 5P/6P = 0.833` at layer-1 injection on a 4-layer model. Matched by
`B' = 1.20·B`. Run **both** `matched-batch` (equal negatives) and `matched-FLOPs` (equal compute).
Report analytic FLOPs *and* measured wall-clock, and flag any arm where they disagree by > 30 %
(indicates we are kernel-bound, not FLOP-bound — very likely for a 24 M model).

### 5.3 Seeds, variance, significance

- **Seeds:** `{11, 22, 33, 44, 55}` for A0–F; `{11, 22, 33}` for every OAT ablation; `{11}` for
  the layer sweeps (sweeps are re-run at 3 seeds only for the top-2 points). Budget: document the
  exact seed list; a seed controls init, data order, negative sampling, and (for baselines) any
  stochastic adapter init.
- **Reporting:** `mean ± std` over seeds **and** the 95 % CI from a **paired bootstrap over queries**
  (10 000 resamples, same resample indices for every arm).
- **Pairwise system comparison:** per-query `AP@k` → **Wilcoxon signed-rank** (two-sided,
  α = 0.05) plus **Cliff's δ** as effect size. For Recall@10 use **McNemar's exact test** on the
  flip table. Report the test *and* the effect size; never report "significant" without δ.
- **Multiple comparisons:** OAT has 14 arms. Apply **Holm–Bonferroni** across the OAT family and
  report adjusted p-values. Report the unadjusted p as well.
- **Pre-registration:** §6's thresholds are fixed *before* the first Tier-D run. Any post-hoc change
  must be logged in `docs/EXPERIMENTS.md` with a date and a reason, and reported as a change.

### 5.4 Held constant across every arm

| Held constant | Value |
|---|---|
| Training triplets (content **and order**) | identical file, identical seeded sampler |
| Tokens seen / optimiser steps | identical |
| Batch size | identical in `matched-batch` arms (§1.5) |
| LR schedule *shape* | identical (linear warmup 10 % → cosine) |
| Temperature, symmetric InfoNCE, hard-negative policy | identical |
| `max_len` | 128 train / 512 eval |
| Pooling, L2 normalisation | identical |
| Eval suite, candidate sampling (100 k), tie handling | identical |
| Seed list | identical |
| Hash seed / table primes / registry free-list policy | identical |

**Deliberately *not* held constant:** the learning-rate *value* and LoRA rank / table size. Each
method gets the same **tuning budget** — a grid of ≤ 12 configurations, 1 seed, on a held-out
`dev` split carved out of the *training* triplets (never from the eval sets) — and the tuned config
is then re-run with the full seed list. The entire tuning log is published.

### 5.5 Leakage and contamination controls

1. **Term-level holdout.** Cold-start terms must have **zero** exact canonical-token-n-gram overlap
   with the training corpus (checked with a suffix-array / Aho–Corasick scan over the compressed ids,
   orders 2–4). Near-duplicate documents are removed by MinHash-Jaccard ≥ 0.8.
2. **Document-level holdout.** The documents containing cold-start terms are excluded from training.
3. **Query leak check.** No cold-start term may appear in any query used for the Tier-D/Tier-G
   numbers — those are separate reports.
4. **Registry leak check.** The registry is asserted empty at the start of every run except where the
   arm explicitly populates it.
5. **Pre-training-data check.** We do **not** claim a term is "unseen by the model"; we claim it is
   unseen *by our training corpus*. The paper must say the weaker thing.

---

## 6. Success criteria / decision gates (pre-registered)

Primary metric: **Tier-D nDCG@10** (mean over the 5 domain C-MTEB tasks).
Guards: **Tier-G nDCG@10**, **Tier-S STS mean Spearman**, **Tier-CS nDCG@10**.
All deltas are **absolute points** (not relative %), because at a 61.77 baseline, 1 % relative
(0.62 pt) is below seed noise.

### 6.1 Continuation thresholds (C1/C2)

| Outcome (5 seeds, paired bootstrap 95 % CI excludes 0, Wilcoxon p < 0.05) | Decision |
|---|---|
| `C − A0 ≥ +3.0` on Tier-D **and** all 5 seeds positive | **GO.** Invest in the full matrix. |
| `+1.5 ≤ C − A0 < +3.0` | **CONDITIONAL.** Continue only if additionally `C ≥ A1` and `C ≥ best matched baseline`, and `D > C` on Tier-CS. |
| `C − A0 < +1.5` | **STOP.** No defensible "domain adaptation" claim. |

*Why +3.0 and not +1?* `[REF]` From the published C-MTEB table, going from `bge-small-zh` to
`bge-base-zh` (24 M → 102 M, 4× params) buys **+5.4** on EcomRetrieval, **+6.7** on
MedicalRetrieval, **+6.6** on CmedqaRetrieval. So a +3.0 absolute domain gain from a **frozen** 24 M
backbone with ~6 M extra trainable params is roughly "half of what quadrupling the model buys". That
is the bar we pre-commit to. +1.5 is reported as a null-to-weak result, not a win.

### 6.2 Specificity check (does the gain come from *domain* memory?)

`ratio = (C − A0)_domain / (C − A0)_general`.

* `ratio ≥ 2.0` → the gain is domain-specific: supports C1.
* `0.5 ≤ ratio < 2.0` → the module helps equally everywhere: it is a generic quality booster, not a
  domain memory. **Reframe** the paper as "a cheap generic contrastive adapter" and drop the domain
  claim. (This is still a publishable result; it is just a different one.)
* `ratio < 0.5` → the module is doing something *other* than domain adaptation. Investigate or stop.

### 6.3 Forgetting guard (hard stop)

| Guard | Threshold | Action |
|---|---|---|
| Tier-G nDCG@10 drop vs `A0` | **> −1.0 pt** | **STOP** the whole line; report the collapse. |
| Tier-S STS mean Spearman drop | **> −1.5** | **STOP.** |
| C-MTEB Overall Avg drop | **> −2.0** | **STOP.** |

*Note this is tighter than PLAN.md §5's "≤ 1 %"*, which at a 61.77 baseline is 0.62 pt — smaller than
typical seed-to-seed std on a 4-layer model. We tighten to absolute 1.0 pt and say so.

### 6.4 The dynamic-update gate (C3) — the headline

| Outcome on Tier-CS | Decision |
|---|---|
| `D − C ≥ +5.0`, p < 0.05, **and** write p50 < 50 ms/term, p99 < 500 ms, **and** zero parameter updates outside the written rows | **GO — headline claim supported.** |
| `+1.0 ≤ D − C < +5.0` | **WEAK.** Report honestly as "some benefit"; do not lead with it. |
| `D − C < +1.0` | **CRUD REGISTRY IS DEAD.** Strip Layer-A; publish C only. |
| `D ≤ C` | **Falsifier `F1`.** The entire thesis fails. Report the negative result. |

Additional dynamic sub-gates:

* **Oracle comparison:** if `F` (retrained with the terms) `≤ D + 1.0`, the write path is *strictly*
  better than retraining → strongest version of the claim. If `F ≥ D + 3.0`, state plainly that
  retraining would have been better.
* **Realistic-write gate:** the write must work when the operator supplies **only the term string**
  (no context corpus). If the oracle-context regime works but the term-only regime does not, the
  product claim ("运营插一条新术语") fails. Report both regimes, never just the flattering one.
* **Scaling gate:** the write path must hold for **≥ 2 000** successive terms without degradation.
  MEMIT collapses after ~250 sequential edits `[REF]` `arXiv:2503.05683`; that is the bar we must
  clear and we must run the same sequential-sweep test on ourselves.

### 6.5 Delete / poison safety (group G)

Recover to **≥ 98 %** of the pre-poison score after deleting the poisoned entries, with no other
metric moving by more than 0.2 pt. Below 96 % → the "the gate makes CRUD safe" claim is **falsified**
and must be withdrawn, regardless of C1/C3 results.

### 6.6 Falsifiers (pre-committed negatives)

| ID | Falsifier | Consequence |
|---|---|---|
| **F1** | `D ≤ C` on Tier-CS | thesis fails; publish negative result |
| **F2** | Cold-start gain requires the oracle context corpus (term-only write fails) | product claim fails; method becomes "write-a-term-with-contexts" |
| **F3** | `CPT-matched` or best `LoRA@6.19 M` ≥ `C` | memory is not the mechanism; stop |
| **F4** | OAT-2 (`α≡1`) within 0.3 pt of `C` | gating claim does not transfer to bi-encoders |
| **F5** | `G` recovers < 96 % of pre-poison score | CRUD safety claim is false |
| **F6** | `A1` (matched-capacity MLP head, no memory) ≈ `C` | gain is capacity, not memory |
| **F7** | nDCG vs memory slots is flat above 4 M slots | "scalable memory axis" unverified at our scale |
| **F8** | Memory-suppression sensitivity (§4.6) does not separate domain from general | Engram→embedding migration is unjustified; re-evaluate the whole premise |

### 6.7 The overall decision rule

> **We claim anything only if `C1 ∧ C3 ∧ C4(falsifier-clean)`, i.e. `C > A0` by the §6.1 threshold,
> `D > C` by the §6.4 threshold, and none of F1–F8 have fired.** Otherwise we publish the null
> result with the full matrix, including the ablations that *did* work.

---

## 7. Cold-start protocol (`Tier-CS`) — specification

This is the experiment the whole project is sold on, so it gets its own rigorous protocol.

1. **Term universe.** From the e-commerce domain corpus, take every canonical-token n-gram
   (orders 2–4) that occurs ≥ 20 times and that is *not* an exact match for any training-corpus
   n-gram. Rank by corpus TF-IDF; sample **2 000** terms with a stratified sample over frequency
   (head / mid / tail buckets of 700 / 700 / 600).
2. **Documents.** For each term, take up to 10 held-out documents containing it as the positives;
   the corpus of these + 100 000 distractors is the CS corpus. Queries are natural sentences from
   those documents that contain the term.
3. **Two write regimes, always both:**
   * **R1 (oracle context):** the operator supplies the term *and* up to 8 sentences containing it.
   * **R2 (term-only):** the operator supplies only the term string. `solve_slot_closed_form` then
     runs with a single, term-only context; if that is degenerate we fall back to mean-initialising
     the row from the frozen backbone's own representation of the term and say so.
4. **Three write methods, all reported:** (a) closed-form ridge solve, 0 gradient steps;
   (b) + 20 single-row refine steps; (c) + 200 single-row refine steps. Report all three — (b) and
   (c) are still "no retraining of the model", but they *are* gradient steps and must not be hidden.
5. **Controls on the same split:** `C` (no write), `C` + random-row write (same number of rows, random
   vectors — must give ≈ 0 or negative), `C` + write without the gate (`W_V` path forced, α≡1),
   `F` (oracle retrained with those documents).
6. **Metrics:** Tier-CS nDCG@10 / Recall@10; α at written-term positions; write latency
   (p50/p95/p99); matcher rebuild latency as a function of registry size (10 k / 50 k / 200 k terms);
   effect on Tier-G (localisation check — a write must not move general retrieval by > 0.5 pt).
7. **Sequential-sweep stress test** (`arXiv:2503.05683` protocol): apply writes one at a time for
   2 000 steps, measuring Tier-CS every 100 writes, to detect our own collapse point.

---

## 8. Run ordering for a single GB10

`[EST]` timings assume the Phase-0 calibration and are **ordered by value-per-hour, not by
cheapness**. The two cheapest experiments are the two most decisive ones.

| # | Run | What it decides | `[EST]` wall-clock | Value/ Hour |
|---|---|---|---|---|
| **P0.1** | **Platform probe**: `torch.cuda.is_available()`, sm_121 support, a 100-step bf16 matmul benchmark, peak memory, aarch64 wheel audit for `peft`/`transformers`/`mteb` | Whether *anything* is possible | **0.5 h** | gate |
| **P0.2** | **Eval-harness repair** (see §9) + **reproduce `A0` = 61.77 ± 0.5 on C-MTEB Retrieval** | Whether our measurements mean anything | **3–4 h** | gate |
| **P0.3** | **Null-inertness tests** (OAT-10, B3 = A0 to fp noise; registry-empty path bit-identical) | Whether the module leaks into A | **0.5 h** | gate |
| **P1** | **KILL-SHOT #1 — can it overfit?** Train `C` on **200** triplets for 300 steps. Pass = train loss collapses and Tier-0 nDCG moves. | Is the implementation even wired up? Fails ~30 % of the time. | **10 min** | ★★★★★ |
| **P2** | **KILL-SHOT #2 — A0 vs C on Tier-0 + 500-query Tier-D, 2 seeds.** | **Can the entire idea die here, in 30 min?** | **30 min** | ★★★★★ |
| **P3** | `A1` (matched-capacity MLP head) vs `C`, 2 seeds | Is it memory or capacity? (`F6`) | **40 min** | ★★★★★ |
| **P4** | `B1`, `B2`, `B3`, 3 seeds | Control-group completeness; `F4` | **1.5 h** | ★★★★ |
| **P5** | Full **A0/A1/B1/B2/B3/C, 5 seeds**, Tier-D + Tier-G + Tier-S | The main table | **6–8 h** | ★★★★★ |
| **P6** | **`Tier-CS` cold-start**, all regimes and write methods | **The headline claim + `F1`/`F2`** | **2–3 h** | ★★★★★ |
| **P7** | **`LoRA` rank grid** (7 ranks × 3 subsets) and **`CPT-matched`** on the same data | **`F3` — the fairest competitor** | **4–6 h** | ★★★★ |
| **P8** | **V-EXP** vocabulary-extension arm (K ∈ {1 k, 4 k, 16 k}) | The closest competitor (§3.2) | **3–4 h** | ★★★★ |
| **P9** | **Injection-layer sweeps** S-sweep-1/2 (4 + 4 runs, 1 seed) | Layer placement on 4 layers | **2 h** | ★★★ |
| **P10** | **OAT grid** (14 variants × 3 seeds) | Component attribution; `F4`, paper-order test | **8–12 h** | ★★★ |
| **P11** | **Capacity sweep** XS→S→L→XL | `F7` | **3 h** | ★★★ |
| **P12** | Memory-suppression sensitivity (§4.6) + gating visualisation (§4.5) | `F8`; qualitative figure | **1 h** | ★★★ |
| **P13** | Poison/delete test (group G) + 2 000-term sequential sweep | `F5` | **1.5 h** | ★★★ |
| **P14** | Tier-X full C-MTEB on the 3 surviving arms only | Final table | **12 h** | ★★ |
| **P15** | **Tier-B scale-up** on `bge-base-zh-v1.5` (12 layers): B-sweep-1 (12 runs), B-sweep-2, then C/D | The only place the "layer 2" claim is meaningful | **12–20 h** | ★★★ |
| **P16** | Learned post-hoc write (a small MLP that predicts the slot row from the term string, trained on written terms) | Turns R2 from a heuristic into a method | **4 h** | ★★ |
| **P17** | `CPT-full`, `V-EXP` at matched FLOPs, OverEncoding-style input-embedding injection | Completes the baseline table | **8 h** | ★★ |

**Total to a defensible paper: ≈ 70 h.** The path from "cannot work" to "does it work at all"
(P0 → P2) is **≈ 5 hours**. Everything past P5 is optional until P2 passes.

**Ordering rationale.** P1 and P2 come first because they are the only runs that can *stop* the
project, and stopping early is worth more than any number of confirmatory runs. P3 (`A1`) is placed
immediately after because it is the single cheapest control that separates "memory" from "capacity",
and a project that cannot separate those two is dead on arrival. P7 (`LoRA`/`CPT`) is deliberately
late: it is expensive, and if P2/P3 fail there is no point. P15 (12-layer) is last because a 4-layer
result cannot be extrapolated, but also cannot be published as-is.

---

## 9. Known measurement hazards in the current repository (must be fixed in Phase 0)

Found by reading `de/evaluation.py`, `de/model.py`, `de/train.py`, `de/registry.py`,
`de/cold_start.py`. **These are not style nits — several of them would silently manufacture the
result we want.** Each is a P0.2 blocker.

| # | Location | Problem | Consequence | Fix |
|---|---|---|---|---|
| **H1** | `evaluation.py::retrieval_metrics_from_scores` | `idcg` is computed as `1/log2(arange(n_queries)+2)` — the **query count**, not `k=10`. The function is named `ndcg@10` but is not nDCG@10. | All headline numbers are wrong by a large, query-count-dependent factor. | Implement standard nDCG@k with `idcg` over `min(k, n_cand)`. |
| **H2** | `evaluation.py::evaluate_retrieval` | Builds the full `Q @ C.T` matrix and then samples `nq` random distractors per query with `torch.randperm(C.shape[0])` — **not** C-MTEB's 100 k candidate protocol, and it never removes the *other* gold documents (false negatives). | Results are not comparable to any published number; Recall@10 is understated. | Use `mteb` + `mteb.Dev` style corpora and C-MTEB's documented 100 k-candidate sampling. |
| **H3** | `evaluation.py::evaluate_retrieval` | `model.memory_enabled = suppress_memory and bool(model.layer_ids)` — the parameter is named `suppress_memory` but is assigned to `memory_enabled`. Passing `suppress_memory=True` **enables** memory. | The Engram §6.3 sensitivity experiment can be run backwards by accident. | Rename to `memory_enabled`, invert nothing, add a unit test. |
| **H4** | `evaluation.py` tie handling | Ranks use strict `>`; a model that emits duplicate embeddings (common when memory is disabled and pooling degenerates) gets optimistic ranks. | Inflates A0 and inflates tie-prone baselines. | Tie-aware rank: `ranks = (sim > pos).sum() + ((sim == pos).sum() - 1)/2`, floored. |
| **H5** | `model.py::_canonical_ids`, `hashing.py` | `nn.Embedding` tables use **dense** gradients. At the XL budget the tables hold 33.55 M elements; dense grad + param + Adam state at 12–16 B/element is **0.40–0.54 GB**, touched every step. | Will not scale past the L budget; also far slower than sparse updates on long-tail tables. | `sparse=True` (or a fused dense-Adam) for the hashed tables; keep the registry table dense (it is small and gets sparse updates anyway). |
| **H6** | `train.py::build_param_groups` | `value_proj` / `key_projs` have `bias=False`, so they fail the `".conv." / "norm_" / "gate_bias" / ".bias"` filters and land in the **memory** group → they get **Adam with lr × 5 and wd = 0**, which is not what Engram Sec 4.1 prescribes (Adam ×5 is for the *tables*). | Silent deviation from the paper's optimiser recipe; confounds OAT-14. | Classify by explicit parameter name/owner, not string matching. |
| **H7** | `cold_start.py::solve_slot_closed_form` | `target = h_plain[0, end]` is computed with the backbone run **with memory suppressed**, i.e. `target == h_pre` up to layers above the injection layer. Then `delta = target - h_pre`, which is **non-zero only because of layers ≥ layer_id**. The docstring's intent ("what the backbone *should* think at this position") is not what the code computes. | The closed-form write solves for a correction that partly cancels the model's own computation above the injection layer. Either useless or actively harmful. | Fix, or replace with a closed-form that targets an *external* target (e.g. the mean representation of the term over contexts, projected). Must be re-derived before P6. **This is the single most likely source of a spurious null on the headline claim.** |
| **H8** | `cold_start.py::refine_slot` | `opt = Adam([w])` over the **entire** registry embedding matrix, then `w.requires_grad_(True)` — it optimises all rows' gradients (which are dense) and applies weight-decay-free updates to every row, not just the new one. | Not a "single-row fine-tune" as claimed; risks erasing all previously written terms. | Optimise a 1-row parameter view, or use `sparse=True`. |
| **H9** | `registry.py::PhraseMatcher._rebuild` | Any registry mutation bumps `version`; the next `match()` pays an **O(\|active phrases\|)** rebuild (hashing + sort for every length class). With 50 k phrases this is not O(1). | The "millisecond write" claim is false at scale; must be measured and reported (§6.4 latency gates include the rebuild). | Measure and report the true write→first-query latency; optionally ship an incremental index update. |
| **H10** | `model.py::forward` | `use_memory` depends on `self.memory_enabled and bool(self.layer_ids)`; `OAT-10` ("compute `y`, discard it") has no switch. | The identity assertion (must equal A0 to fp noise) cannot be run. | Add an explicit `inject: bool` switch. |
| **H11** | `model.py::prepare_registry_slots` / `encode` | The matcher runs on CPU numpy **per batch**; `match()` does an inner Python loop over candidate positions. | Likely the dominant training cost, and it scales with sequence length, not model size. | Profile in P0; if it dominates, this is the real bottleneck to report, not FLOPs. |
| **H12** | `losses.py::info_nce_with_hard_negatives` | Hard negatives are **only** used in the query→doc direction; the doc→query direction is dropped. Asymmetric. | Any baseline using symmetric loss is not comparable. | Either make DE symmetric or make all arms asymmetric — pick one and hold it fixed (§5.4). |
| **H13** | `train.py::main` | No seed argument is threaded anywhere; `torch`/`numpy` are unseeded. | **All seed-variance requirements (§5.3) are currently unimplementable.** | Add `--seed` and seed `torch`, `numpy`, `random`, plus the DataLoader generator. |

**P0.2 acceptance criteria (all must pass):** H1–H4, H7, H10, H13 fixed and unit-tested;
`A0` reproduces 61.77 ± 0.5 on C-MTEB Retrieval; `OAT-10` ≡ `A0` to fp32 noise (max abs diff < 1e-5).

---

## 10. Run manifest (every experiment gets an id)

```
P0.1  platform-probe              P0.2  eval-harness-v2            P0.3  null-inertness
P1    overfit-200                 P2    A-vs-C-killshot             P3    A1-capacity-control
P4    B1/B2/B3                    P5    main-table-5seed            P6    cold-start
P7    lora-grid / cpt-matched     P8    vexp-vocab-extension        P9    layer-sweep-S
P10   oat-01..oat-14              P11   capacity-sweep              P12   sensitivity+gating-viz
P13   poison-delete / seq-2000    P14   tier-X-full                 P15   tier-B-scaleup
P16   learned-write               P17   cpt-full / overencoding
```

Each run emits: `config.json`, `resolved_config.json` (after defaults), `metrics.json`,
`metrics_per_query.parquet`, `timing.json`, `env.txt`, `git_rev.txt`, and the trained checkpoint.

---

## 11. Verified references

**Primary**
- Cheng et al., *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language
  Models* (Engram), **`arXiv:2601.07372v1`**, CC BY 4.0; ACL 2026 long paper
  (`aclanthology.org/2026.acl-long.226.pdf`); code `github.com/deepseek-ai/Engram` (Apache-2.0).
  All numbers quoted in §1.3, §4.2, §4.4, §4.6 were read from the v1 HTML, Sec 4.1 / 6.2 / 6.3 / 6.5.
- Herold, Kozielski, Santavas, Versley, Khadivi (eBay), *Vocabulary Customization for Efficient
  Domain-Specific LLM Deployment*, **`arXiv:2509.26124v1`**, NEURIPS 2025 CCFM Workshop.
- Nakash, Calderon, Ben-David, Hoffer, Reichart, *AdaptiVocab*, **`arXiv:2503.19693`**.
- Xiao et al., *C-Pack: Packed Resources For Advance General Chinese Embeddings*, **`arXiv:2309.07597`**;
  C-MTEB task/size table & leaderboard via `C-MTEB` PyPI 1.1.1 (released 2024-04-17).
- Muennighoff et al., *MTEB*, **`arXiv:2210.07316`**; Enevoldsen et al., *MMTEB*, **`arXiv:2502.13595`**.
- Xie et al., *Qwen3 Embedding*, **`arXiv:2506.05176`** / model card `Qwen/Qwen3-Embedding-0.6B`
  (source of the C-MTEB comparison table used in §3.1).
- Jiang et al., *Benchmarking and Enabling Efficient Chinese Medical Retrieval via Asymmetric
  Encoders* (CMedTEB / CARE), **`arXiv:2604.10937`**, ACL 2026; repo `github.com/PhilipGAQ/CARE`.
- *Understanding the Limits of Lifelong Knowledge Editing in LLMs*, **`arXiv:2503.05683`**.
- *Teaching Old Tokenizers New Words: Efficient Tokenizer Adaptation for Pre-trained Models*,
  **`arXiv:2512.03989`**.
- *Scaling Embedding Layers in Language Models* (SCONE), **`arXiv:2502.01637`**;
  repo `github.com/llmsresearch/scone`.
- Huang et al., *Over-Tokenized Transformer: Vocabulary is Generally Worth Expanding* (OverEncoding),
  ICML 2025, PMLR **v267**, paper key `huang25bb`.
- NVIDIA, *DGX Spark* (GB10) product page + DGX Spark User Guide: 128 GB unified LPDDR5x,
  273 GB/s, up to 1 PFLOP FP4 (sparse), 20-core Arm, sm_121.

**Model / dataset identifiers used above (all verified by web search, none invented)**
`BAAI/bge-small-zh-v1.5` · `BAAI/bge-base-zh-v1.5` · `BAAI/bge-large-zh-v1.5` ·
`BAAI/bge-m3` · `intfloat/multilingual-e5-small` · `Qwen/Qwen3-Embedding-{0.6B,4B,8B}` ·
`gte-Qwen2-1.5B-instruct` · `gte-Qwen2-7B-instruct` ·
C-MTEB `T2Retrieval` `MMarcoRetrieval` `DuRetrieval` `CovidRetrieval` `CmedqaRetrieval`
`EcomRetrieval` `MedicalRetrieval` `VideoRetrieval` `T2Reranking` `CMedQAv1` `CMedQAv2` `Ocnli`
`Cmnli` `CLSClusteringS2S/P2P` `ThuNewsClusteringS2S/P2P` `ATEC` `BQ` `LCQMC` `PAWSX` `STSB`
`AFQMC` `QBQTC` `TNews` `IFlyTek` `Waimai` `OnlineShopping` `MultilingualSentiment` `JDReview`
(Hub: `C-MTEB/<name>`).

**Explicitly UNVERIFIED (do not cite until checked)**
- `ritrieve_zh_v1` (0.3 B, C-MTEB 72.71 / retrieval 76.97) — appears in Qwen3's comparison table;
  HF id unknown.
- `KaLM-Embedding-Gemma3-12B-2511`, `Nemotron-3-Embed-8B-BF16`, `Octen-Embedding-{4B,0.6B}`,
  `jina-embeddings-v5-text-nano`, `harrier-oss-v1-*` — seen in a live MTEB leaderboard homepage
  snapshot; ids, sizes and C-MTEB ranks not confirmed.
- Current ranking on `mteb-leaderboard.hf.space` for `MTEB(cmn, v1)` — dynamic; snapshot with a date.
- arXiv ids for OverEncoding, SuperBPE, ROME, MEMIT, AlphaEdit, WISE, REALM, RETRO, PlugLM,
  Memory+ (names and venues are confirmed; numeric ids are not).
- Per-task C-MTEB numbers for any `*-v1.5` BGE checkpoint (only aggregates are published).
- Actual GB10 throughput, `peft`/`bitsandbytes` aarch64 wheel availability, and stable sm_121
  PyTorch packaging — must be measured in P0.1.

---

## 12. Open questions for the parent session

1. **Domain choice.** §1 assumes **e-commerce** as the primary domain (best public zh retrieval
   resources: `EcomRetrieval`, plus AdaptiVocab/Herold et al. as the closest comparator, plus
   `Herold et al. 2025 acl-industry.74` as the CPT reference). PLAN.md §3.3 suggests e-commerce **or**
   medical. Medical would let us use **CMedTEB** (`arXiv:2604.10937`) as an independent judge but
   makes the vocabulary-expansion comparison less direct. *Recommendation: e-commerce primary,
   CMedTEB as a second, out-of-scope-of-tuning domain for generalisation.*
2. **Training corpus.** PLAN.md §3.1 lists PubMed/FinQA/CaseHOLD/Amazon Reviews. For a **retrieval**
   experiment we need `(query, positive, hard-negative)` triples, and none of those are retrieval
   datasets. `EcomRetrieval` (1000 q) is too small to train on; the natural sources are
   T2Ranking (train split) and DuReader. **This data-engineering step is on the critical path and is
   not yet scoped anywhere.** It is the biggest under-planned item in PLAN.md.
3. **Tier-B timing.** §8 P15 (12-layer `bge-base-zh-v1.5`, 12+ run sweep) is the only place the
   "layer 2" claim is meaningful, at an estimated 12–20 h. Worth it only if P5 shows a real domain
   effect *and* the effect survives the P7 baselines.
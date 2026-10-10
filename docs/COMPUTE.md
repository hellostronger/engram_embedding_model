# Domain Engram — Compute & Budget Plan (NVIDIA GB10 / DGX Spark class)

> Target hardware: NVIDIA GB10 Grace Blackwell (DGX Spark class), 128 GB coherent unified memory.
> Purpose: parameter/memory accounting, training + inference throughput estimates, cold-start write cost,
> and a concrete recommended EngramConfig for the first real run.
>
> Method: the sandbox cannot download from HuggingFace and has no GPU, so **every throughput number
> below is analytic**. Hardware facts and a few third-party measured GB10 figures are **VERIFIED** via
> public sources (cited). FLOPs, parameter counts, memory sizes, and wall-clocks are **DERIVED** from
> the actual source code (`de/config.py`, `de/model.py`, `de/engram.py`, `de/hashing.py`,
> `de/registry.py`, `de/cold_start.py`, `de/train.py`) plus the verified hardware specs.
> No benchmark numbers are invented; anything not verified is labeled DERIVED.
>
> Date: 2026-10-10

---

## 0. TL;DR for the first GB10 run

| Item | Value |
|---|---|
| Backbone | BGE-small-zh-v1.5, **23.95 M params**, d_model=512, 4 layers (frozen) |
| DE memory module (config **M**) | **81.2 M trainable params** (67.2 M hashed tables + 12.8 M registry + 1.2 M fusion) |
| Full model | 105.2 M params; Adam fp32 footprint ≈ **1.30 GB** + ~2.4 GB activations at B=64,T=128 |
| Training speed (DERIVED) | ~0.37 M tokens/s (18 TFLOPS central) → **5 M triplets ≈ 1.9 h**, 50 M ≈ 19 h (T=128, B=64) |
| Inference (DERIVED) | ~0.63 M tokens/s encode; memory module costs only **+9 % FLOPs** (on/off) |
| Cold-start write | ~0.5–2 ms amortized ridge solve + ~2 full backbone forwards; 100 k terms ≈ **10–20 min batched** |
| Recommended first config | config **M** below: `table_size_per_ngram=65536, d_head=32, n_head=8, layers=[1,3], B=64, T=128, lr=2e-5×5` |

Everything comfortably fits in 128 GB with >95 % headroom. **The bottleneck is compute FLOPS, not memory** (§3).

---

## 1. VERIFIED GB10 hardware facts

### 1.1 Specs (all VERIFIED from NVIDIA + TechPowerUp)

| Property | Value | Source |
|---|---|---|
| Architecture | NVIDIA Grace Blackwell | NVIDIA datasheet |
| GPU chip | GB20B, Blackwell, **6144 CUDA cores, 192 tensor cores, 48 SMs** | TechPowerUp |
| CUDA compute capability | **sm_121** (compute capability 12.1) | llama.cpp on-device report: "NVIDIA GB10, compute capability 12.1" |
| Unified system memory | **128 GB LPDDR5x**, coherent CPU+GPU (NVLink-C2C); **64 GB variant also ships** | NVIDIA product page |
| Memory interface / bandwidth | 256-bit, **273 GB/s** | NVIDIA datasheet; TechPowerUp 273.2 GB/s |
| L2 cache | 24 MB | TechPowerUp |
| CPU | **20-core Arm**: 10× Cortex-X925 + 10× Cortex-A725 (aarch64 / ARMv9) | NVIDIA datasheet |
| TDP | 140 W (GB10), 240 W system PSU | NVIDIA datasheet |
| Storage | 4 TB NVMe M.2 (SED) | NVIDIA datasheet |
| GPU Tops | **FP4: 1 PFLOP** (theoretical, with sparsity) | NVIDIA datasheet footnote |

### 1.2 Precision throughput (dense; TechPowerUp matrix table, VERIFIED)

| Format | Dense TFLOPS | With 2:4 sparsity |
|---|---|---|
| FP4 (NVFP4/MXFP4) | **475.4** | 950.8 → "1 PFLOP" |
| FP6 | 237.7 | 475.4 |
| FP8 | **237.7** | 475.4 |
| INT8 | 237.7 (TOPS) | 475.4 |
| **FP16 / BF16 (tensor)** | **118.8** | 237.7 |
| TF32 | 59.4 | 118.8 |
| FP32 (non-tensor) | 29.7 | — |

*Note on marketing numbers:* the "1 PFLOP" figure is **FP4 with 2:4 structured sparsity**. Dense FP4 is
475 TFLOPS. For our workload (a tiny fp32/bf16 model) we care about the **BF16 tensor** and **FP32** rows.

### 1.3 Measured GB10 compute points (third-party, VERIFIED as *reported measurements*)

These anchor the analytic estimates in §3. None were produced on this box.

| Measurement | Value | Source |
|---|---|---|
| BF16 matmul 4096³ | 2.20 ms → **≈ 62.5 TFLOPS** achieved (53 % of 118.8 peak) | ugonfor/DGX-Spark-Benchmark (README table) |
| FP8 matmul 4096³ | 0.79 ms → **≈ 174.6 TFLOPS** (73 % of 237.7 peak) | same |
| LLaMA-3.1-8B prompt eval (Ollama) | ~575 tok/s prefill | G3nadh/dgx-spark-benchmarks |
| gpt-oss-120B MXFP4 prefill (llama.cpp) | pp2048 ≈ **2438 tok/s** | llama.cpp DGX Spark thread |

The ~62 TFLOPS BF16 matmul point is the most useful anchor: it says cuBLAS reaches roughly half of
tensor-core peak on this chip for a realistic GEMM. I use **8 / 18 / 35 TFLOPS** as conservative /
central / optimistic *end-to-end* effective rates for our full model (much lower than the raw matmul
62 TFLOPS because we have 4 skinny layers, small GEMMs, framework overhead, and an Adam step).

### 1.4 PyTorch builds that support sm_121 — RECOMMENDED SETUP

**Recommended: NGC PyTorch container, `nvcr.io/nvidia/pytorch:25.08-py3` or newer (e.g. `25.12`, `26.04`).**

- The NGC PyTorch catalog tags releases as **"DGX Spark Supported"** and ships **arm64** manifests
  (VERIFIED: NGC tag page for 26.07 lists `linux/arm64` + `linux/amd64`).
- The DGX Spark User Guide points to `nvcr.io/nvidia/pytorch:24.08-py3` as a baseline; PyTorch 2.7+
  containers include Blackwell kernels and cu128/cu130.

**Alternatives (pip wheels on aarch64):**
- `pip install torch --index-url https://download.pytorch.org/whl/cu130` → torch 2.9/2.10/2.11 `+cu130`
  have **linux_aarch64** wheels (VERIFIED: PyTorch aarch64 CUDA support shipped from 2.11 per PyTorch
  blog; Isaac-GR00T issue confirms `torch 2.9.1+cu130` aarch64 works on GB10). On install you may see a
  harmless warning that cu capability 12.1 is outside the wheel's declared max — it still runs.
- `cu128`/`cu129` wheels work for Blackwell but a **cu130** build is the safest match for GB10/driver 580.

**Gotchas (VERIFIED from issue trackers / forum reports):**
1. **flash-attention on aarch64 + sm_121 is effectively unavailable.** Upstream issue
   Dao-AILab/flash-attention#1969 is open ("no precompiled linux_aarch64+sm121 wheel; source build
   emits sm80/sm90 objects → 'no kernel image is available'"). Community prebuilds exist
   (`mjun0812/flash-attention-prebuild-wheels`, `HenryZ838978/flash-attention-blackwell` for sm_120) but:
   - a GB10 forum post reports FA2 = 489 ms, FA3 = 489 ms vs **SDPA = 481 ms** at 512 tokens — **SDPA is
     ~2 % faster than FA** on GB10, so there is **no benefit**;
   - AI2's open-instruct DGX Spark branch **marks flash-attn unavailable on aarch64 and falls back to SDPA**.
   → **Use `torch.nn.functional.scaled_dot_product_attention` (PyTorch SDPA).** Do not spend time on FA.
2. **vLLM/SGLang are not needed** and have open sm_121 stability bugs (silent EngineCore death resolved by
   `CUDA_LAUNCH_BLOCKING=1`). Irrelevant to this pure-PyTorch project, but avoid adding them.
3. **Unified-memory pitfall:** because CPU and GPU share 128 GB, `pin_memory=True` in a DataLoader can
   cause page-locking thrash and vLLM needed a `gpu_memory_utilization` guard for unified memory
   (vLLM #49760). **Set `pin_memory=False`**; do H2D copies on default streams. Also `Triton`/CUTLASS MoE
   kernels may fail on sm_121 (not used here).
4. Some early NGC/custom images shipped **only sm_120 gencode**; GB10 is sm_121. Verify with
   `torch.cuda.get_device_capability() == (12,1)` and `torch.cuda.get_arch_list()` containing
   `sm_120`/`sm_121`. If missing, fall back to a `cu130` pip wheel or a newer NGC tag.
5. Native **FP8** works well (measured 174 TFLOPS), and `torch._scaled_mm` is fine — but we do **not**
   need it (our model is ~100 M params). `MXFP8/MXFP4` TransformerEngine training is **not yet supported on
   sm_121** (forum report) — avoid.

---

## 2. Parameter and memory accounting (matches the code)

### 2.0 Backbone parameter count — VERIFIED from config

`BAAI/bge-small-zh-v1.5` `config.json` (fetched via web): **hidden_size=512, num_hidden_layers=4,
num_attention_heads=8, intermediate_size=2048, vocab_size=21128, max_position_embeddings=512.**

| Component | Count |
|---|---|
| word embeddings (21128×512) | 10,817,536 |
| position + token_type + emb LayerNorm | 264,192 |
| 4 × BERT block (3,152,384 each) | 12,609,536 |
| pooler | 262,656 |
| **Total** | **23,953,920 (23.95 M)** |

This matches the official "24 M params / 95.8 MB" figure (23.95 M × 4 B = 95.8 MB fp32). **d_model = 512,
NOT 384** — important because all fusion projections scale with d_model.

### 2.1 Hashed N-gram tables (Layer-B)

Topology from the code: orders `2..max_ngram_size` (= {2,3} → **2 orders**), `n_head_per_ngram=8`
heads/order, one **distinct prime ≥ base** per (order, head, layer) via `find_next_prime`, `d_head=32`
columns per row, `layer_ids=[1,3]` → **2 injection layers**.

- tables per layer = (max_ngram_size−1) × n_head = 2 × 8 = **16**
- total tables = 16 × 2 layers = **32**
- rows per layer ≈ 16 × (base + a few) ; total rows ≈ **32 × base** (measured below with the exact prime walk)

Exact totals (computed by replaying `find_next_prime` for `base−1`, 8 consecutive primes, 2 orders, 2 layers):

| `table_size_per_ngram` (base) | total rows | hash params (`rows×d_head`) | notes |
|---|---|---|---|
| 16,384 (small / **S**) | 529,714 | 16,950,848 | mean row = 16,554 (primes cluster just above base) |
| 65,536 (**M**, current default) | 2,101,122 | 67,235,904 | mean row = 65,660 |
| 262,144 (large / **L**) | 8,394,998 | 268,639,936 | mean row = 262,344 |
| 1,048,576 (XL) | 33,561,138 | 1,073,956,416 | |

Because each head gets a *distinct* prime starting at `base−1`, mean rows ≈ base × 1.001 (tiny).

### 2.2 Registry table (Layer-A)

`nn.Embedding(max_registry_slots, d_registry=64, padding_idx=0)`:
- `max_registry_slots=200,000` → **12,800,000 params**. (Slot 0 is NULL/frozen, still allocated.)

### 2.3 Fusion params (per injection layer)

From `EngramFusion`, with `d_mem = num_tables×d_head + d_registry = 16×32 + 64 = **576**`, `d_model=512`,
`n_branches=1`:

| Param | Shape | Count |
|---|---|---|
| `value_proj.weight` (W_V, no bias) | 512×576 | 294,912 |
| `key_projs[0].weight` (W_K, no bias) | 512×576 | 294,912 |
| `norm_q`, `norm_k`, `norm_v` (RMSNorm 512) | 3×512 | 1,536 |
| `gate_bias` | 1 | 1 |
| `conv` (depthwise, k=4, groups=512) | 512×1×4 | 2,048 |
| **per layer** | | **593,409** |
| **×2 layers** | | **1,186,818** |

(Depthwise causal conv, dilation = max_ngram_size = 3 → receptive field (4−1)·3+1 = 10 tokens.)

### 2.4 Full accounting table (all DERIVED; ×d_head=32, d_registry=64, 2 layers, d_model=512)

`DE trainable = hash + registry + fusion`. Backbone (23.95 M) is **frozen** (no grads, no Adam state).

| Config | hash rows | hash params | registry params | fusion params | **DE trainable** | fp32 mem | bf16 mem |
|---|---|---|---|---|---|---|---|
| **S** (16 K) | 529,714 | 16,950,848 | 12,800,000 | 1,186,818 | **30,937,666** (30.9 M) | 124 MB | 62 MB |
| **M** (64 K, default) | 2,101,122 | 67,235,904 | 12,800,000 | 1,186,818 | **81,222,722** (81.2 M) | 325 MB | 162 MB |
| **L** (256 K) | 8,394,998 | 268,639,936 | 12,800,000 | 1,186,818 | **282,626,754** (282.6 M) | 1.13 GB | 565 MB |
| **XL** (1 M) | 33,561,138 | 1,073,956,416 | 12,800,000 | 1,186,818 | **1,087,943,234** (1.09 G) | 4.35 GB | 2.18 GB |

### 2.5 Optimizer + gradient memory

Backbone is frozen (`requires_grad=False` and `.eval()`), so:
- **No backbone gradients** and **no backbone Adam state**. Confirmed in code: `freeze_backbone()` sets all
  backbone `requires_grad=False`; `trainable_parameters()` / `engram_parameters()` only select
  `engram_head_*` and `registry_table.*`.

Per **trainable** memory param, resident bytes (fp32 master weights):

| Item | bytes/param | why |
|---|---|---|
| param | 4 | fp32 weights |
| grad | 4 | dense grad (embedding grads are dense tensors same shape) |
| Adam `exp_avg` (m) | 4 | fp32 |
| Adam `exp_avg_sq` (v) | 4 | fp32 |
| **total resident** | **16 B/param** | |

(`foreach=True` builds fused temporaries transiently; `fused=True` Adam avoids them — see §1.4.)

**Total training footprint = 16 B × DE trainable + activations + backbone weights.**

| Config | 16 B×DE | backbone fp32 | activations (B=64,T=128, est.) | **total resident** | % of 128 GB |
|---|---|---|---|---|---|
| **S** | 0.49 GB | 0.096 GB | ~2.5 GB | **~3.1 GB** | 2.4 % |
| **M** | 1.30 GB | 0.096 GB | ~2.5 GB | **~3.9 GB** | 3.0 % |
| **L** | 4.52 GB | 0.096 GB | ~2.5 GB | **~7.1 GB** | 5.6 % |
| **XL** | 17.4 GB | 0.096 GB | ~2.5 GB | **~20.0 GB** | 15.6 % |

Activation estimate (DERIVED): 3 trainable blocks (indices 1,2,3) store per-layer inputs/attn/FFN
intermediates ≈ 3 × (a few × B×T×512 + B×T×2048) × 4 B. At B=64, T=128 (8192 tok) this is ≈ 2–3 GB.
We do **not** need block 0 activations (nothing before it requires grad) — a real saving.

**Verdict: memory is a non-issue up to config L.** GB10's 128 GB is enormous for a 24 M-param frozen
backbone + 81 M memory module. The plan should treat memory as *not* the constraint (§3 shows compute is).

### 2.6 What still needs storing despite the frozen backbone (activation nuance)

Because we inject at layers 1 and 3 (0-indexed), gradient must flow **back through blocks 1,2,3** to reach
the layer-1 Engram head. So:
- **Blocks 1, 2, 3 retain activations** for backward (they are on the path from the injection point to the
  loss).
- **Block 0 does NOT retain activations.** Nothing trainable exists upstream of block 0 — the embedding
  weights are frozen and `input_ids` are integers — so block 0 runs forward-only with no autograd tape.
  This saves 1/4 of the activation memory and 1/4 of the forward compute sits outside the backward path.
- The token-embedding output *is* retained, because the layer-1 head reads it as an input to the graph.
- The layer-3 head's own inputs/activations are retained too, but it is the last block, so no backbone
  activations are needed downstream of it beyond the pooled output.

Net: **3 of 4 blocks retain activations.** The frozen backbone still costs full forward compute for all 4
layers but backward for only 3. This is why training is ≈1.7× inference FLOPs here, not 3× (§3.1).

---

## 3. Training throughput estimate (DERIVED)

### 3.1 Forward+backward FLOPs per token

Let `d=512`, `d_ff=2048`, `L=4`, `N=seq len`, `B=batch`.

**Backbone, per layer, per token (forward):**
- Q,K,V,O projections: `4 × 2d² = 8d² = 2,097,152`
- FFN: `2 × 2·d·d_ff = 4·d·d_ff = 4,194,304`
- attention score+AV (quadratic in N): `2 × 2·N·d = 4·N·d`
- **layer total** ≈ `6,291,456 + 4·N·512`

At N=128: `6,291,456 + 262,144 = 6,553,600` /layer → ×4 layers = **26.2 MFLOP/token forward (backbone)**.
At N=256: `6,291,456 + 524,288` → ×4 = **27.3 MFLOP/token**.

**Memory module, per token, per injection layer (forward):**
- 16 embedding gathers: memory-bound, ~0 FLOP (a gather is a read).
- W_K: `2·d_mem·d = 2·576·512 = 589,824`
- W_V: `589,824`
- RMSNorms + dot + conv(512×4): ≈ `3·2·512 (norms) + 2·512 (dot) + 512·4 (conv) ≈ 6,656`  (≈ 6.7 kFLOP)
- **layer total** ≈ `1,186,304` FLOP/token forward. ×2 layers = **2.37 MFLOP/token**.

**The gather is O(rows accessed), NOT O(table size)** — explicitly confirming the O(1) lookup claim:
- Rows read per token per layer = `n_orders × n_head = 2 × 8 = 16` rows, each 32 dims.
- Total gathered floats/token = `16 × 32 × 2 layers = 1,024` floats + registry `4×64 = 256` floats.
- This is **independent of `table_size_per_ngram`**. Growing the table from 64 K to 256 K to 1 M adds
  *parameters and memory*, but **zero** additional per-token lookup work. ✔ O(1) lookup confirmed.

**Backward** (training): grads w.r.t. trainable memory params. Backward ≈ 2× forward FLOPs for the
memory module (dgrad + wgrad through W_K/W_V); backbone backward adds dgrad for the 3 trainable-path blocks
but not wgrad for frozen weights. Net **training ≈ 1.6–2.0× forward** for this architecture. Use **≈1.7×**.

**Total (memory on, training, per token):**

| N | fwd backbone | fwd memory | fwd total | **train total (×~1.7)** |
|---|---|---|---|---|
| 128 | 26.2 MFLOP | 2.37 MFLOP | 28.6 MFLOP | **~48.6 MFLOP/token** |
| 256 | 27.3 MFLOP | 2.37 MFLOP | 29.7 MFLOP | **~50.4 MFLOP/token** |

(Memory-off training ≈ same minus 2.37 MFLOP ×1.7 ≈ 4 MFLOP/token, i.e. the memory module is **~8–9 % of
training FLOPs** — small, which is the whole point of the design.)

### 3.2 Realistic tokens/sec on GB10 — is it FLOP- or bandwidth-bound?

**FLOP ceiling.** Effective end-to-end rates (see §1.3): 8 / 18 / 35 TFLOPS.

Training tokens/s at N=128 (48.6 MFLOP/token):
- 8 TFLOPS → 165 k tok/s
- 18 TFLOPS → **370 k tok/s** (central)
- 35 TFLOPS → 720 k tok/s

**Bandwidth ceiling from the embedding gather — this is the key "is it memory-bound?" analysis.**

Bytes moved by the gather per token (fp32 rows, d_head=32):
- 16 rows/layer × 32 dims × 4 B × 2 layers = 4,096 B/token (hash) + 2,048 B/token (registry, 4×64×4×2) ≈ **6.1 KB/token** read.

Backward adds grad writes of the same volume → **~12 KB/token** of memory traffic for the gather.

At 273 GB/s, 12 KB/token → **273e9 / 12,288 ≈ 22 M tok/s** ceiling for the gather alone. This is **~60×
above** the compute ceiling (370 k tok/s). 

**Conclusion (DERIVED): the embedding gather is NOT the bottleneck up to and far beyond config XL.** The
"bottleneck may be the gather" hypothesis is **false** here because `d_head=32` (128 B/row) is a tiny
per-row footprint and each token reads only 16–32 rows regardless of table size.

**Where table size DOES start to matter — the Adam step sweep.** Adam touches every memory parameter each
step (16 B/param), so its cost scales **linearly with table size**, not with rows accessed. Adam step time
= `16 B × P_DE / 273 GB/s`:

| Config | P_DE | Adam sweep bytes | **Adam step time** |
|---|---|---|---|
| S | 30.9 M | 0.49 GB | 1.81 ms |
| **M** | 81.2 M | 1.30 GB | **4.76 ms** |
| L (200 k slots) | 282.6 M | 4.52 GB | 16.6 ms |
| L (1 M slots) | 333.8 M | 5.34 GB | 19.6 ms |
| L, 3 layers, 1 M slots | 468.9 M | 7.50 GB | 27.5 ms |
| XL | 1.09 G | 17.4 GB | 63.8 ms |

**This Adam sweep is the first thing to become a bottleneck as you grow the table.** Compare against the
compute time of a real training step. `de/train.py` runs **3 separate forwards** per step (query,
positive, negative), each of batch B and length N, so tokens/step = `B × 3 × N`. At B=64, N=128 →
**24,576 tokens/step**, compute at 48.6 MFLOP/token:

| effective F | compute/step | Adam crossover (P_DE where sweep = compute) |
|---|---|---|
| 8 TFLOPS | 149.3 ms | 2,547 M |
| **18 TFLOPS (central)** | **66.4 ms** | **1,132 M** |
| 35 TFLOPS | 34.1 ms | 582 M |

So the Adam sweep as a fraction of step time:

| Config | @8 TF | @18 TF | @35 TF |
|---|---|---|---|
| S | 1.2 % | 2.7 % | 5.3 % |
| **M** | **3.2 %** | **7.2 %** | **13.9 %** |
| L (1 M slots) | 13.1 % | 29.5 % | 57.3 % |
| L, 3 layers, 1 M slots | 18.4 % | 41.4 % | 80.5 % |
| XL | 42.7 % | 96.1 % | **186.8 %** ← optimizer dominates |

**Answer to "at what table size does memory become the bottleneck": the *gather* never does (§3.2 above);
the *dense-Adam sweep* does, and the crossover is at roughly 0.6 B DE params (35 TF), 1.1 B (18 TF), or
2.5 B (8 TF). Config M is far below all of these. Config L with 1 M registry slots reaches ~30 % overhead
at central TFLOPS — acceptable. Config XL breaks even with compute at 18 TF and loses at 35 TF.**

**Recommendation: cap the first real run at config M; allow config L with a fused optimizer;
treat XL as requiring `fused`/8-bit Adam plus a larger batch, not as a free win.**

### 3.3 Wall-clock for warm-up training (DERIVED)

Setup: 3 forward passes per step (query, positive, negative) in `train_stage1`, so tokens/step =
`B × N × 3`. A "triplet" row in the dataset = one such step-item. Let `F` = effective TFLOPS, and use
`48.6 MFLOP/token`, `Adam sweep per step` from §3.2, and a **fixed overhead factor 1.25×** for framework/
kernel-launch/IO not captured by the GEMM model (this is a judgment call, labeled as such).

Per **1 M triplets** (1 M steps of B=1, i.e. per-triplet tokens = 3N):

**Config M, B=64, N=128** (24,576 tokens/step, 1 M triplets = 15,625 steps):
- compute/step = 24,576 × 48.6e6 / F. At F=18e12 → 66 ms/step.
- Adam/step = 4.76 ms.
- net/step ≈ (66 + 4.76) × 1.25 ≈ **88 ms**.
- steps for 1 M triplets = 1e6/64 = 15,625 → **1,375 s ≈ 0.38 h per 1 M triplets**.

| workload | config M, B=64, N=128 (18 TF) | at 8 TF (slow) | at 35 TF (fast) |
|---|---|---|---|
| **5 M triplets** | **~1.9 h** | ~4.3 h | ~1.0 h |
| **50 M triplets** | **~19 h** | ~43 h | ~10 h |

At N=256 multiply tokens/step by 2 → **~2× wall-clock** (≈ 3.8 h for 5 M, ≈ 38 h for 50 M at 18 TF).

At **B=32** instead of 64: tokens/step halves (12,288) → Adam fraction doubles, per-step compute halves,
so wall-clock rises ~10–15 % (more optimizer overhead per token). B=64 is the better default on this box.

**Sanity check vs. the machine:** 5 M triplets × 3 × 128 = 1.92 G tokens; at 370 k tok/s = **5,190 s ≈ 1.4 h**
pure-compute, plus Adam + overhead → ~1.9 h. Consistent.

---

## 4. Cold-start write cost (`cold_start` / `solve_slot_closed_form`)

Per new term, the code (DERIVED from `cold_start.py`):
1. **Step 1 — aggregate target**: run the frozen backbone over `K` support contexts (default `max_len=192`),
   take the hidden state at the term's last token, average → `s*`.
2. **Step 2 — closed-form ridge solve**: for each context compute `residual = (s*−h_pre) − W_h·e_hash`, then
   `e = A⁻¹ W_rᵀ residual` where `A = W_rᵀW_r + λI`, and average `e` over contexts, write one row.

**Linear-algebra size**: `d_reg_total = d_registry × max_matches_per_pos = 64 × 4 = **256**`. So `A` is
256×256, `W_rᵀ` is 256×512. Per context solve = one triangular solve of a **256×256** system: ≈ (2/3)·256³
≈ 11 MFLOP — trivially fast on GPU (sub-100 µs at these sizes; launch-latency bound, not FLOP bound).

**Per-term latency (DERIVED, amortized):**
- `A` is refactored once per *call* (not per context), 256³ factor + 256×512×256 `W_rᵀW_r` ≈ 33 MFLOP → **~0.2–0.5 ms**.
- Each context's residual needs 1–2 small GEMV (`W_h·e_hash`, `W_rᵀ·resid`) ≈ 0.5 MFLOP → **~10 µs each**, plus
  the solve **~50–100 µs**. With K=8 contexts: solve portion ≈ 8×0.1 ms + 0.4 ms ≈ **~1.2 ms**.
- **Dominant cost = Step 1**: K full backbone forwards (frozen, inference-only) at len 192. At 18 TFLOPS,
  one ctx forward = 192 × 26.2 MFLOP / 18e12 ≈ 0.28 ms compute, but **batch-of-1 inference is
  launch/overhead-bound, ~1–3 ms wall-clock**. K=8 → **~8–25 ms**.
- **Per-term total: ~10–25 ms amortized** (the ridge solve itself is ~1 ms; the rest is the aggregation forwards).

*How to make it ~1 ms (recommended):* **batch the K contexts** into one forward (B=K) instead of looping
them one at a time. Then Step 1 becomes one batched inference pass (~1–2 ms) + one batched solve, giving
**~2–4 ms/term**. This is the single biggest lever and matches the "millisecond updates" claim.

**Inserting 100 k terms (DERIVED):**
- Naive (loop of 100 k `cold_start`, each ~10–25 ms): **~17–42 min** (100 k × 15 ms ≈ 25 min).
- **Batched contexts (~2–4 ms/term): 100 k × 3 ms ≈ 300 s ≈ 5 min**, plus registry rebuild overhead.
- **Registry rebuild gotcha:** every `add()` bumps `registry.version`, and the `PhraseMatcher` rebuilds its
  hash index **lazily on the next `maybe_rebuild()`** — O(#active phrases). Calling `cold_start` one term at
  a time would trigger a rebuild per term ⇒ O(N²). **Bulk-import must add all terms first, then call
  `sync_matcher()` once** → one O(N log N) rebuild (~0.1 s for 100 k). This is required for the 5-min number.

**Net claim check:** "millisecond updates, no retraining" is **accurate for the write** once contexts are
batched (~2–4 ms/term, and the *retrieval-visible* update is instant since it's just a row write), and
100 k terms ≈ 5 min (batched) vs. a ~2–19 h warm-up retrain. The ordering of magnitude is sound; the
microsecond-vs-millisecond framing in the code docstring ("microseconds per new term") is optimistic —
the honest number is **single-digit ms with batching**, and that's what backs "milliseconds, no retraining."

---

## 5. Inference / encoding throughput (DERIVED)

Forward-only, per token (from §3.1):
- memory **off**: 26.2 MFLOP/token (N=128); memory **on**: 28.6 MFLOP/token.
- **Projected slowdown from memory = 28.6 / 26.2 = 1.091 → ~9 % more FLOPs.** Because the gather is
  memory-light (§3.2) and the fusion module is tiny (2.4 MFLOP/token), the projected slowdown is small.

Encoding tokens/sec (F=18 TFLOPS central, N=128):
- **memory off:** 18e12 / 26.2e6 ≈ **687 k tok/s**
- **memory on:** 18e12 / 28.6e6 ≈ **629 k tok/s**
- (8 TF: ~305 k vs 280 k; 35 TF: ~1.34 M vs 1.22 M.)

Concrete example (DERIVED): 1 M documents × 128 tokens = 128 M tokens:
- memory off: ~186 s (~3.1 min); memory on: ~203 s (~3.4 min). Cost of enabling memory: **~17 s per 1 M docs**.

**Real-world overhead (DERIVED caveat):** these are compute-model numbers. In practice `encode()` /
batch inference adds kernel-launch and CPU-slot-matching overhead (`PhraseMatcher.match_batch` runs on CPU
per batch in `encode()`), which at small batch will cut real throughput below these figures. With large
batches and the GPU saturated, the ~9 % FLOP slowdown is the right order of magnitude. **`memory_enabled=False`
must give bit-identical backbone output** (the code's NULL-slot design guarantees `e=0 ⇒ y=0`), so the
off-path is a clean A/B baseline.

### Zipfian hot/cold caching (Engram §2.5)

Engram §2.5 recommends tiering the memory by access frequency (Zipf): hot N-grams resident on GPU, cold
tail on CPU/SSD, because hashed N-gram lookups follow a power law so a small hot set absorbs most traffic.
Applied here:
- Our hashed tables (config M = 67 M params = 269 MB fp32) **fit entirely in GB10's 128 GB unified
  memory**, so hot/cold CPU offload is **unnecessary for configs S/M/L** — the whole point of GB10's
  large unified memory is that this tiering trick (needed for a 100 B table on H800) is moot here.
- Hot/cold caching **becomes relevant at config XL** (1.09 G params = 4.35 GB) *only if* you go far larger
  (say 10 B+ params), which we are not.
- **Registry (Layer-A) is the tiering target instead.** The registry is the operator-writable, O(1) surface.
  Keeping the top-N most-frequent domain terms' vectors pinned on GPU and cold terms in the (already
  on-CPU `PhraseRegistry`) index is natural. Practically: with 200 k slots × 64 = 12.8 M params (51 MB),
  it all fits on GPU too — so tiering is again optional; prioritize simplicity.
- **Where Zipf actually helps us:** the `max_registry_slots` budget and the choice of `table_size_per_ngram`
  should be allocated by observed Zipf frequency of domain N-grams, so the most frequent patterns land in
  the hashed tables (which are dense/trained) rather than being under-served. This is a *design-time*
  allocation, matching Engram's "sparsity allocation" idea, not a runtime cache. Runtime hot/cold GPU/CPU
  split is unnecessary on this box.

---

## 6. Recommended configuration for the first real GB10 run

### 6.1 Run 1 — mechanism validation (recommended first run)

```python
EngramConfig(
    backbone_name       = "BAAI/bge-small-zh-v1.5",   # d_model=512, 4 layers, 23.95M (frozen)
    layer_ids           = [1, 3],                      # 0-indexed; 2 injection points (shipped default)
    max_ngram_size      = 3,                            # orders {2,3}
    n_head_per_ngram    = 8,
    table_size_per_ngram= 65536,                        # config M: 2.10M rows total, 67.2M hash params
    d_head              = 32,
    max_registry_slots  = 200_000,                      # 12.8M params
    d_registry          = 64,
    max_matches_per_pos = 4,
    conv_kernel_size    = 4,
    pooling             = "cls",
    normalize           = True,
    freeze_backbone     = True,
    torch_dtype         = "float32",                   # keep fp32 weights; use autocast bf16 for compute
)
# Training (de/train.py):
#   batch_size = 64          (triplets/step; ~24.6k tokens/step at len 128)
#   max_len    = 128
#   lr         = 2e-5        (fusion); memory group = 5x  -> 1e-4  (Engram §4.1)
#   memory_lr_mult = 5.0 ; memory weight_decay = 0.0 ; conv zero-init
#   temperature = 0.02 (InfoNCE)
#   epochs     = 1 (warm-up) ; grad_clip = 1.0
```

**Justification per field:**
- `layer_ids=[1,3]` — the **shipped default** (`de/config.py`), so run 1 needs zero code change and matches
  the existing smoke tests. Engram's layer scan found an early single point (layer 2) is best and that two
  points (early + late) beat one; with only 4 blocks, `[1,3]` is the closest analogue ("after block 2" and
  "after block 4"). Note block 3 is the **last** block, so the layer-3 head feeds pooling with no
  downstream refinement. **Ablate `[1]`, `[2]`, `[1,2]`, `[1,3]` in run 2** — this is the one topology
  choice worth an early sweep, since with only 4 layers the injection sites are not interchangeable.
- `table_size_per_ngram=65536` — config M: 81 M trainable params total, ~1.30 GB Adam state. Per §3.2 the
  optimizer sweep (4.76 ms) stays well under compute (66 ms) at B=64. Large enough for 5 M-triplet
  warm-up (≈1.9 G token-positions spread over 2.1 M rows → plenty of coverage) without an optimizer wall.
- `d_head=32` — matches Engram's per-head dim style for small models; 128 B/row keeps the gather
  bandwidth-light (§3.2). Not a bottleneck, so no need to shrink; keeping ≥16 keeps one row ≈ a 64–128 B
  sector (efficient).
- `n_head_per_ngram=8` — the Engram default; gives 16 tables/layer (512 rows/token). More heads would
  increase gather volume linearly with no compute benefit.
- `max_registry_slots=200_000, d_registry=64` — 12.8 M params (51 MB). 200 k domain terms is a generous
  operator-facing budget; O(1) CRUD and fits trivially.
- `batch_size=64, max_len=128` — 24.6 k tokens/step: large enough that the Adam sweep is a small fraction
  of step time; N=128 matches typical query/doc lengths for retrieval (PLAN §2) and keeps wall-clock
  tractable (5 M triplets ≈ 1.9 h).
- `lr=2e-5`, memory `×5 → 1e-4`, `wd=0` on memory, conv zero-init — Engram §4.1 verbatim (per `train.py`).
- `torch_dtype="float32"` + `torch.autocast("cuda", bfloat16)` — keep the **master weights fp32** (memory
  tables are the trained object; bf16 masters + Adam at bf16 lose precision on rare rows). Use bf16 autocast
  for speed. This is a **code/config choice to make explicit**: current `train_stage1` does not use autocast;
  add `with torch.autocast(...)` around forward/backward.
- **Recommended run-time additions** (not config, but first-run hygiene):
  - `DataLoader(..., num_workers=8, pin_memory=False, persistent_workers=True)` — the current
    `num_workers=0` makes the CPU-side `PhraseMatcher` serialize against GPU compute; on a 20-core Arm CPU
    8 workers remove that. (`pin_memory=False` per unified-memory gotcha, §1.4.)
  - `Adam(..., fused=True)` for the memory group — cuts the 4.76 ms sweep.
  - `attn_implementation="sdpa"` (default in modern transformers for BERT) — **do not** install
    flash-attn (SDPA is equal/faster on GB10, §1.4).

**Run-1 expected cost (DERIVED):** 5 M triplets, 1 epoch, ≈ **1.9 h** (18 TFLOPS central) on one GB10.

### 6.2 Run 2 — scale-up (if results are promising)

**2a. Capacity scale-up** (same topology, bigger tables — isolates capacity from topology, cleaner science):

```python
EngramConfig(
    layer_ids            = [1, 3],        # UNCHANGED from run 1
    table_size_per_ngram = 262144,         # config L: 8.39M rows, 268.6M hash params
    max_registry_slots   = 1_000_000,      # 64.0M registry params (256 MB)
    d_registry           = 64,
    d_head               = 32, n_head_per_ngram = 8,
    # same backbone (frozen), pooling=cls, max_ngram_size=3
)
# batch_size = 64 ; max_len = 192 or 256 ; lr = 2e-5 (fusion), memory 5x
```

- DE trainable = 268.6 M (hash) + 64.0 M (registry) + 1.19 M (fusion) = **333.8 M**.
- Adam (16 B/param, fp32) = **5.34 GB** → per-step optimizer sweep **19.6 ms** at 273 GB/s.
- Compute at B=64, N=256 (49,152 tokens/step) at 18 TFLOPS ≈ 49,152 × 50.4e6 / 18e12 ≈ **138 ms/step**.
  → optimizer is ~14 % of the step. Comfortable. **This is the practical ceiling for dense Adam on GB10.**
- Per-token FLOPs barely move (the gather is O(rows accessed), §3.1), so 5 M-triplet warm-up ≈ **2.2–4 h**
  (slightly more than run 1 due to the larger optimizer sweep + longer sequences, not more compute).
- Expected 50 M triplets, N=256: **~44 h** central.

**2b. Topology ablation** (independent axis — do this *before* 2a, it is nearly free):

Sweep `layer_ids ∈ {[1], [2], [1,2], [1,3]}` **at config M**. Each injection point owns its **own** hashed
tables (`EngramTableHead` creates its own `nn.Embedding`s), so going from 2 → 3 points multiplies the hash
params by 1.5. Cost of the 3-point variant, for reference:

| layers | base | slots | hash | registry | fusion | DE total | Adam 16 B | sweep |
|---|---|---|---|---|---|---|---|---|
| 2 | 262144 | 200 k | 268.6 M | 12.8 M | 1.19 M | 282.6 M | 4.52 GB | 16.6 ms |
| 2 | 262144 | 1 M | 268.6 M | 64.0 M | 1.19 M | 333.8 M | 5.34 GB | 19.6 ms |
| **3** | 262144 | 1 M | **403.1 M** | 64.0 M | 1.78 M | **468.9 M** | **7.50 GB** | **27.5 ms** |

The 3-point variant is still fine on GB10 (27.5 ms sweep vs ~138 ms compute at N=256), but it is **not**
where the extra capacity buys quality — Engram's own ablation ranks multi-branch fusion, context gating
and tokenizer compression *above* injection count (§1 of PLAN.md). Spend the budget on tables first.

> ⚠️ **Hard stop at ~470 M DE params.** Past that the dense-Adam sweep (§3.2) starts to dominate step
> time and you must switch to `fused`/8-bit Adam or raise the batch — do not simply grow the table.

---

## 7. Consolidated numbers quick-reference (GB10)

| Quantity | Value | Label |
|---|---|---|
| Unified memory | 128 GB LPDDR5x, 273 GB/s | VERIFIED |
| Compute capability | sm_121 (12.1) | VERIFIED |
| CPU | 20-core Arm (10×X925 + 10×A725, aarch64) | VERIFIED |
| Dense BF16 / FP8 / FP4 TFLOPS | 118.8 / 237.7 / 475.4 | VERIFIED |
| Backbone params | 23.95 M (d=512, 4 layers) | VERIFIED (config) |
| DE trainable, config M | 81.2 M | DERIVED |
| Adam+grad+param, config M | ~1.30 GB | DERIVED |
| Training FLOPs/token (N=128, mem on) | ~48.6 MFLOP | DERIVED |
| Memory module share of training FLOPs | ~9 % | DERIVED |
| Embedding gather traffic | ~12 KB/token (mem-light, **not** bottleneck) | DERIVED |
| Adam sweep becomes bottleneck | ≈ config L (282 M params) @ B=64 | DERIVED |
| Training tokens/s (18 TF central) | ~370 k tok/s | DERIVED |
| **5 M triplets, config M, B=64, N=128** | **~1.9 h** (8 TF: 4.3 h; 35 TF: 1.0 h) | DERIVED |
| **50 M triplets, config M, B=64, N=128** | **~19 h** (8 TF: 43 h; 35 TF: 10 h) | DERIVED |
| Encode, memory off / on (18 TF) | ~687 k / ~629 k tok/s (**~1.09× slowdown**) | DERIVED |
| 1 M docs × 128 tok encode | off ~3.1 min / on ~3.4 min | DERIVED |
| Cold-start per term (batched ctx) | ~2–4 ms (solve itself ~1 ms) | DERIVED |
| Insert 100 k terms (batched) | ~5 min (build registry once, then one matcher rebuild) | DERIVED |
| Recommended wheel/container | NGC `pytorch:25.08+` (DGX Spark tagged) or `torch --index-url .../cu130` | VERIFIED |
| Attention | **SDPA** (FA not beneficial on GB10) | VERIFIED |

---

## 8. Open items / what to measure on the real box

These throughput numbers are analytic. Before committing to a long run, **microbenchmark on the actual
GB10** (the analytic model has ~25 % overhead slack built in):

1. **Achieved GEMM TFLOPS** for our exact shapes (M=24 k, K=N=512 / N=2048): microbench `torch.matmul`
   fp32 and bf16-autocast; confirm we're near 18–35 TFLOPS end-to-end, or recalibrate the wall-clocks.
2. **Real Adam step time** with `fused=True` (validate the 4.76 ms config-M sweep).
3. **Encode throughput** with SDPA at batch 64–256; confirm the ~9 % memory-on overhead end-to-end
   (and that `memory_enabled=False` is bit-identical to the baseline).
4. **Cold-start at scale**: batch contexts, add-terms-then-one-rebuild, time 100 k inserts.
5. **Registry matcher cost** on CPU with `num_workers=8`; confirm it is off the critical path.

Sources for VERIFIED facts: NVIDIA DGX Spark datasheet & product page; NVIDIA DGX Spark User Guide
(hardware, NGC); TechPowerUp GB10 GPU DB; PyTorch 2.7/2.11 blogs; Dao-AILab/flash-attention #1969 &
#2842; vLLM #50067 & #49760; NVIDIA GB10 developer-forum FA-vs-SDPA post; Isaac-GR00T #474;
ugonfor/DGX-Spark-Benchmark; G3nadh/dgx-spark-benchmarks; llama.cpp DGX Spark thread; BAAI/bge-small-zh-v1.5
config.json.
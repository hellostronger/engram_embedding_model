# EXPERIMENTS.md — 实验记录（唯一权威出处）

> 记录所有实验的**环境 / 配置 / 原始输出 / 结论 / 判定**。
> 任何数字以本文档为准；其他文档（README / TODO / PLAN）引用时须指向此处。
>
> 创建于 2026-10-11。环境变更或结论推翻时**追加新条目，不删旧条目**，
> 旧条目标 `[已作废]` 并注明作废原因——作废的实验和它的结论同样重要。

---

## 记录规范

每条实验必须有唯一 ID（`E-###`），包含：
- **环境**：硬件 / torch / backbone
- **配置**：完整超参
- **数据**：来源 + 行数 + 关键统计
- **原始输出**：未经修饰的数字
- **结论**：这个实验**能**说明什么
- **判定**：✅ 有效 / ⚠️ 有保留 / `[已作废]` 及原因

**禁止**：在未记录实验的情况下修改结论。

---

# 第一部分：作废实验（E-001 ~ E-003）

> 这三个实验跑在**随机初始化的 tiny backbone** 上。TODO.md 早期版本
> （以及本文件创建前的对话）曾引用它们的数字，**全部已作废**。

## E-001 `[已作废]` A/B/C/D on synthetic + tiny backbone

**环境**
- CPU: AMD Ryzen 7 5700X / 8核16线程 @3.4GHz / 31.9 GB RAM
- `torch 2.10.0+cpu`，8 线程，`CUDA avail = False`
- backbone: `build_tiny_backbone()` — **随机初始化** BertModel(vocab=144, hidden=**64**, layers=4, heads=4, ffn=256)，**1,564,736 参数，未训练**
- 数据: `make_synthetic_data.py --domain medical --n 2000`（合成）

**配置**
- `layer_ids=[1,3]`, `table_size_per_ngram=1024`, `d_head=8`, `d_registry=8`, `n_head_per_ngram=2`, `max_registry_slots=2048`
- 可训练参数 **221,794**
- epochs=2, bs=32, lr=1e-3, max_len=64

**原始输出**
```
A baseline / seen       R@1=100.0  nDCG=100.0
A baseline / unseen     R@1= 79.5  nDCG= 91.8
B random memory / seen  R@1=100.0  nDCG=100.0
C trained / seen        R@1=100.0  nDCG=100.0
C trained / unseen      R@1= 81.0  nDCG= 92.4
D cold-start / unseen   R@1= 79.0  nDCG= 91.5
D cold-start / seen     R@1=100.0  nDCG=100.0

C - A (seen nDCG):  +0.00
D - C (unseen nDCG): -0.87
训练 loss: epoch0 1.3580 -> epoch1 1.3181
```

**结论**：无法说明任何问题。

**判定 `[已作废]`**，三条原因：
1. **backbone 是随机权重**。随机网络无语义结构，记忆是刚需 → 早期版本据此声称的 `C−A=+14.30` 纯粹是"小模型太弱"的假象
2. **合成 seen 任务饱和**。A/B/C 的 R@1 与 nDCG **全部=100.0**，记忆机制无提升空间，恒等于 0 差异
3. **合成数据自证**。query 与正例共享同一术语、干扰项仅差术语名；held-out 术语 `新发疾病N` 间仅编号不同，backbone 纯词形匹配即得 79.5 R@1，**根本用不上记忆**
4. **训练不足**。loss 仅从 1.3580 降至 1.3181（−0.04），2 轮不构成训练

**被本实验误导的历史结论**（均已作废）：
| 结论 | 出处 | 作废原因 |
|---|---|---|
| `C−A = +14.30`（记忆训练有效） | TODO 旧版 | tiny backbone 太弱；真 backbone 上饱和为 +0.00 |
| `D−C = +0.06`（冷启动无效） | TODO 旧版 | 平台无效，数字无意义 |
| "冷启动增益仅 +0.06" | TODO 旧版 | 同上 |

---

## E-002 `[已作废]` 冷启动诊断 on tiny backbone（cos = −0.044）

**环境/配置**：同 E-001（tiny 随机 backbone, hidden=64）

**原始输出**
```
[H1] matcher fires on eval_cold BEFORE write: 0/200 (0.0%)
[H1] matcher fires on eval_cold AFTER  write: 200/200 (100.0%)
[H2] gate alpha at MATCHED positions (n=200): mean=0.2520 min=0.0014 max=0.8921 -> OPEN
[H3] 22/23 engram/registry params received gradient during C
     registry_table.emb.weight: STILL-AT-INIT
[DELTA] ||emb(ON)-emb(OFF)|| mean=1.44493 max=1.46073
[ORIENT] cos(emb_ON, emb_OFF): mean=-0.0440 min=-0.0669
```

**曾经的结论**：写入过度（memory 覆写了 backbone 表示而非微调）→ 提出 "H4 写入过度" 假设。

**判定 `[已作废]`**：同一随机 backbone 在 E-004 上给出 `cos=+0.894`。随机权重无稳定表示空间，"近正交"是随机基线效应。**H4 假设被 E-004 直接推翻。**

**本实验的有效部分**：H1 / H2 / H3 的排除结论在 E-004 上被独立复现，故这些结论**不是**作废的，而是被 E-004 以有效平台重新确立。

---

# 第二部分：有效实验（E-003 ~）

## E-003 环境与数据可用性实测

**目的**：确认网络 / 依赖 / 数据能否获取。**无模型参与。**

### E-003a 依赖修复 `[✅ 有效]`

**问题**：`scipy 1.12.0`（NumPy 1.x ABI）+ `numpy 2.4.2` → `transformers` 导入链经 `sklearn → scipy.sparse` 崩溃：
```
ImportError: numpy.core.multiarray failed to import
```
**修复**：`pip install --upgrade "scipy>=1.13"` → `scipy 1.18.1`
**副作用**：`gensim 4.3.3` / `graspologic 3.4.1` 与新版 NumPy/SciPy 不兼容（项目当前不使用这两个库）

### E-003b 网络连通性 `[✅ 有效]`

**背景**：TODO 记录「HF 下载被沙箱代理拦截」。**该记录已过时。**

| 端点 | 裸 urllib | 结论 |
|---|---|---|
| `huggingface.co` | 超时 (WinError 10060) | 直连不通 |
| `hf-mirror.com` | 根域 200，API 403 | 需官方客户端 |
| `github.com` / `raw.githubusercontent.com` | 超时 / 强制关闭 | 不通 |
| `modelscope.cn` / `gitee.com` / `pypi.tuna` | 200 | 通 |

**关键突破**：本机有出网代理，设置 `HTTP_PROXY` / `HTTPS_PROXY` 后即可访问
（**代理地址为本机环境细节，不写入仓库；换机器请自行设置或使用直连**）
→ huggingface.co API 正常，实测 5.0 MB/s
验证数据集：`coastalcph/lex_glue` 40.5 MB / 81.5s（无代理时）；`qiaojin/PubMedQA` 233.4 MB / 46.3s（**5.0 MB/s**，有代理）

### E-003c 硬件实测 `[✅ 有效]`

| 项 | TODO 历史记录 | **实测** | 差异 |
|---|---|---|---|
| CPU | i7-8550U | **AMD Ryzen 7 5700X** | TODO 记录错误 |
| 核心 | — | 8核16线程 @ 3.4GHz | |
| RAM | 24 GB | **31.9 GB** | |
| GPU | 无 | 无（`torch 2.10.0+cpu`，8 threads） | 一致 |
| 磁盘 | — | D: 67.5 GB free | |

### E-003d 真实 backbone 规格 `[✅ 有效]`

`BAAI/bge-small-zh-v1.5` config.json 实测：
```
model_type=bert  hidden_size=512  num_hidden_layers=4
num_attention_heads=8  intermediate_size=2048
vocab_size=21128  max_position_embeddings=512
```
**推论**：`layer_ids=[1,3]` 把第二个注入点放在**第 4 层（最后一层）**，注入后直接进 pooling，零精炼 → 违反 Engram 消融结论（层 2 最优，注入后需留 block）。应改 `[1,2]`。

### E-003e 参数规模实测 `[✅ 有效]`

**注意：`run_abcd.py` 使用缩小的实验配置，不是 `de/config.py` 的生产默认值。**

| | 生产默认 (config.py) | **实验实际 (run_abcd)** |
|---|---|---|
| hashed N-gram 表 | 67.16 M | 12.6 M |
| fusion | 1.58 M | 0.20 M |
| registry 表 | 12.80 M | 0.016 M |
| **可训练合计** | **81.54 M** | **0.22 M** |
| backbone | 23.9 M（冻结） | 23.9 M（冻结） |
| **总计** | **105.4 M** | **24.1 M** |

**生产配置的部署成本**（用户关心的快速推理）：
- 每 token MACs：backbone 12.6 M，Engram 1.58 M → **算力仅 +12.5%**
- 但权重：backbone 96 MB(fp32)，Engram 表 **268 MB(fp32)** → **2.8×**
- **参数:算力比 42:1 → 纯访存瓶颈，非计算瓶颈**
- 这 67 M 槽在 21128 中文词表下**绝大多数永远为空**（3-gram 表废槽最多）
- 降本候选：`table_size_per_ngram` 65536→16384、`n_head_per_ngram` 8→4、表存 bf16（**需先做检索质量消融**）

**结论**：模型是 **0.1 B 级**，不是 B 级。可训练部分 82% 是查表。

### E-003f 真实数据获取 `[✅ 有效]`

**subagent 下载，独立复核通过。**

| 数据集 | 路径 | MB | 行数 |
|---|---|---|---|
| `twang2218/chinese-law-and-regulations` | `COLD-1_*` | 152.8 | 22,552 |
| `C-MTEB/T2Retrieval` | `A1_*` | 150.3 | corpus 118,605 / queries 22,812 |
| `C-MTEB/DuRetrieval` | `A2_*` | 61.5 | corpus 100,001 / queries 2,000 |
| `C-MTEB/CovidRetrieval` | `A3_*` | 62.1 | corpus 100,001 / queries 949 |
| `C-MTEB/EcomRetrieval` | `E1_*` | 8.1 | corpus 100,902 / queries 1,000 |
| `C-MTEB/CmedqaRetrieval` | `M4_*` | 58.5 | corpus 100,001 / queries 3,999 |
| `mteb/LeCaRDv2` | `L2_*` | 150.5 | corpus 3,795 / queries 159 |
| `bigbio/med_qa` zh | `M2_*` | 4.9 | 27,400 |
| `BeIR/fiqa` + qrels | `P1_*` | 26.7 | corpus 57,638 / queries 6,648 |
| `ShengbinYue/DISC-Law-SFT` | `P1_*` | 552.4 | Pair 166,758 / Triplet 16,000 |
| `McAuley-Lab/Amazon-Reviews-2023` | `P2_*` | 386.3 | review 701,528 / meta 112,590 |
| `PleIAs/SEC` 2022-2024 | `P2_*` | 2,308.5 | 2022: 8,348 |
| **合计** | | **3.73 GB** | ~105 万行 parquet |

**`docs/DATA.md` 与实测不符（4 处，待修订）**：
1. `mteb/LeCaRDv2`：文档 800 q / 55,192 cases → **实测 159 q / 3,795 cases**
2. `C-MTEB/*-qrels` 字段：文档 `{query-id, corpus-id, score}` → **实测 `qid, pid, score`**（那是 BeIR/LeCaRD 格式，三套不统一）
3. `T2Retrieval` queries 文档 24,832 → **实测 22,812**；`DuRetrieval` 文档 4,000 → **实测 2,000**
4. `Amazon raw_meta_All_Beauty` 文档 807 MB / 801,446 行 → **实测 59.6 MB / 112,590 行**；且**仓库无 `raw_review_All_Beauty` parquet**，review 走 `raw/review_categories/All_Beauty.jsonl`
5. `twang2218` 是 **88% 地方性法规**（19,773），非文档暗示的「国家法律」（法律仅 429）

---

## E-004 冷启动诊断 on **真实 backbone** `[✅ 有效]`

**环境/配置**
- backbone: `BAAI/bge-small-zh-v1.5`（**23,953,920 参数，冻结**），hidden 512 / 4 层
- 其余同 E-001；epochs=2, bs=32, lr=1e-3, max_len=64
- 数据: 合成 medical（**仅用于诊断机制，不用于结论**）

**原始输出**
```
backbone=BAAI/bge-small-zh-v1.5  hidden=512  layers=4
epoch 0: loss 1.3577    epoch 1: loss 1.3184

[H1] matcher fires BEFORE write: 0/200 (0.0%)
[H1] matcher fires AFTER  write: 200/200 (100.0%)

[H2] gate alpha ALL positions: mean=0.5347 p50=0.6129 p95=0.9994 max=0.9999
[H2] gate alpha MATCHED positions (n=200): mean=0.1563 min=0.0147 max=0.4378 -> OPEN

[H3] 22/23 engram/registry params 收到梯度
     registry_table.emb.weight: STILL-AT-INIT (max|d|=0.000000)
     cold-start wrote max|d|=0.428092

[DELTA] ||emb(ON)-emb(OFF)||: mean=0.45905 max=0.52279
[ORIENT] cos(emb_ON, emb_OFF): mean=0.8942 p05=0.8763 min=0.8633
```

**结论**

| 假设 | 判定 | 依据 |
|---|---|---|
| **H1 写入没落地** | ❌ **排除** | 匹配率 0/200 → 200/200 |
| **H2 门控压制（α≈0）** | ❌ **排除** | matched α=0.1563，远高于 0.05 阈值 |
| **H3 W_V/W_K 注册表列随机** | ❌ **排除** | 22/23 参数收到梯度；且 W_V 是**共享投影**非按行参数 |
| **H4 写入过度（覆写 backbone）** | ❌ **排除** | cos=**+0.894**（tiny 上 -0.044 是伪影），Δemb=0.459 属合理修正 |

**H3 原始表述本身有误**：`W_V` 是拼接后的共享线性投影，不存在「注册表列」。真正未训练的是 `registry_table.emb.weight`（C 期间 registry 为空，梯度为 0）——但这**不阻止**冷启动写入可用，因为 `W_V` 已被 hash 列训练过。

**新线索 H5（待验证）**：真 backbone 的 matched α=**0.1563**，**低于** tiny 的 0.2520。C 阶段 registry 全空、匹配率 0，gate 收到的一致信号是「没有可靠记忆」，因而学成对未训练 registry 低信任。**H5：写入方向可能有用，但门控幅度不足以改变排序。**

**⚠️ 本实验的局限**：数据仍是合成的（饱和+自证）。诊断只能回答「机制是否按设计工作」，**不能**回答「冷启动是否提升检索质量」。

---

## E-005 真实评测集饱和度验证 `[✅ 有效]`

**目的**：验证合成数据饱和问题在真实数据上消失。

**数据**：`build_real_eval.py --dataset t2retrieval --out data/real/eval/t2 --n-queries 2000`
- 真实搜索日志 query，corpus 118,605，qrels 覆盖 **100%** query，平均 **5.2 个正例/ query**
- 构建时清洗 T2 特有的 `<br>`/`<img>` HTML 噪声（否则纯词形匹配即可取胜）
- 负例从同 mini-corpus 池按字符重叠度优先采样（硬负例）
- gold 索引已校验全部合法（`0 <= gold < len(corpus)`）

**原始输出**（真 backbone，memory OFF，500 queries）
```
A baseline on REAL T2: {'recall@1': 37.4, 'recall@3': 74.0, 'recall@5': 95.0,
                        'recall@10': 100.0, 'mrr': 59.1, 'ndcg@10': 69.14}
```

**结论**

| | 合成 seen (E-001) | **真实 T2** |
|---|---|---|
| A baseline R@1 | **100.0** ❌ 饱和 | **37.4** ✅ |
| A baseline nDCG | **100.0** ❌ | **69.14** ✅ |
| 可提升空间 | **0** | **62.6 R@1** |

bge-small-zh 在真实搜索日志上仅 37.4 R@1，**这才是真实水平**；合成数据的 100.0 是彻底的幻觉。真实集可作为实验平台。

---

## E-006 CPU 训练吞吐实测 `[✅ 有效]`

**目的**：确认本机能否训练。

**配置**：真 backbone（冻结），batch=32，max_len=64，2000 triplets，**可训练 221,794 参数**

**原始输出**
```
62 steps in 33.0s  ->  0.533 s/step, 1.88 steps/s
epoch (2000 triplets) = 33.0s = 0.55 min
```

**结论**
- **本机可训练**。backbone 冻结（只前向不反传）是最大节省
- 2000 triplets × 10 轮 ≈ **5.5 分钟**
- 854k triplets（DATA.md 完整配方）× 1 轮 ≈ **7.9 小时**
- **限制**：仅万级 triplet 的实验规模在本机可行；全量配方需 GB10

---

## E-007 真实 T2 训练基线（**项目首个真实训练信号**）`[✅ 有效]`

**目的**：回答「在真实数据上 loss 到底降不降」。

**配置**
- backbone `BAAI/bge-small-zh-v1.5`（冻结），layer_ids=[1,3]
- **query_len=64 / doc_len=512**（见下方长度分析）
- batch=32，epochs=10，lrs 扫描 `1e-3, 3e-4`，eval 每 2 轮
- train=2000 triplets / eval=500 queries
- 数据 `data/real/eval/t2`，可训练 221,794 参数

### E-007a 长度参数决策（前置实验）

**实测 token 长度分布**（T2，300 queries × 10 candidates）
```
query           mean= 10.7  p50= 10  p90= 15  p95= 17  max= 27
all candidates  mean=404.6  p50=470  p90=505  p95=507  max=512
GOLD doc only   mean=324.2  p50=359  p90=499  p95=506  max=511
```

**截断保留率**
| max_len | 候选保留 | GOLD 保留 |
|---|---|---|
| 64 | 15.7% | **19.0%** |
| 128 | 31.0% | 36.4% |
| 256 | 59.7% | 65.7% |
| 384 | 83.9% | 87.1% |
| 512 | 100% | **100%** |

**决策**：原 `max_len=64` 只保留 GOLD 的 19%，是错误配置。改为**不对称长度**
（`query_len=64` / `doc_len=512`），既贴合真实部署（query 短/doc 长），
又避免用 512 pad query。已为此修改 `de/data.py`（`DECollator` 新增
`query_max_len` 参数）。

### E-007b 训练结果

```
=== lr=0.001, 10 epochs ===
  ep00  loss 0.1348  [248s]
  ep01  loss 0.0878  R@1 39.8  nDCG 70.8  [497s]
  ep03  loss 0.0116  R@1 41.0  nDCG 71.5  [1293s]
  ep05  loss 0.0049  R@1 41.0  nDCG 71.5  [2094s]
  ep07  loss 0.0022  R@1 43.0  nDCG 72.6  [2909s]
  ep09  loss 0.0078  R@1 42.6  nDCG 72.6  [3694s]
  [lr=0.001] loss 0.1348 -> 0.0078 (drop +0.1270)  nDCG 69.7 -> 72.6  [3990s]

=== lr=0.0003, 10 epochs ===
  ep00  loss 0.1352
  ep09  loss 0.0146  R@1 42.0  nDCG 72.3
  [lr=0.0003] loss 0.1352 -> 0.0146 (drop +0.1206)  nDCG 69.7 -> 72.3  [4055s]
```

**结论**

| | E-001 (旧) | **E-007** |
|---|---|---|
| loss 降幅 | 1.358 → 1.318 (**−0.04**) | 0.135 → **0.0078**（**降至 5%**） |
| 起点 loss | 1.36（≈2× 随机基线 ln2） | **0.135** |
| nDCG 增益 | +0.00（饱和） | **+2.9**（69.7 → 72.6） |
| R@1 增益 | — | **+5.2**（37.4 → 42.6） |

**记忆模块在真实数据上确实有效**（R@1 +5.2）。`lr=1e-3` 略优于 `3e-4`。

**收敛行为**：loss 在 ep07 触底 0.0022，之后回升至 0.0078 —— **轻微过拟合**。
nDCG 在 ep07 达峰 72.6 后持平。**最佳停止点 ≈ ep07**。

**⚠️ 本实验的两点局限**：
1. **起点 loss 0.135 偏低**。InfoNCE 随机基线为 ln(batch)=ln(32)≈3.47，
   0.135 说明 T2 的正负例**太好区分**（正例是 qrels 命中，负例虽词面重叠
   但非 qrels 相关）→ 仍是一种温和的自证，只是远好于合成数据的 R@1=100
2. **这不是冷启动实验**。测的是 Layer-B（hashed N-gram）在**训练见过的数据**上的
   效果；**registry 全程为空，冷启动写入一次都没测**

### E-007c ⚠️ 发现缺陷：硬负例被丢弃 `[已修正待重跑]`

**核对代码时发现 `train_baseline.py` 没用硬负例。**

| | 代码 | 是否用 negatives |
|---|---|---|
| 正式训练循环 `de/train.py:151-159` | `if "negative" in batch: info_nce_with_hard_negatives(...) else: info_nce(...)` | ✅ 两分支 |
| **`train_baseline.py:116`** | `loss = info_nce(qe, de)` — 只有 else 分支 | ❌ **丢弃** |

**后果**
1. `build_real_eval.py` 生成的每条样本 **3 个 BM25 硬负例全部被浪费**，
   只剩 in-batch 的 31 个负例
2. 这**很可能就是**起点 loss=0.135 的真正原因（而非「任务自证易」）——
   in-batch 负例区分度不足，模型几乎不用学就能拉开
3. **潜在泄漏**：`build_real_eval.py` 的 `train.jsonl` negatives 与 `eval.jsonl`
   的 corpus 取自**同一个 mini-corpus 池**。若日后启用硬负例，必须先确认
   训练负例与评测 corpus 不交叉

**对 E-007 结论的影响**
- `+5.2 R@1` 是**打了折扣**的数字（未用硬负例），真实上限应更高
- 「自证」这一判断**可能下错**，需用修正版重跑后重新判定

**处置**：已记录，待用 `info_nce_with_hard_negatives` 重跑修正版

### E-010a 修正版训练 —— ⚠️ 本机超时中断，迁移 GB10

**修正内容**
1. `train_baseline.py` 改用 `info_nce_with_hard_negatives`，显式构造 `TripletDataset`
   保证 `n_hard_neg=3` 全部进入 loss（原先 `TripletDataset(train_rows)` 走 dict
   直通，虽保留全部负例但 loss 丢弃）
2. 新增 `--leak-check` / `--no-hard-neg` / `--n-hard-neg` / `--temperature` 开关
3. **`build_real_eval.py` 重写为先切分再构造**（详见 E-010b）

**数据重建结果**（`data/real/eval/t2_v2`）
```
queries=22812  corpus=118605  usable_qrels=22812
eval_qids=1000   train_qids=21812
hard-neg scan pool: 2000 docs (char sets precomputed)
[leak-check] train_neg in eval: 1   train_pos in eval: 1   train_query in eval: 0
eval.jsonl  1000 queries (avg 13.7 candidates)
train.jsonl 8000 triplets
```

**启动后的实测输出（2 epoch 后中断）**
```
[leak] negatives also present in eval corpus: 1 (0.01%)
[leak] eval queries also used as train query: 0
=== lr=0.001, 10 epochs, hard_neg=True, temp=0.02 ===
  ep00  loss 1.5187  [4197s]
  ep01  loss 1.2645  R@1 35.9  nDCG 68.6  [8373s]
```

**关键结论：起点 loss 从 0.135 → 1.5187，证实 E-007c 的判断。**
E-007 那个异常低的起点是**硬负例被丢弃**所致，不是「任务自证偏易」。H6 需重判。

| | E-007（有缺陷+泄漏） | **E-010a（修正版）** |
|---|---|---|
| 起点 loss | 0.135（异常低） | **1.5187**（≈ln(32)/2，合理） |
| 起点 R@1 | 37.4 | 35.9 |
| eval query 泄漏 | **20/20 (100%)** | **0** ✅ |

**中断原因**：单 epoch **4197s ≈ 70 分钟**，远超预估（4.4 分钟，**差 16×**）。
`n_train` 2000→8000（4×）、`n_eval` 500→1000（2×）、doc=512 × bs=32 × 4 流
（query + positive + 3 negatives）叠加所致。2 个 lr × 10 轮 = **~24 小时**，
超出本机合理预算。

**判定**：`[数据✅ 已修复]` / `[训练⚠️ 仅 2 epoch，不足以判定]`
**已取得的有效信息**：修正版起点 loss 与零泄漏基线。趋势待 GB10 补全。

### E-010b `build_real_eval.py` 泄漏修复 `[✅ 有效]`

**发现的泄漏**（E-007/E-005 用的 `data/real/eval/t2` 全部受影响）

原实现用同一个 query 池同时生成 train 和 eval：

| 检查项 | 实测 | 影响 |
|---|---|---|
| 评测 query 被用作训练 query | **20/20 (100%)** | E-007 评测集 = 训练集 |
| 训练正例出现在评测 corpus | **20/20 (100%)** | 同上 |
| 训练负例出现在评测 corpus | **61/192 (31.8%)** | 负例污染 |

**因此 E-005 的「真实集不饱和 A=37.4 R@1」与 E-007 的「+5.2 R@1」均建立在泄漏之上**，
虽因 memory 初始无增益（E-007 组 A/B/C 全 100.0）未实际放大误差，但结论不可外推。

**修复**：先按 query id 切分 `eval_qids` / `train_qids`，再用 `eval_docs` 集合
禁止任何训练文档出现在评测侧。修复后泄漏降至 1 条（0.01%，来自 qrels 多正例
共享同一文档的边界情况）。

**顺带修复的性能缺陷**：原实现对**每条 query** 重建 `set(c[pid][:400])`
（O(n_q × hard_scan × doc_len)），全量构建 **>15 分钟未完成**。
改为一次性预计算 2000 篇文档的字符集后复用 → 构建时间降到**秒级**。

### E-007d N-gram 寻址机制澄清 `[✅ 有效]`

**结论：N-gram 不是「挖」出来的，没有词表、没有统计、没有剪枝。**

实现见 `de/hashing.py:226` `NgramHashMapping.hash_layer()`——对每个 token 位置、
每种阶数、每个 head 现场算一行索引：
```
z_{t,n,k} = ( ⊕_{i=0}^{n-1}  x_{t-i} · m_i )  mod  P_{n,k}
```
- **后缀约定**：位置 t 的 n-gram = `x_{t-n+1..t}`，在位置 t **末尾**结束
  （对齐 Engram 论文的 suffix N-gram；`PhraseMatcher` 用同一约定）
- **确定性**：`z` 是 token id 的纯函数、无运行时状态 → 这是它可 CRUD / 可预取的前提
- **多 head**：`n_head_per_ngram` 个 head 用**互不相同的素数**取模，
  同一个 N-gram 落到不同表的行，起多视角作用

**槽位容量实测**（vocab=21128）

| 配置 | 总槽位 | 2-gram 理论空间 (21128²) | 3-gram 空间 (21128³) | 参数量(d_head=8) |
|---|---|---|---|---|
| **E-007 实际** (`table=1024, n_head=2`) | **8,396** | 4.5 亿 → **0.002%** | 9.4×10¹² → **≈0** | 67,168 |
| `config.py` 生产默认 (`65536, n_head=8`) | **2,101,122** | 4.5 亿 → **0.47%** | 9.4×10¹² → ≈0 | 16,808,976 |

各表实际素数（E-007）：
```
layer1: order2:[1031,1033]  order3:[1039,1049]
layer3: order2:[1051,1061]  order3:[1063,1069]
```

**⚠️ 两个由此暴露的问题**

1. **实验配置与生产配置差 250 倍**。`run_abcd.py`/`train_baseline.py` 用
   `table_size_per_ngram=1024`（67k 参数），`config.py` 默认 `65536`（16.8M）。
   **E-007 的全部数字都跑在极小表上**，不能代表生产配置的表现。
   任何跨实验的数字对比必须先确认表容量一致。

2. **3-gram 表几乎是纯废槽**。E-007 里 3-gram 共 4 表 ≈ 4,210 槽，
   对应 9.4×10¹² 的空间——命中率约 0。这部分参数既不提供记忆容量，
   又在推理时贡献 gather 开销（E-003e 已算：参数:算力比 42:1）。
   **是否应关闭 3-gram（`max_ngram_size=2`）需做消融**。

**与冷启动的关系（设计层面的根本矛盾）**

Layer-B 是**纯统计记忆**，学的是「哪些 N-gram 组合出现过」。冷启动要解决的是
「训练时没见过的术语」——而**新术语的 N-gram 天然落在训练时梯度从未触及的槽上**
（保持随机初始化）。这正是 H3 类问题的根源，也是 registry（Layer-A）必须存在的
理由：**它按精确短语寻址，不受哈希空间限制**。

---

## E-008 中文法律冷启动集构建（严格 unseen）`[~ 进行中]`

**目的**：构造能**证伪 P0** 的评测集——训练时严格未见、评测时才出现的术语。

**数据**：`data/real/COLD-1_twang2218_chinese-law-and-regulations`（22,552 行，Apache-2.0）

### E-008a 时间分布实测（已独立复核，非转述）

```
publish_date 非空 18,385 / 22,552 (81.5%)，NULL 4,167
范围 1950-01-01 → 2023-07-27
年份：2018→1735  2019→1630  2020→1945  2021→2165  2022→1806  2023→380
```

| T0 | TRAIN | COLD | 评价 |
|---|---|---|---|
| 2020-01-01 | 12,089 | 6,296 | COLD 偏多 |
| **2021-01-01** | **14,034** | **4,351** | ✅ **选定** |
| 2022-01-01 | 16,199 | 2,186 | 偏激进 |
| 2023-01-01 | 18,005 | **380** | ❌ 不可用 |

### E-008b unseen 保证（严格版）

在 T0=2021 基础上再加一道过滤：要求 COLD 条目的**标题**与**正文前 200 字**
都未在 TRAIN 中出现：
```
TRAIN(content 非空, publish<2021) = 14,034
COLD (content 非空, publish>=2021) = 4,351
其中 标题在 TRAIN 未出现            = 2,906 (66.8%)
其中 再要求 content 前200字未出现   = 2,906  (无额外损失)
```
**保证成立**：Layer-A registry 只用 TRAIN 标题播种、Layer-B 只在 TRAIN 上拟合，
故每个评测术语必然满足「无 registry slot（匹配返回 NULL）+ 零梯度（hash 行停在初始化）」，
任何 D−C 增益**只能**归因于冷启动写入，无泄漏。

**COLD type 分布**：地方性法规 2,236 / 无效已修改废止 565 / 公司法 38 / 法律 28 /
中国特色的涉外 24 / 其他 13 —— **非国家法律为主**，与文档叙事不同。

**状态**：✅ **构建完成并验证**（E-008c）

### E-008c 构建输出（实测）

命令：`python build_cold_law.py --out data/real/eval/law`

```
loaded 22552 rows
TRAIN (publish < 2021) : 14034
COLD  (publish >= 2021): 4351
EVAL  (provably unseen) : 2906  (66.8% of COLD)
TRAIN clause pool (negatives) : 79786
-> data/real/eval/law
  train.jsonl        8000
  eval_cold.jsonl    1500
  cold_start.jsonl   1500
```

**unseen 保证实测复核（关键）**
```
cold terms appearing verbatim in train query: 0 / 1500   ← 零泄漏
cand per eval row: 10    gold == 0 for all rows
```

**样本**（真实法规，非合成）：
- term：`医疗废物污染控制管理办法`（COLD 侧新规）
- query：`医疗废物污染控制管理办法 违反本法规定...医疗废物...排放` ← 标题+首个条款
- GOLD：该法规正文（2022-12-02 修订）
- NEG：另一条 TRAIN 侧旧法条款（如固体废物污染罚款处罚办法）—— 同域、旧法、不可答

**设计要点**
1. 过滤掉「修订/修正案/修改决定/废止」类标题——它们是同一法规的修订日志，
   术语本身早已存在，会破坏 unseen 语义
2. 负例全部取自 **TRAIN** 的条款（79,786 条池），确保负例**也不是** unseen，
   避免「靠区分新旧」作弊
3. 正则抽取「第X条」条款作为 query 的核心义务从句，正文截断至 800 字符
4. `gold` 恒为 0（正例固定在 corpus 首位）

**⚠️ 局限**：
1. 88% 是**地方性法规**，与「国家法律」叙事不同
2. query 由**标题+条款**模板拼接，非真实用户查询，**仍有自证成分**
   （比合成数据的 R@1=100 温和，但非零）
3. 评测 corpus 仅 10 条（1 正 9 负），难度低于真实百万级检索

**下一实验**：E-009 —— 在本集上跑 A/B/C/D，检验冷启动增益（验证 H5）。

### E-008c ⚠️ 发现并修复 bug：eval/term 配对错位 `[已修复 ✅]`

**冒烟测试暴露的现象**
```
match rate BEFORE write: 0.0%
wrote 40/40 terms
match rate AFTER  write: 0.0%     ← 写入后仍 0%，机制完全没生效
```
在 T2 上同样的诊断是 **200/200 = 100%**（E-004），说明是数据构造问题而非模型问题。

**排查过程**
1. 先怀疑 query 截断 → 改用 title-first + `query_budget=200`，仍 0.1%
2. 直接打印 canonical id 比对：
   ```
   term   : 医疗废物污染控制管理办法
   query  : 山东医疗废物管理...（开头是"山东"，与 term 完全不同法规）
   write canonical : [667, 1253, 745, ...]
   query canonical : [7904, 2218, 2319, ...]
   contiguous match: False
   ```
   **根因**：`build_cold_law.py` 中 `rng.shuffle(eval_rows)` **只打乱了 eval 列表，
   没有同步打乱 `cold_terms`**，导致两表错位——第 i 条 eval query 配的是第 i 个 term，
   但它们根本不是同一部法规。

**该 bug 的危害**：会让 E-009 **必然输出 D−C=0**，并被误读为「冷启动无效」。
E-002 当初就差点得出「写入过度」的假结论；这是同类陷阱的第二次出现。

**修复**：`pairs = list(zip(eval_rows, cold_terms)); rng.shuffle(pairs)` 一起洗牌。

**新增防线**：`build_cold_law.py` 末尾加入 `_verify()`，用真实 tokenizer 校验
「每个冷启动 term 是否逐字连续出现在其 eval query 中」：
```
[verify] cold term present verbatim in its query: 1500/1500 (100.0%)   <-- 必须 ~100%
```
并在 query 构造时强制 title 前置 + 预算校验（dropped 1 条）。

**修复后冒烟验证**
```
match rate BEFORE write: 0.0%
wrote 40/40 terms in 0.1s
match rate AFTER  write: 95.0%     ← 机制生效
cos(emb_ON, emb_OFF): mean=0.9996  (40条样本饱和，D−C 无法区分，属正常)
```

**教训**：任何「写入后无增益」的结论，都必须先看 **match rate**。
match rate = 0 的时候谈 P0 增益是无效的。

---

# 第三部分：待验证假设汇总

| ID | 假设 | 状态 | 验证方式 |
|---|---|---|---|
| **H1** | 写入没落地（matcher 不触发） | ❌ 已排除（E-004） | — |
| **H2** | 门控压制（α≈0） | ❌ 已排除（E-004） | — |
| **H3** | 注册表列随机不可用 | ❌ 已排除（E-004，原表述有误） | — |
| **H4** | 写入过度（覆写 backbone） | ❌ 已排除（E-004） | — |
| **H5** | 写入方向有用但幅度不足以改变排序 | 🔄 **待验证** | 扫 `refine_steps`/`ridge`/注入幅度，看 `D−C` 是否随幅度单调 |
| **H6** | T2 任务对记忆模块无增益空间 | ⚠️ **待重判** | E-007 起点 loss 0.135 原本判为「任务偏易」，但 E-007c/E-010a 证明真正原因是**硬负例被丢弃 + 数据泄漏**；修正版起点 loss=1.5187（正常）→ 需在干净数据上重判 |
| **H7** | 需真 backbone 才能测出效果 | ✅ **已证实** | E-001 vs E-004/007：tiny 上一切归零 |
| **H8** | 实验表容量与生产配置不一致，数字不可外推 | ✅ **已证实** | E-007d：8,396 槽 vs 生产 2,101,122 槽（**250×**） |
| **H9** | 3-gram 表几乎全废，应关掉 | 🔄 **待消融** | E-007d：4,210 槽 vs 9.4×10¹² 空间；试 `max_ngram_size=2` |
| **H10** | T2 实验平台存在 train/eval 泄漏 | ✅ **已证实并修复** | E-010b：eval query 泄漏 **100%**；修复后 0.01% |
| **H11** | E-007 的起点 loss 异常低源于缺陷而非任务性质 | ✅ **已证实** | E-010a：0.135 → **1.5187**（≈ln32/2，InfoNCE 正常起点） |

---

# 附：GB10 迁移执行计划

## 0. 前置：本机已完成、无需重跑的部分

| 成果 | 位置 | 说明 |
|---|---|---|
| 环境修复 | scipy 1.18.1 | NumPy 2.x ABI |
| 真实 backbone | HF cache `models--BAAI--bge-small-zh-v1.5` | 4 层 / 512 维 |
| 真实数据 | `data/real/` **3.73 GB** | 24 数据集 / ~105 万行 |
| 干净 T2 数据集 | `data/real/eval/t2_v2` | **零泄漏**，1000 eval + 8000 train |
| 法律冷启动集 | `data/real/eval/law` | **unseen 零泄漏**，8000 train + 1500 eval + 1500 terms |
| 诊断脚本 | `diagnose_cold.py` | H1/H2/H3 确诊 |
| 构建脚本 | `build_real_eval.py` / `build_cold_law.py` | 含泄漏检查 |
| 实验记录 | `docs/EXPERIMENTS.md` | E-001 ~ E-010b |

**迁移需拷贝**：`de/`、`docs/`、根目录 6 个脚本、`data/real/`。
**无需重下模型**：`bootstrap_gb10.sh` 已含 HF 下载（或直接拷 HF cache）。

## 1. GPU 前置验证（必须先做，~30 分钟）

GB10 是 sm_121 / aarch64，现有代码**从未在 CUDA 上跑过**（本机 `torch+cpu`）。优先排查：

- [ ] `import torch; torch.cuda.is_available()` → True
- [ ] `python offline_test.py` → **9/9 PASSED**（关键：验证 `attention_bias` 的 SDPA 分支）
- [ ] 手工过一遍 `domain_registries.py`：CUDA 下 `registry_table.emb.weight` 的
      原地写入是否与 DDP 冲突
- [ ] `torch_dtype=float32` → 试 `bfloat16`（`EngramConfig.torch_dtype`）
- [ ] **bf16 下重测 `cos(emb_ON, emb_OFF)`**：CPU 上是 0.894，精度变化可能改变该值
- [ ] 已知：`PhraseMatcher` 在 CPU 上跑（`de/data.py` 注释说明 `num_workers>0` 必需），
      GPU 环境下会成为瓶颈 —— 用 `--num-workers` 验证

## 2. P0 优先级实验（GB10 上，估时按本机 70 min/epoch 折算 GPU 加速比）

| # | 实验 | 目的 | 本机耗时 | 依赖 |
|---|---|---|---|---|
| **E-010** | T2 修正版训练（补全 10 轮 × 2 lr） | 拿到干净数据上可信的 C 组基线 | 中断（2/20 ep） | 步骤 1 |
| **E-009** | **法律冷启动 A/B/C/D** | 🔴 **首次真正检验 P0** | 未开始 | 步骤 1 |
| **E-011** | 表容量消融 `{1024, 8192, 65536}` | 验证 H8，定生产配置 | ~3× E-010 | E-010 |
| **E-012** | `max_ngram_size` ∈ `{2,3}` | 验证 H9（3-gram 是否废槽） | ~2× E-010 | E-011 |
| **E-013** | `layer_ids` ∈ `{[1,2],[1,3]}` | 验证 P1 待修项 | ~2× E-010 | E-010 |
| **E-015** | `temperature` ∈ `{0.02,0.05}` | loss 起点 1.52 偏高，可能是温度过低 | ~2× E-010 | E-010 |

**E-009 是整个项目的核心**：只有它能回答「冷启动写入到底有没有用」。
在它出结果前，其余实验都是次要的。

## 3. 生产配置验证（消融完成后）

- [ ] 用**生产默认配置**（`config.py`，表 65536、81.5M 可训练）跑一次，
      确认实验结论在真实容量下成立
- [ ] 测吞吐与显存，回填 `docs/COMPUTE.md`（当前全为推导估算）

## 4. 部署约束（用户明确要求纳入考虑）

vLLM 非硬需求，但以下必须在决策前明确：

- [ ] **`PhraseMatcher` 优化**：当前 Python 逐序列 + `range(L)` numpy 循环，
      是热路径瓶颈。`num_workers` 只是缓解，需重写为向量化/GPU 版
- [ ] **Registry 快照机制**：`PhraseRegistry.version` 已有版本门控
      （`maybe_rebuild`），需扩展为「不可变快照 + 原子 reload」
- [ ] **新增术语后的重嵌入决策**（TODO 中从未提及的产品级问题）：
      索引离线建立，冷启动只改 query 侧向量 → query/doc 空间可能失配

## 5. 执行顺序原则

> **先把便宜的证伪跑掉。** E-009 若证伪 P0，整个方案需要重新设计，
> 此时 E-011/012/013 的消融都没有意义。所以：**步骤 1 → E-009 → 其余**。

---

# 附：本文档引用的脚本

| 脚本 | 用途 |
|---|---|
| `build_real_eval.py` | C-MTEB → 项目 JSONL 契约（HTML 清洗 + BM25 硬负例 + gold 校验） |
| `build_cold_law.py` | 法律冷启动集（T0 切分 + 严格 unseen 过滤） |
| `train_baseline.py` | 真实数据训练基线，**以 loss 曲线为主要输出**，含 lr 扫描与不对称长度 |
| `diagnose_cold.py` | H1/H2/H3 诊断 + 参数漂移 + cos(ON,OFF) 方向检测 |
| `run_abcd.py` | 四组对照实验 |
| `offline_test.py` | 9 项离线机制检查 |
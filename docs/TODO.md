# TODO — Domain Engram

> 创建于 2026-10-11。**本轮推翻了大量历史结论**，旧数字已作废，勿再引用。
>
> 📌 **2026-10-11 收尾：已全部推送至 `origin/main`（commit `0db5d7a`），工作树干净。**
> 下一步在 NVIDIA 卡上执行，顺序见 **[GB10_RUNBOOK.md](GB10_RUNBOOK.md)**。
> 所有实验数字以 **[EXPERIMENTS.md](EXPERIMENTS.md)** 为唯一权威出处。
>
> 图例：`[ ]` 未开始 · `[~]` 进行中 · `[x]` 完成 · `[!]` 被环境阻塞 · `[✗]` 已作废

---

## 🔴 下一步（GPU 上执行，按序）

1. `python offline_test.py` → 必须 **9/9**（验证 CUDA/SDPA 分支，本项目唯一未在 GPU 跑过的路径）
2. `python download_data.py --tier p0` → 复现数据（0.65 GB，约 2 分钟）
3. **`python run_cold_start.py --data data/real/eval/law --epochs 10 --lr 1e-3 --device cuda`**
   → **E-009，冷启动有效性。这是第一次能真正证伪 P0 的运行。**
   判据：`D − C (unseen nDCG) ≥ +2`，且 `match rate ≈ 100%`、`cos(ON,OFF) ≈ 0.85~0.95`
4. 结果写入 `docs/EXPERIMENTS.md` 编号 E-009（含环境/配置/原始输出/判定）
5. 之后才做消融 E-010~E-015（见 EXPERIMENTS.md 附录）

---

## 当前状态快照（2026-10-11）

**本轮突破**：网络打通（本机出网代理，约 5 MB/s；**代理地址为本机环境细节，未写入仓库**）。
已下载 **3.73 GB / 24 个真实数据集** → `data/real/`（gitignored，
用 `python download_data.py --tier p0` 复现，约 0.65 GB）。
真 backbone `BAAI/bge-small-zh-v1.5`（**4 层 / hidden 512 / 8 heads / vocab 21128**）已下载并跑通。

**⚠️ 最重要的结论：此前所有 A/B/C/D 数字都来自一个随机初始化的 tiny
backbone（hidden=64, 1.56M, 未训练），全部作废。** 详见「已作废结论」。

**下一个动作**：

```
python build_real_eval.py --dataset t2retrieval --out data/real/eval/t2
python run_abcd.py --data data/real/eval/t2 --epochs 2 \
       --backbone BAAI/bge-small-zh-v1.5
```

---

## 已作废结论 `[✗]`（勿引用）

历史 TODO 里的这组数字来自 `build_tiny_backbone()`（随机权重）：

| 对比 | 旧（tiny） | 新（真 backbone） | 判定 |
|---|---|---|---|
| C − A（seen）nDCG | +14.30 | **+0.00** | ✗ 假象 |
| C − A（unseen）nDCG | +5.81 | +0.61 | ✗ 大幅缩水 |
| D − C（unseen）nDCG | +0.06 | **−0.87** | ✗ 且无统计意义 |
| 冷启动写入 cos(ON,OFF) | −0.044（"覆写"） | **+0.894**（合理修正） | ✗ 假设被推翻 |

### 为什么全部作废

1. **tiny backbone 是随机权重**（`run_abcd.py` 的 `--backbone` 默认 `None` →
   `build_tiny_backbone()`，hidden=64、1.56M）。随机网络没有语义结构，
   记忆是刚需 → `C−A=+14.30` 纯粹是小模型太弱的假象。
2. **合成 seen 任务已饱和**。真 backbone 上 A/B/C 的
   R@1 / nDCG **全部 = 100.0**，记忆机制没有任何可提升空间 → 恒等于 0 差异。
3. **合成数据自证**。query 与正例共享同一术语、干扰项只差术语名，
   任务恰好匹配记忆表机制；held-out 术语 `新发疾病N` 之间仅编号不同，
   bge 纯靠词形匹配就有 79.5 R@1，**根本用不上记忆**。

---

## P0 — 冷启动有效性 `[!]`（当前唯一拦路的科学问题）

**状态：实验平台已就绪，但 P0 在合成数据上无法被证伪。**
必须先换成真实评测集（见 P0-1）才能继续。

三个原始假设**已全部排除**（真 backbone 实测）：

| 假设 | 排除依据 |
|---|---|
| **H1 写入没落地** | 匹配率 写入前 `0/200` → 写入后 `200/200` |
| **H2 门控压制** | matched 位置 gate α `mean=0.156`（阈值 0.05），门是开的 |
| **H3 值不可用** | 22/23 个 engram/registry 参数在 C 阶段收到梯度 |
| ~~H4 写入过度~~ | cos(ON,OFF)=**0.894**，Δemb=0.459，是合理修正非覆写 |

**剩余线索（下一个要查的）**：C 阶段 registry 全空、匹配率 0，
gate 收到的一致信号是「没有可靠记忆」，因此学成对未训练 registry 低信任
（真 backbone α=0.156 < tiny 的 0.252）。**H5：写入方向可能有用，
但幅度/门控不足以改变排序**。

- [ ] **`[ ]` P0-1 构建真实评测集** ← **当前第一步**
  - [ ] `build_real_eval.py`：`T2Retrieval` + qrels → 项目 JSONL 契约
  - [ ] 确认 bge 在该集上**不饱和**（这是能测的前提）
  - [ ] 注意：C-MTEB qrels 字段是 `qid, pid, score`，
        **不是** BeIR/LeCaRD 的 `{query-id, corpus-id, score}`，loader 要分开
- [ ] **`[ ]` P0-2 真实集上重跑 A/B/C/D**，目标 `D − C ≥ +2 nDCG`
- [ ] **`[ ]` P0-3 验证 H5**：扫 `refine_steps` / `ridge` / 注入幅度，
      看 `D − C` 是否随幅度单调变化

---

## P0.5 — 中文法律冷启动集（真实 unseen 保证）`[ ]`

**数据已到位**：`data/real/COLD-1_twang2218_chinese-law-and-regulations/`
（22,552 行，Apache-2.0，字段实测含 `publish_date`/`type`/`status`/`title`/`content`）。

**实测时间分布（已独立复核，非转述）**：

- `publish_date` 非空 **18,385 / 22,552（81.5%）**，NULL 4,167 行
- 范围 `1950-01-01` → `2023-07-27`
- 近 15 年：2018→1735, 2019→1630, 2020→1945, 2021→2165, 2022→1806, 2023→380

| T0 | TRAIN | COLD | 评价 |
|---|---|---|---|
| 2020-01-01 | 12,089 | 6,296 | COLD 偏多 |
| **2021-01-01** | **14,034** | **4,351** | ✅ **推荐** |
| 2022-01-01 | 16,199 | 2,186 | 偏激进 |
| 2023-01-01 | 18,005 | **380** | ❌ 不可用 |

**已验证的新术语供给**：publish≥2020 的 6,296 条中，
**4,200 条标题在 TRAIN(<2020) 中完全未出现（66.7%）**。

⚠️ **数据构成注意**：88% 是**地方性法规**（19,773），国家法律仅 429 条。
`docs/DATA.md` 的「中文法规」叙事与实测不符，采样要按 `type` 分层。

- [ ] **`[ ]` 构建** train/eval_cold JSONL，T0=2021-01-01
- [ ] **`[ ]` Registry 术语只从 TRAIN 抽取**（保证 COLD 术语真的 unseen）

---

## P1 — 设计缺陷

- [ ] **`[ ]` 修正注入点 `layer_ids=[1,3]` → `[1,2]`**
  实测 backbone **只有 4 层**（0..3），`[1,3]` 把第二个注入点压在**最后一层**，
  注入后直接进 pooling，零精炼 → 违反机制假设（原论文消融：层 2 最优，
  注入后需留 block 精炼）。改为 `[1,2]` 保证注入后至少留 1 层。
- [x] **`[x]` 修 `eval_rows` gold 索引**（全局 → 行内局部）
- [x] **`[x]` 修 `make_synthetic_data.py` 闭包复用最后一次 gold**
- [x] **`[x]` `DECollator._field()` 同时支持 `Triplet` 与 `dict`**
- [ ] **`[ ]` `_built_version` 缓存失效**：`registry.delete()` 后若忘调
  `sync_matcher()`，匹配结果仍是陈旧的。应在 delete/update 内自动失效。
- [ ] **`[ ]` 参数规模与部署成本**（实测）
  可训练 **80.2 M**，其中 **67.2 M（82%）是 hashed N-gram 查表**。
  但每 token Engram 只增 **1.58 M MACs**、权重却达 **268 MB**（fp32）
  → 参数:算力比 **42:1**，**纯访存瓶颈**，非计算瓶颈。
  这 67 M 槽在 21128 中文词表下**绝大多数永远为空**（3-gram 表废槽最多）。
  降本候选：`table_size_per_ngram` 65536→16384、`n_head_per_ngram` 8→4、表存 bf16。
  **需消融检索质量影响后再定。**

---

## P2 — 推理 / 部署约束（vLLM 非硬需求，但这些要满足）

**明确结论：当前架构无法直接用 vLLM**，三条硬冲突：
1. `de/model.py:224` 手工遍历 blocks 注入记忆，绕过 HF `forward()`；
   vLLM 替换整个建模代码，永不执行这段循环
2. `registry_slots (B,T,R)` 是**非 token 的第三个 forward 输入**（`de/model.py:208`），
   vLLM 输入管线只产出 token ids
3. `RegistryTable` 是**推理期可变权重**（`de/registry.py:163` `reset_slot`），
   多 worker × TP rank 各持私货，写入需跨进程广播控制面

**且引擎选型本就错配**：vLLM 是自回归解码引擎（paged KV + continuous batching）；
bi-encoder 每条文本产一个稠密向量，无 KV 复用、无生成。
对应的是 SentenceTransformers / TEI / ONNX Runtime + FAISS。

- [ ] **`[ ]` 改写成 HF `PreTrainedModel` 子类**（让 ST/TEI 可加载）
- [ ] **`[ ]` 优化 CPU 侧 `PhraseMatcher`**（`de/registry.py:315` Python 逐序列
      循环 + `_rolling_hash` 对每长度跑 `range(L)` 次 numpy）——**这才是热路径瓶颈**
- [ ] **`[ ]` Registry 用「不可变快照 + 广播 + 原子 reload」**，不要请求热写
- [ ] **`[ ]` 定义新增术语后是否重嵌入命中文档**：
      索引是离线建的，冷启动只改 query 侧向量 → query/doc 空间可能失配。
      **这是产品级决策，TODO 里此前从未提及。**

---

## P3 — GB10 实机

- [x] **`[x]` `bootstrap_gb10.sh`**（sm_121 / aarch64，禁 flash-attn，仅 SDPA）
- [x] **`[x]` `run_gb10.py`** 带 `--preset` / `--stage`
- [x] **`[x]` `docs/COMPUTE.md`**
- [ ] **`[ ]` 实机首跑**，并**实测校准吞吐**
- [ ] **`[ ]` 回填实测吞吐** ⚠️ 当前所有吞吐数字均为推导估算，从未实测

---

## P4 — 文档修订 `[ ]`

`docs/DATA.md` 与实测有三处偏差（subagent 发现，已复核）：

1. **`mteb/LeCaRDv2` 规模严重高估**：文档写 800 queries / 55,192 cases，
   实测仅 **159 queries / 3,795 cases**
2. **`C-MTEB/*-qrels` 字段名是 `qid, pid, score`**，不是文档写的
   `{query-id, corpus-id, score}`（那是 BeIR / LeCaRD 的格式）
3. **`T2Retrieval` queries 实测 22,812**（文档 24,832）、
   **`DuRetrieval` 实测 2,000**（文档 4,000）；
   `Amazon raw_meta_All_Beauty` 实测 **112,590 行 / 59.6 MB**（文档 807 MB / 801,446 行），
   且**仓库无 `raw_review_All_Beauty` parquet**，review 走
   `raw/review_categories/All_Beauty.jsonl`
4. **`twang2218` 是 88% 地方性法规**，非「国家法律」

---

## 已知环境阻塞 `[!]`

| 阻塞 | 影响 | 状态 |
|---|---|---|
| ~~HF 下载被代理拦截~~ | — | ✅ **已解决**（需出网代理；代理地址不写入仓库） |
| ~~无真实数据集~~ | — | ✅ 已下 3.73 GB / 24 数据集 |
| 本机无 NVIDIA GPU（i7-8550U / 24GB RAM / 8 torch 线程） | 吞吐全为估算 | 待 GB10 |
| `torch 2.10.0+cpu` | 无法验证 bf16 / kernel 路径 | 代码仅标准算子 |
| `gensim 4.3.3` / `graspologic 3.4.1` 与 NumPy 2.x / SciPy 1.18 冲突 | 用到相关库时需隔离环境 | 已记录 |

---

## 验证状态

```
offline_test.py       9/9 PASSED
diagnose_cold.py      真 backbone 上跑通，H1/H2/H3 全排除
run_abcd.py (真bge)   跑通，但合成数据饱和，无法证伪 P0
真实数据              3.73 GB / 24 数据集 / ~105 万行 parquet
publish_date 统计     已独立复核（22552 行、年份分布、T0 切分全部核对）
git                   有未提交改动（de/testing.py, diagnose_cold.py）
```

---

## 建议执行顺序

1. **构建真实评测集**（T2Retrieval + qrels），确认 bge 不饱和
2. 真实集上重跑 A/B/C/D → 拿到可信的 `D − C`
3. 验证 H5（扫写入幅度）
4. 修 P0 + 同时改 `layer_ids=[1,2]`
5. 构建中文法律冷启动集（T0=2021-01-01）
6. 再上 GB10

> **原则**：先把便宜的证伪跑掉，再上 GB10。
> 本轮教训：*跑在随机权重上的实验，其数字没有任何外推价值*。
> 合成数据 saturated 之后，任何 P0 数字都不可信 —— 换真实数据是唯一的出路。
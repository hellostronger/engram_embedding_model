# GB10_RUNBOOK.md — 上机运行手册（NVIDIA GPU）

> 面向「拿到一张英伟达卡就开跑」。按顺序执行，**不要跳步**。
> 配套文档：[EXPERIMENTS.md](EXPERIMENTS.md)（结论唯一出处）·
> [TODO.md](TODO.md)（任务状态）

---

## 0. 一句话现状

本机（CPU）已完成：环境打通、3.85 GB 真实数据、干净无泄漏数据集、
冷启动集、诊断工具、10 组实验记录。**未完成**：GPU 上任何训练、
以及核心实验 **E-009（冷启动是否有效）**。

> 🚨 **最重要的一件事**：先跑 **E-009**，不是 E-010。
> E-009 若证伪冷启动，整个方案要重新设计，后面的消融都失去意义。

---

## 1. 环境准备

### 1.1 系统依赖（sm_121 / aarch64，禁 flash-attn）

```bash
bash bootstrap_gb10.sh
```
该脚本已处理：禁用 flash-attn、仅走 SDPA、装 CPU-only flash-attn 的规避逻辑。

### 1.2 Python 依赖

```bash
pip install -r requirements.txt
```

**⚠️ 已知版本冲突**（本机实测）：`gensim` / `graspologic` 需要 `numpy<2`，
而 transformers 4.57 + torch 2.10 要 `numpy>=2`。本项目**不使用**这两个库，
若报冲突可忽略或卸载它们。

```bash
python -c "import torch, transformers, scipy, numpy; print(torch.__version__, torch.cuda.is_available())"
```
**必须看到 `True`**，否则停下排查。

### 1.3 代码与数据

```bash
git clone https://github.com/hellostronger/engram_embedding_model.git
cd engram_embedding_model
```

数据不在 git 里（`data/` 已忽略，3.85 GB），用脚本复现：

```bash
# 只下实验必需的（~0.65 GB，约 2 分钟）
python download_data.py --tier p0

# 需要英文金融/法律时再加
python download_data.py --tier p0 p1

# 全量（含 SEC 2.2GB + Amazon 386MB，约 5 分钟）
python download_data.py --tier all

# 校验已有数据
python download_data.py --tier p0 --verify
```

**网络**：脚本不硬编码任何代理。有直连出口即可；若你的机器需要代理，
自行 `export HTTP_PROXY=... HTTPS_PROXY=...` 后再运行。

---

## 2. 第 0 关：机制自检（必做，~5 分钟）

```bash
python offline_test.py
```
**必须 9/9 PASSED。** 这验证 `attention_bias` 在 CUDA/SDPA 下的分支正确性——
**这是本项目唯一从未在 GPU 上验证过的代码路径**。

失败就停下，先修，不要继续。

---

## 3. 第 1 关：GPU 前置验证（~30 分钟）

代码从未在 CUDA 上跑过。以下几项**先于任何训练**确认：

```bash
python -m pytest tests/ -q 2>/dev/null || python offline_test.py
```

**必须逐条确认：**

- [ ] `torch.cuda.is_available()` → `True`
- [ ] `offline_test.py` → **9/9**
- [ ] `EngramConfig.torch_dtype="bfloat16"` 下 forward 不报错
      （CPU 上只跑过 fp32，bf16 的 RMSNorm/attention 路径未验证）
- [ ] **bf16 下重测冷启动写入方向**：
      ```bash
      python diagnose_cold.py --data data/real/eval/law --epochs 2 --backbone BAAI/bge-small-zh-v1.5
      ```
      CPU/fp32 下 `cos(emb_ON, emb_OFF) = 0.894`（E-004）。
      **精度变化若让它偏离到 <0.5，说明 bf16 不足以支撑该机制。**
- [ ] `PhraseMatcher` 在 GPU 下的瓶颈确认：`de/data.py` 注释指出它跑在 CPU，
      `num_workers=0` 会让它与 GPU 串行。测 `--num-workers 4` 的加速比。
      **这直接决定吞吐上限，是热路径瓶颈。**
- [ ] CUDA 下 `registry_table.emb.weight` 的原地写入是否与梯度更新冲突
      （冷启动写入 + 训练共用一张表）

---

## 4. 第 2 关：核心实验 E-009 —— 冷启动有效性 🔴

**这是整个项目的核心卖点，目前完全未被证实。**

数据集已备好（`data/real/eval/law`）：T0=2021 严格时间切分，
1500 个冷启动术语**零泄漏**（E-008 实测）。

```bash
# A/B/C/D 四组对照；D = C + 冷启动写入，不重训
python run_cold_start.py \
    --data data/real/eval/law \
    --epochs 10 --lr 1e-3 \
    --backbone BAAI/bge-small-zh-v1.5 --device cuda
```

**判据**

| 指标 | 含义 |
|---|---|
| `D − C (unseen nDCG)` | **核心**。TODO 原定目标 ≥ +2 |
| `D − C (unseen R@1)` | 同上，R@1 口径 |
| `D − C (seen)` | 若**下降**，说明冷启动污染了已学术语 → 严重问题 |
| `cos(emb_ON, emb_OFF)` | 写入是否为合理修正（期望 ~0.85~0.95，非 0 或负） |
| 匹配触发率 | 写入前应 0%，写入后应 ~100%（E-004 验证过） |

**若 `D − C ≈ 0`**，按此顺序排查（E-004 已排除 H1/H2/H3/H4）：
1. **H5**：门控幅度不足 → 扫 `--refine-steps {0,5,20,50}`，
   看 `D−C` 是否随幅度单调变化
2. registry 术语是否真的命中（打印 match 触发率）
3. 注入幅度：`solve_slot_closed_form` 的 `ridge` 参数扫描

**完整记录要求**：结果写入 `docs/EXPERIMENTS.md`，编号 **E-009**，
含环境/配置/原始输出/结论/判定。禁止只在对话里报数。

---

## 5. 第 3 关：补全与消融（E-009 出结果后再做）

| 编号 | 实验 | 命令要点 | 目的 |
|---|---|---|---|
| **E-010** | 补全修正版训练 | `train_baseline.py --data data/real/eval/t2_v2 --n-train 8000` | 本机只跑到 2/20 epoch |
| **E-011** | 表容量消融 | `--table-size {1024,8192,65536}` | 验证 H8（实验/生产差 250×） |
| **E-012** | N-gram 阶数 | `--max-ngram 2` vs `3` | 验证 H9（3-gram 几乎全废槽） |
| **E-013** | 注入点 | `--layers 1 2` vs `1 3` | 修 P1：`[1,3]` 把第二注入点压在**最后一层**，零精炼 |
| **E-015** | 温度 | `--temperature {0.02,0.05}` | E-010a 起点 loss 1.52 偏高 |

**GPU 上单轮预估**：本机 CPU 为 70 min/epoch（8000 triplets, doc=512, bs=32），
A100/GB10 级通常 **10–20× 加速 → 3–7 min/epoch**。
先跑 1 个 epoch 实测，再据实调整规模。

---

## 6. 生产配置验证（消融完成后）

实验用的是**缩小配置**（表 1024 → 8,396 槽 / 67k 参数）；
`de/config.py` 默认是**生产配置**（表 65536 → 2,101,122 槽 / 16.8M 参数），
差 **250 倍**（E-003e / H8）。

```bash
python train_baseline.py --config de/config.py --data data/real/eval/t2_v2 ...
```
**必须确认实验结论在真实容量下依然成立**，否则消融结果不可外推。

显存估算（fp32）：backbone 96 MB + Engram 表 **268 MB** + registry 51 MB，
**总计不足 0.5 GB** —— 模型不是显存瓶颈，**吞吐才是**。

---

## 7. 部署约束（用户明确要求纳入考虑）

vLLM **非硬需求**，但以下必须在方案定稿前明确：

### 7.1 为什么当前架构无法直接用 vLLM

1. `de/model.py:224` **手工遍历 blocks 注入记忆**，绕过 HF `forward()`；
   vLLM 替换整个建模代码，永不执行这段循环
2. `registry_slots (B,T,R)` 是**非 token 的第三个 forward 输入**
   （`de/model.py:208`），vLLM 输入管线只产出 token ids
3. `RegistryTable` 是**推理期可变权重**（`de/registry.py:163`），
   多 worker × TP rank 各持私货 → 写入需跨进程广播控制面

且引擎选型本就错配：**vLLM 是自回归解码引擎**（paged KV + continuous batching）；
bi-encoder 每条文本产一个稠密向量，无 KV 复用、无生成。
对应方案是 **SentenceTransformers / TEI / ONNX Runtime + FAISS**。

### 7.2 真正的性能瓶颈

| | 计算量 | 权重(fp32) |
|---|---|---|
| backbone 4 层 | 12.6 M MACs/token | 96 MB |
| Engram 注入 | 1.58 M MACs/token (+12.5%) | **268 MB (2.8×)** |

**参数:算力比 42:1 → 纯访存瓶颈，不是计算瓶颈。**
推理时随机 gather 268 MB 的表会吃掉带宽，batch 小的时候尤其明显。

降本候选（**须先做 E-011/E-012 消融再定**）：
`table_size_per_ngram` 65536→16384 · `n_head_per_ngram` 8→4 · 表存 bf16

### 7.3 待决策的产品级问题（TODO 中从未提及）

- [ ] **新增术语后是否重嵌入命中文档？**
      索引是离线建立的；冷启动只改 query 侧向量 →
      **query/doc 向量空间可能失配**。这是必须先回答的产品决策。
- [ ] **Registry 一致性**：用「不可变快照 + 广播 + 原子 reload」，
      而非请求热写。`PhraseRegistry.version` 已有版本门控
      （`maybe_rebuild`），需扩展为快照式。
- [ ] **`PhraseMatcher` 重写**：当前 Python 逐序列循环 +
      `_rolling_hash` 对每长度跑 `range(L)` 次 numpy，是热路径瓶颈。
      `--num-workers` 只是缓解。

---

## 8. 执行清单（打印勾选）

```markdown
## GB10 执行清单

### 第 0 关 · 机制自检
- [ ] `python offline_test.py` → 9/9 PASSED

### 第 1 关 · GPU 前置验证（从未在 CUDA 上跑过）
- [ ] torch.cuda.is_available() == True
- [ ] bf16 forward 不报错
- [ ] bf16 下 cos(emb_ON, emb_OFF) 仍 ≈ 0.89
- [ ] PhraseMatcher num_workers 加速比实测
- [ ] CUDA 下 registry_table 原地写入无冲突

### 第 2 关 · 核心 🔴
- [ ] `download_data.py --tier p0` → 0 missing
- [ ] **E-009 法律冷启动 A/B/C/D**
- [ ] 记录到 docs/EXPERIMENTS.md（E-009）
- [ ] 判据：D − C (unseen nDCG) 是否 ≥ +2

### 第 3 关 · 消融（E-009 有结论后）
- [ ] E-010 补全训练
- [ ] E-011 表容量 / E-012 N-gram 阶数 / E-013 注入点 / E-015 温度

### 第 4 关 · 生产配置
- [ ] 用 de/config.py 默认配置复跑，确认结论可外推
- [ ] 实测吞吐与显存 → 回填 docs/COMPUTE.md（当前全为推导估算）

### 部署决策
- [ ] 新增术语是否重嵌入命中文档？
- [ ] Registry 快照式 reload 方案
- [ ] PhraseMatcher 重写
```

---

## 9. 已知陷阱速查

| 坑 | 说明 |
|---|---|
| `data/` 不在 git | 必须跑 `download_data.py` |
| HF 网络不通 | 本机曾需代理才能访问；换机器请自行设置 `HTTP_PROXY`，或用 `hf-mirror.com`（约慢 3×） |
| `C-MTEB/*-qrels` 字段 | `qid, pid, score`（**不是** BeIR 的 `query-id, corpus-id, score`） |
| `lex_glue` case_hold 字段 | `endings`（**不是** README 说的 `holdings`） |
| LeCaRDv2 规模 | 实测仅 **159 q / 3,795 cases**（文档误写 800/55,192） |
| 训练负例 vs 评测 corpus | `build_real_eval.py` 已做禁交集校验，**改脚本后务必重跑该检查** |
| 表容量 | 实验 1024 vs 生产 65536，**差 250 倍，数字不可混用** |
| 合成数据已作废 | 饱和（R@1=100）+ 自证，见 E-001，**不要再用** |

---

## 10. 相关文档

| 文档 | 内容 |
|---|---|
| [EXPERIMENTS.md](EXPERIMENTS.md) | **所有实验结论的唯一权威出处**（E-001~E-010b） |
| [TODO.md](TODO.md) | 任务优先级与状态 |
| [DATA.md](DATA.md) | 数据集规格（⚠️ 有 4 处与实测不符，已记录待修订） |
| [COMPUTE.md](COMPUTE.md) | 算力核算（⚠️ 全为推导估算，需实机回填） |
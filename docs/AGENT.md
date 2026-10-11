# AGENT.md — 明日操作手册（NVIDIA GPU）

> **给下一个执行者（人或 agent）的一句话**：
> 先跑 `offline_test.py`，再跑 `run_cold_start.py`（E-009）。
> **E-009 是整个项目第一次能真正证伪 P0 的实验**，先做它，别做别的。
>
> 创建于 2026-10-11。代码基线 commit `69efbd4`（已推 `origin/main`）。
> 相关：[EXPERIMENTS.md](EXPERIMENTS.md)（结论唯一出处）·
> [TODO.md](TODO.md)（任务状态）· [GB10_RUNBOOK.md](GB10_RUNBOOK.md)（背景）

---

## ⚠️ 动手前必读：三条已踩过的坑

这三条都曾导致**错误结论**，且都是静默失败（不报错、数字看起来正常）。

| 坑 | 后果 | 防线 |
|---|---|---|
| **随机 tiny backbone** | 所有历史数字归零 | 只用 `BAAI/bge-small-zh-v1.5`，确认 `hidden=512 layers=4` |
| **match rate = 0** | `D−C` 必然为 0，会被误读成「冷启动无效」 | 每跑 D 组，**先看 match rate 是否 ≈100%** |
| **train/eval 泄漏** | 评测集 = 训练集，指标虚高 | 构建脚本已内置断言；改脚本后必须重跑 |

> 历史教训：E-002 因为没查 match rate，差点得出「写入过度（覆写 backbone）」的假结论。
> **任何「无增益」结论，在排除 match rate = 0 之前都不成立。**

---

## Step 0 · 环境（~15 分钟）

```bash
git clone https://github.com/hellostronger/engram_embedding_model.git
cd engram_embedding_model

bash bootstrap_gb10.sh                    # sm_121 / aarch64，禁 flash-attn，仅 SDPA
pip install -r requirements.txt

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**门槛**：`torch.cuda.is_available()` 必须是 `True`。不是就停下排查，别往下走。

> 已知冲突（可忽略）：`gensim` / `graspologic` 要 `numpy<2`，本项目不用这两个库。

---

## Step 1 · 复现数据（~2 分钟）

`data/` 不在 git 里，用脚本复现（已在本机验证 `--tier p0` → 0 missing）：

```bash
python download_data.py --tier p0          # 0.65 GB，实验必需
python download_data.py --tier p0 --verify # 必须输出 missing 0/13
```

需要英文金融/法律再 `python download_data.py --tier p0 p1`。

**网络**：脚本不硬编码代理。有直连出口即可；需要代理就自己
`export HTTP_PROXY=... HTTPS_PROXY=...`。实在不通可换 `HF_ENDPOINT=https://hf-mirror.com`。

---

## Step 2 · 机制自检（~5 分钟）🔴 门槛

```bash
python offline_test.py
```

**必须 9/9 PASSED。** 这是验证 `attention_bias` 在 CUDA/SDPA 下分支正确性的唯一手段——
**本项目从未在 GPU 上跑过**。失败就停，先修。

---

## Step 3 · GPU 前置验证（~30 分钟）

逐条确认，每条都要有实际输出作证：

- [ ] `torch.cuda.is_available()` → `True`
- [ ] `offline_test.py` → 9/9
- [ ] **bf16 可用**：`EngramConfig` 的 `torch_dtype="bfloat16"` 下 forward 不报错
      （本机只跑过 fp32，bf16 的 RMSNorm/attention 路径未验证）
- [ ] **bf16 下重测写入方向**：
      ```bash
      # diagnose_cold.py 在 CPU 上跑，用它确认机制在 bf16 下方向不翻转
      python diagnose_cold.py --data data/real/eval/law --epochs 2 \
              --backbone BAAI/bge-small-zh-v1.5
      ```
      fp32/CPU 参考值 `cos(emb_ON, emb_OFF) = 0.894`（E-004）。
      **若掉到 <0.5，说明 bf16 撑不住该机制**，换回 fp32 再继续。
      （该脚本无 `--device` 参数，固定 CPU；bf16 专项验证需在 `run_cold_start.py`
      侧通过 `EngramConfig.torch_dtype` 改配置。）
- [ ] **`PhraseMatcher` 瓶颈实测**：它在 CPU 上跑，`num_workers=0` 会与 GPU 串行。
      对比 `num_workers` 0 / 4 的 step 耗时。这决定后续所有实验的吞吐上限。
- [ ] **CUDA 下 registry 原地写入无冲突**：`registry_table.emb.weight` 同时被
      冷启动写入和训练梯度更新，确认不互相覆盖。

---

## Step 4 · 🔴 E-009 冷启动有效性（核心，估 1–3 小时）

**这是整个项目的核心卖点，至今未被证实或证伪。**

```bash
python run_cold_start.py \
    --data data/real/eval/law \
    --epochs 10 --lr 1e-3 \
    --backbone BAAI/bge-small-zh-v1.5 --device cuda \
    --out data/real/eval/law/e009.json
```

数据集已验证零泄漏：T0=2021 严格时间切分，1500 个冷启动术语
**1500/1500 逐字连续出现在其 query 中，0 个泄漏进训练集**（E-008）。

### 四组含义

| 组 | 配置 | 作用 |
|---|---|---|
| A | memory OFF | 基线 |
| B | memory ON 未训练 | 噪声对照 |
| C | memory ON 训练 | 记忆训练效果 |
| **D** | **C + 冷启动写入，不重训** | **核心：冷启动增益** |

### 判据（脚本会自动打印 verdict）

| 指标 | 期望 | 说明 |
|---|---|---|
| `D − C` unseen nDCG | **≥ +2** | TODO 原定目标 |
| `D − C` unseen R@1 | > 0 | 同上 R@1 口径 |
| `D − C` **seen** | **不下降** | 若下降 → 冷启动污染了已学术语，严重问题 |
| `match rate AFTER write` | **≈ 100%** | **若为 0，实验无效**，先修数据 |
| `cos(emb_ON, emb_OFF)` | 0.85 ~ 0.95 | 合理修正；~0 或负 → 写入覆写了 backbone |

### 若 FAIL，按此顺序排查（H1–H4 已在 E-004 排除）

```bash
# H5：门控幅度不足。扫 refine steps，看 D-C 是否随幅度单调
for s in 0 5 20 50; do
  python run_cold_start.py --data data/real/eval/law --epochs 10 --refine-steps $s
done

# 扫 ridge（越大注入越弱）
for r in 1e-2 1e-3 1e-4; do
  python run_cold_start.py --data data/real/eval/law --epochs 10 --ridge $r
done
```

### 记录要求

**结果必须写入 `docs/EXPERIMENTS.md`，编号 E-009**，含环境 / 配置 / 原始输出 / 结论 / 判定。
禁止只在对话里报数——E-001~E-002 就是因为结论没落实处、被反复误引。

---

## Step 5 · 补全与消融（E-009 有结论后）

前置：E-009 出结果。**若 E-009 证伪 P0，后面的消融都失去意义，先停下来重新设计。**

| 编号 | 实验 | 命令要点 | 验证假设 |
|---|---|---|---|
| **E-010** | 补全修正版训练 | `train_baseline.py --data data/real/eval/t2_v2 --n-train 8000 --epochs 10` | H6（本机只跑到 2/20 epoch） |
| **E-011** | 表容量消融 | `run_cold_start.py --table-size {1024,8192,65536}` | H8：实验与生产差 **250×** |
| **E-012** | N-gram 阶数 | `run_cold_start.py --max-ngram 2` vs `3` | H9：3-gram 几乎全废槽 |
| **E-013** | 注入点 | `run_cold_start.py --layer-ids 1 2` vs `1 3` | P1 待修：`[1,3]` 把第二注入点压在**最后一层**，零精炼 |
| **E-015** | 温度 | `train_baseline.py --temperature {0.02,0.05}` | 起点 loss 1.52 偏高 |

> `train_baseline.py` 不支持 `--device` 之外的消融开关（如 `--table-size`）；
> 架构类消融统一用 `run_cold_start.py`（它暴露了完整的架构参数）。

**单轮预估**：本机 CPU 70 min/epoch（8000 triplet, doc=512, bs=32）。
GPU 通常 10–20× 加速 → **3–7 min/epoch**。先跑 1 轮实测再定规模。

---

## Step 6 · 生产配置验证（消融完成后）

实验用**缩小配置**（表 1024 → 8,396 槽 / 67k 参数）；
`de/config.py` 默认是**生产配置**（表 65536 → 2,101,122 槽 / 16.8M 参数），
**差 250 倍**。消融结论必须在这里复核，否则不可外推。

显存不是瓶颈（fp32 全模型 < 0.5 GB），**吞吐才是**。

实测吞吐与显存后回填 `docs/COMPUTE.md`（⚠️ 当前全部是推导估算，从未实测）。

---

## Step 7 · 部署决策（用户明确要求纳入考虑）

vLLM **非硬需求**，但这三件必须在方案定稿前回答：

### 7.1 当前架构为何用不了 vLLM

1. `de/model.py:224` **手工遍历 blocks 注入记忆**，绕过 HF `forward()`；
   vLLM 替换整个建模代码，永不执行这段循环
2. `registry_slots (B,T,R)` 是**非 token 的第三个 forward 输入**（`de/model.py:208`），
   vLLM 输入管线只产出 token ids
3. `RegistryTable` 是**推理期可变权重**（`de/registry.py:163`），
   多 worker × TP rank 各持私货 → 写入需跨进程广播控制面

且引擎选型本就错配：**vLLM 是自回归解码引擎**（paged KV + continuous batching）；
bi-encoder 每条文本产一个稠密向量，无 KV 复用、无生成。
对应方案是 **SentenceTransformers / TEI / ONNX Runtime + FAISS**。

### 7.2 真正的性能瓶颈

| | MACs/token | 权重(fp32) |
|---|---|---|
| backbone 4 层 | 12.6 M | 96 MB |
| Engram 注入 | 1.58 M (+12.5%) | **268 MB (2.8×)** |

**参数:算力比 42:1 → 纯访存瓶颈，不是计算瓶颈。**

### 7.3 待决策项

- [ ] **新增术语后是否重嵌入命中文档？** ← 产品级决策，TODO 中从未提及。
      索引离线建立，冷启动只改 query 侧向量 → query/doc 空间可能失配
- [ ] **Registry 一致性**：用「不可变快照 + 广播 + 原子 reload」，非请求热写
- [ ] **`PhraseMatcher` 重写**：当前 Python 逐序列循环，
      `_rolling_hash` 对每长度跑 `range(L)` 次 numpy

---

## 执行清单（打印勾选）

```markdown
## GPU 执行清单

### Step 0 · 环境
- [ ] bootstrap_gb10.sh + requirements.txt 装好
- [ ] torch.cuda.is_available() == True

### Step 1 · 数据
- [ ] download_data.py --tier p0
- [ ] --verify 输出 missing 0/13

### Step 2 · 门槛
- [ ] offline_test.py → 9/9 PASSED

### Step 3 · GPU 前置
- [ ] bf16 forward 不报错
- [ ] bf16 下 cos(ON,OFF) 仍 ≈ 0.89
- [ ] PhraseMatcher num_workers 加速比实测
- [ ] CUDA 下 registry 原地写入无冲突

### Step 4 · 🔴 核心
- [ ] E-009 跑完，match rate ≈ 100%
- [ ] D − C (unseen nDCG) 记录
- [ ] cos(ON,OFF) 记录
- [ ] 写入 docs/EXPERIMENTS.md 编号 E-009

### Step 5 · 消融（E-009 有结论后）
- [ ] E-010 / E-011 / E-012 / E-013 / E-015

### Step 6 · 生产配置
- [ ] de/config.py 默认配置复跑
- [ ] 实测吞吐/显存回填 COMPUTE.md

### Step 7 · 部署决策
- [ ] 新增术语是否重嵌入命中文档？
- [ ] Registry 快照式 reload
- [ ] PhraseMatcher 重写
```

---

## 常见故障速查

| 现象 | 原因 | 处理 |
|---|---|---|
| `offline_test.py` 失败 | CUDA/SDPA 路径未验证过 | 停下修，别继续 |
| `match rate AFTER write = 0%` | 数据配对错位（E-008c 的坑） | 跑 `build_cold_law.py`，看 `_verify` 是否 100% |
| `D − C = 0` 且 match rate = 100% | 可能是真的无增益，也可能是饱和 | 检查 R@1 是否已 100（饱和则无效） |
| HF 下载超时 | 网络受限 | `export HF_ENDPOINT=https://hf-mirror.com` |
| CUDA OOM | — | 本模型显存 < 0.5 GB，OOM 说明有别的问题 |
| import 报 numpy/scipy 冲突 | 环境版本 | `pip install "scipy>=1.13"`（需 NumPy 2.x ABI） |

| 常被引用的错误数字 | 真相 |
|---|---|
| `C−A = +14.30` | ❌ 随机 tiny backbone，真 backbone 饱和为 **+0.00** |
| `D−C = +0.06` | ❌ 平台无效（tiny + 泄漏） |
| `cos(ON,OFF) = −0.044` | ❌ tiny 伪影，真 backbone 是 **+0.894** |

以上均见 `docs/EXPERIMENTS.md` 第一部分「作废实验」，**勿再引用**。

---

## 任务分配建议（若并行）

E-009 依赖 Step 3 的 GPU 前置结论，不能并行。
但以下可与之并行，互不干扰：
- 准备 `t2_v2` / `du` / `ecom` 等其他评测集（纯 CPU）
- `PhraseMatcher` 重写的原型（纯 CPU，可另开分支）
- `docs/DATA.md` 4 处实测偏差的修订（纯文档）
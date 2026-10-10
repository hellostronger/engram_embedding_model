# Domain Engram — 面向垂域的动态 Embedding 模型

借鉴 DeepSeek **Engram**（*Conditional Memory via Scalable Lookup*, arXiv:2601.07372）
的「条件记忆」机制，把领域新词/实体/术语存入一个**可动态增删改查**的外部记忆表，
与冻结的通用编码器解耦，实现**存储与计算分离**的垂域 Embedding。

```
输入 ──► 基础编码器(通用语义) ──► 隐状态 h_t
                                    │        ▲
                     ┌──────────────┘        │ 轻量融合(门控+卷积)
                     ▼                       │
        哈希 N-gram 表 ──► e_t ──► α_t·W_V e_t ──► H_t ← H_t + y_t
        (Layer-B, 训练学)              │
        注册表 CRUD  ──► e_t ──────────┘
        (Layer-A, 可运营)
```

## 核心设计：双层记忆

| 层 | 内容 | 寻址 | 可写 | 用途 |
|---|---|---|---|---|
| **Layer-A** Registry | 领域术语/实体/短语 | 显式 `phrase→slot` 字典 | ✅ O(1) 增删改查 | 运营写入新词 |
| **Layer-B** Hashed N-gram | 通用局部模式 | 乘法-XOR 哈希 | ❌ 训练填充 | 大容量统计记忆 |

> **为什么必须双层**：Engram 原表的槽位由哈希决定，**不可寻址**，无法做「改一条不动
> 其它」。Registry 层补上了这个缺口，同时保留哈希层的大容量。

## 快速开始

```bash
pip install -r requirements.txt

# 0) 先读懂原论文思路（推荐）
#    docs/ENGRAM_PAPER.md   配 3 张 SVG 架构图

# 1) 离线自检（无需网络，验证全部机制）
python offline_test.py

# 2) 生成合成数据（4 领域，零依赖）
python make_synthetic_data.py --out data/synth --domain medical --n 2000

# 3) A/B/C/D 四组对照实验（验证设计）
python run_abcd.py --data data/synth --epochs 2

# 4) 真实数据（需联网）
python prepare_data.py --dataset med_qa --out data/medical.jsonl
```

## 在 GB10 / GPU 上跑真实实验

```bash
# 自动探测设备 + 选 preset + 校准吞吐
python run_gb10.py --preset medium --stage describe
python run_gb10.py --preset medium --stage bench

# 完整流程：训练 → 冷启动写入 → 评估
python run_gb10.py --preset medium --stage all \
    --train data/medical.jsonl \
    --new-terms data/cold_start.jsonl
```

## 目录结构

```
de/
  hashing.py      压缩 tokenizer + 多头乘法-XOR 哈希（O(1) 寻址）
  registry.py     Layer-A CRUD 记忆（append-only 槽位、墓碑删除、free-list）
  engram.py       上下文门控 + 深度因果卷积（零初始化）
  model.py        backbone 注入 + 记忆抑制开关
  cold_start.py   跨上下文共识 + 闭式 ridge 解（零梯度冷启动）
  evaluation.py   Recall@k / MRR / nDCG（IDCG 已修正）
  runtime.py      设备探测 + bf16 降级 + 显存自适应 preset
  testing.py      离线 tiny BERT + 数据派生 tokenizer
docs/
  ENGRAM_PAPER.md 原论文思路解析 + 架构图（先读这篇）
  figures/        3 张 SVG 架构图（模块 / 洞察 / 稀疏性定标律）
  PLAN.md         可行性分析与实施计划
  DATA.md         训练数据规格（19 个已验证数据集）
  EVAL.md         评估集与冷启动评测协议
  EXPERIMENTS.md  基线/消融/公平对比/成功门槛
  COMPUTE.md      GB10 算力与显存核算
  HARDWARE.md     硬件支持矩阵与限制
```

## 关键结论

- **代码不挑卡**：纯标准算子，无自定义 kernel。bf16 需 sm_80+，其余自动降级。
- **显存 = 记忆表规模**：`medium` preset 可训参数 80M / 优化器状态 1.28 GB。
- **冷启动零梯度**：闭式 ridge 解，单术语 ~毫秒级，10 万术语 ~5 分钟。
- **门控是安全网**：写错的记忆在上下文不匹配时 α→0，不污染 backbone。

## 许可

本仓库代码 MIT。Engram 机制引用自 arXiv:2601.07372（CC BY 4.0）与
deepseek-ai/Engram（Apache-2.0）。
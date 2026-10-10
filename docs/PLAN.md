# 面向垂域的动态 Embedding 模型（Engram 式条件记忆）— 可落地实施计划

> 本文档基于对 DeepSeek **Engram / Conditional Memory via Scalable Lookup**
> (arXiv:2601.07372, v2 2026-07-12; github.com/deepseek-ai/Engram, Apache-2.0) 的完整技术分析，
> 目标是把它从"预训练 LM 的条件记忆模块"改造成"**可增删改查的垂域 Embedding**"。
>
> 编写日期：2026-10-10
> 结论前置：本方案**可行**，但必须按本机硬件（无 GPU）重新划定验证范围。见 §7 卡资源。

---

## 0. 一句话方案（Domain Engram, 简称 DE）

```
                 ┌─────────────── 可动态增删改查的外部记忆表（Engram Store）───────────────┐
                 │  显式短语/实体/术语  ──►  Registry（字符串→slot_id，可 CRUD，有日志）  │
                 │  通用 N-gram 模式    ──►  Hashed N-gram 表（O(1) lookup，训练学）    │
                 └───────────────────────────────────────────────────────────────────────┘
输入文本 ──► 基础编码器 (Transformer, 通用语义) ──► 隐状态 h_t
                                        │                    ▲
                                        │                    │ 轻量融合模块（Engram Fusion）
                                        ▼                    │
                       z = hash(compressed N-gram) ──► 记忆向量 e_t
                                        │
                    k_t = W_K e_t,  v_t = W_V e_t
                    α_t = σ( RMSNorm(h_t)ᵀ RMSNorm(k_t) / √d )      ← 上下文感知门控
                    ỹ_t = α_t · v_t
                    y_t = SiLU(Conv1D(RMSNorm(ỹ))) + ỹ               ← 深度因果卷积
                    H_t ← H_t + y_t                                 ← 残差注入
                                        │
                                        ▼
                              Pooling → 最终 embedding（领域术语/长尾实体被"点亮"）
```

**三个目标与机制的对应关系：**

| 你的目标 | 机制 | 是否被 Engram 原文支撑 |
|---|---|---|
| 动态更新词表/短语记忆，不重训 | 记忆表与编码器解耦；Registry 支持 O(1) 增删改 | ✅ 2.5 节"存储与计算解耦"+确定性寻址 |
| 新词/长尾实体冷启动 | 门控 α_t 可"打开"记忆；写入=直接写 slot 向量 | ✅ 6.5 节门控可视化（新词/实体命中高 α） |
| 低成本垂域适配 | 只需训融合模块+记忆表，backbone 可冻结 | ✅ 6.2/6.4（offload 开销 <3%） |
| 存储与计算解耦 | 确定性 hash 索引 → 可预取、可 CPU offload | ✅ 2.5 + 6.4（100B 表 offload 仅 -2.8%） |

---

## 1. 关键证据：Engram 机制逐条拆解（决定我们怎么改）

从论文与官方 `engram_demo_v1.py` 中提炼，**直接决定 Embedding 改造的每一个设计点**：

1. **Tokenizer 压缩（Compression）**：预计算满射 `P: V→V'`，按 NFKC/小写等归一化，
   128k→降 23%。→ **Embedding 版必须复用**：术语大小写/全半角归一，否则
   "Walmart"/"␣Walmart" 占两个 slot，浪费记忆容量。
2. **多阶 N-gram + 多头哈希（Multi-Head Hashing）**：`z = φ(g_{t,n})`，
   每个 n∈[2,N] 用 K 个不同哈希头，落到素数大小表 `E_{n,k}`；`e_t = concat_{n,k} e_{t,n,k}`。
   φ 用"轻量乘法-XOR 哈希"。→ **Embedding 版直接沿用**，这是 O(1) 的来源。
3. **上下文感知门控（Gating）**：`α_t = σ(RMSNorm(h_t)ᵀ RMSNorm(k_t)/√d)`，
   `ỹ_t = α_t·v_t`。若记忆与上下文矛盾，α→0，自动抑制噪声（哈希碰撞/多义词）。
   → **这是"冷启动可控"的关键**：写进记忆但上下文不匹配时不会被强行注入。
4. **轻量深度因果卷积**：`y = SiLU(Conv1D(RMSNorm(ỹ))) + ỹ`，kernel=4，dilation=N。
   扩大感受野。消融里"去掉卷积影响小"，但保留可提升鲁棒性。→ 可选，低成本。
5. **残差注入 + 分层放置**：`H ← H + y`，**不是每层都放**。层扫描显示 Layer 2 单点最优；
   拆成两个小模块放 Layer 2 和 6 更好（既早干预又有后期强上下文门控）。
   → **Embedding 版**：短序列（query/doc 通常 <64 token），建议"早期一层注入"为主。
6. **多分支（mHC, M=4）分支专属门控**：共享一张记忆表 + 一个 W_V，M 个 W_K^(m)，
   融成一次 FP8 GEMM。→ **Embedding 版可选**：若不用 mHC，退化为单分支单 gate，参数更省。
7. **敏感性分析（关键，决定任务选择）**：把 Engram 输出推理时全关掉，
   - 事实知识类 → 只剩 **29–44%**（TriviaQA 仅 29%）→ Engram 是事实知识的主要载体
   - 阅读理解类 → 保留 **81–93%** → 靠 backbone attention
   → **含义**：本方案适合"领域术语/实体/短语的精确语义匹配（检索/对比学习）"，
     **不适合**纯上下文推理任务。Embedding 检索恰好属于前者。✔
8. **消融重要性排序**：①多分支融合 ②上下文门控 ③tokenizer 压缩 —— 三者去掉损失最大。
   → 优先级最高，**不能省**。④卷积影响小 ⑤4-gram 在 1.6B 预算下略不如 2/3-gram。
9. **确定性寻址 → 预取/offload**：索引在 forward 前即可确定，可与计算重叠；
   100B 表全 CPU offload，吞吐仅降 <3%（8B backbone，H800）。→ 系统层解耦的证据。

---

## 2. 从 LM → Embedding 的适配（我们要做的 5 处改造）

Engram 原文注入在 LM 的残差流里，目标是 next-token loss。Embedding 模型要输出一个
**池化后的句向量**供检索/聚类/对比学习。关键改造点：

### 改造 1：注入位置与池化协同
- 不用 LM head；把 Engram 放在 encoder **第 2 层之后**（早干预），再交给后续层精炼。
- 池化策略：CLS / mean / last-token 三选一做消融。Engram 增强的是**局部模式**，
  预期 mean-pooling 收益最大（覆盖更多注入点），但需实测。

### 改造 2：对比学习目标（替换 LM CE loss）
原文用 CE。我们用 InfoNCE（对称，in-batch negatives），可选加：
- 难负例挖掘（domain 内的近邻）
- 多向量/多语对齐（若做中英垂域）
- 与 backbone 联合训练（backbone 微调 or LoRA/冻结二选一做消融）

### 改造 3：记忆表的可 CRUD 设计（**这是我们相对 Engram 的核心增量**）
Engram 原文的表是"训练出来的"，槽位由 hash 决定，**没有显式的增删改接口**。
Embedding 落地要求"运营能插一条新术语"。我们做**双层记忆**：

- **Layer-A：显式短语/实体 Registry（可写）**
  - 结构：`phrase_string → (slot_id, table_id)` 的哈希字典（可持久化到磁盘/DB）。
  - 向量存在一个可增长的连续表 `[num_slots, d_mem]`（append-only + free-list 复用）。
  - **增删改查 O(1)**：写一条新术语 = 分配 slot + 写一行 + 更新 registry 索引；
    改 = 覆写该行；删 = 标记 tombstone + 入 free-list（索引不动，避免重训）。
  - 冷启动初始化（**很关键**，否则新 slot 是噪声）：
    - 方案 a：从语料中该短语的出现上下文里聚合 backbone 隐状态均值 → 初始化向量；
    - 方案 b：正交初始化 + 小学习率在线更新（安全但慢）；
    - 方案 c（推荐）：a 得到初值，再对**新 slot 的这一行**做少量梯度步（几十步），
      只更新该行与共享 W_V（LoRA 风格），backbone 与其他 slot 冻结。
  - **安全性**：门控天然兜底——即使写了错误记忆，不匹配上下文时 α→0，不污染。

- **Layer-B：Hashed N-gram 表（训练学，只读）**
  - 完全照 Engram：压缩 tokenizer + 多头乘法-XOR 哈希 + 素数表。
  - 这层负责"通用 N-gram 模式"，容量大（可达数十亿槽），由训练填充，不可手工改。
  - 部署时按 Zipf 分层缓存（热 pattern 上 GPU，冷 tail 留 CPU/SSD）。

> 为什么必须双层？Hashed N-gram 槽位不可寻址（不知道某 phrase 落在哪个槽），
> 无法做"改一条不动其它"。而纯 Registry 表容量小、覆盖不到通用局部模式。
> 双层兼顾"可运营 CRUD"和"大容量模式记忆"。

### 改造 4：融合模块保持轻量
只训：(a) 记忆表（Layer-A 行 + Layer-B 表）、(b) W_K/W_V、(c) 小 conv、RMSNorm。
- 参数量估计：backbone 冻结时，可训参数 ≈ 几百万～几千万，**远小于全量微调**。
- 遵循原文优化：记忆表用 Adam，lr 放大 5×、**无 weight decay**；conv 零初始化
  （保证初始为恒等映射，训练稳定）。

### 改造 5：可选的"预算再分配"启发
原文 Sparsity Allocation 的 U 形曲线（MoE vs Engram 分配 ~75–80% 给 MoE 最优）。
我们没有 MoE，但**可借用其"给记忆留固定比例预算、其余给 backbone"的思路**，
做一个小消融：{backbone 容量 vs 记忆容量} 的粗粒度分配，验证是否也有最优区间。
（这是加分项，非必须。）

---

## 3. 数据层面可行性

### 3.1 训练/适配数据（构造垂域对比学习语料）
| 领域 | 语料来源（公开可下） | 用途 |
|---|---|---|
| 通用底座 | BGE / E5 公开检索语料子集 | 保持通用能力不塌陷 |
| 医疗 | PubMed 摘要、MedNLI/PubMedQA、健康类 QA | 术语/实体 |
| 金融 | FinQA/FiQA、财报/年报文本、财经新闻 | 术语/N-gram |
| 法律 | 法律文书/判例（CaseHOLD 等）、LexGLUE 子集 | 长实体/术语 |
| 电商 | Amazon/淘宝评论、类目属性文本 | 短语/型号 |

- 构造方式：从上述语料抽 (query, positive, hard-negative) 三元组，InfoNCE 训练。
- **关键：必须构造"记忆可解决"的任务**——即 query 与 doc 共享领域短语/实体，
  记忆能提供该短语的稳定语义。这类样本是 DE 相对 backbone 的增量来源。

### 3.2 记忆表语料（挖 N-gram / 术语）
- 对每个领域语料：分词 → tokenizer 压缩 → 统计 suffix N-gram 频次（Zipf）。
- 挖"领域术语/实体"候选：
  - 规则法：高频 N-gram、低普适性（在通用语料里低频）、领域内高 TF-IDF；
  - 词典法：结合领域词表（如 ICD、药品名、法条、商品类目）；
  - 无监督法：基于互信息/凝固度（PMI）+ 子串切分抽领域短语。
- 这些候选进 Layer-A Registry；Hashed N-gram 覆盖其余。

### 3.3 评估数据（分三层，对应你的四个目标）
1. **通用能力不掉**：MTEB / C-MTEB 子集（Retrieval + STS + Classification）。
   防"领域适配过头，通用塌陷"。
2. **垂域检索**：选 1 个领域做深（建议**中文电商或医疗**，公开检索评测更成熟）。
   指标：Recall@k / nDCG@10 / MRR。
3. **动态/冷启动专测（我们的杀手锏，直接对应你的目标 1、2）**：
   - **更新测试**：训练完结后，向 Registry 插入一批**训练时没见过的新术语/新实体**，
     只做 Layer-A 写入（不重训 backbone、不动 Layer-B），立刻测检索涨点。
     这直接量化"动态更新词表"的收益。
   - **删除测试**：删掉记忆里错误的条目，验证 α 门控 + tombstone 能恢复性能
     （测"查错/污染"鲁棒性）。
   - **消融**：backbone-only / DE 无门控 / DE 无卷积 / DE 无 tokenizer 压缩 / DE 全量，
     对应原文消融优先级。

> MTEB 全量很大，本机跑不动 → 只取 Retrieval + 少量 STS 子集做"守门"，垂域检索做"主战场"。

---

## 4. 模型层面可行性（对应 §1 改造的落点）

- **Base encoder 选型（CPU 优先）**：
  - 首选 **BGE-small-zh-v1.5 / bge-base-zh-v1.5**（中文，384/768 维，CPU 可跑）
    或 **multilingual-e5-small**；若追求上限且有 GPU 再上 bge-base/m3。
  - 维度 384 足够做检索，训练与推理都轻，便于 CPU 迭代。
- **冻结 vs 微调 backbone**：
  - 主实验先**冻结 backbone**（只训记忆+融合）→ 最快验证"记忆是否真的带来增益"，
    且这正是"低成本"卖点的最小可行版本。
  - 若增益确认，再开 backbone LoRA（秩 8~16）联合微调 → 追求更高上限。
- **维度对齐**：d_mem（记忆表维度）建议与 backbone hidden 对齐或取其 1/2；
  原文 27B 用 d_mem=1280 对 hidden 2560（约 1/2）。小模型按比例缩小。
- **注入层**：单点放 Layer 2（原文最优）；可选加第二点 Layer 6/尾层（用小 d_mem）。
- **池化**：CLS vs mean vs last 消融。

---

## 5. 验证层面（如何证明"可行且有效"）

设 4 组对照（同一 backbone、同一数据、同一检索评测）：

| 组 | 配置 | 期望（依据 Engram 原文） |
|---|---|---|
| A. Baseline | 冻结 backbone，纯检索 | 基准分 |
| B. DE-RandomMem | 加记忆表，但向量随机/无门控 | ≈ A 或略降（噪声） |
| C. DE（完整） | 压缩+多哈希+门控+卷积，记忆学 | **显著 > A**（垂域尤明显） |
| D. DE + 动态更新 | C 之后插入新术语，只写 Registry | **D > C** 且无需重训 ← 核心卖点 |

**消融（对应原文 §6.2 重要性排序）**：去 W_K 多分支/门控、去 tokenizer 压缩、去 conv、去 4-gram(只留 2/3-gram)。预期去掉"门控/压缩/多分支"掉点最多。

**关键指标与成功门槛**（建议，可按领域调）：
- 垂域检索 nDCG@10：DE 相对 A 提升 **≥ +3% 相对提升**（对齐原文 MMLU +3~4 的量级逻辑）。
- 通用检索不掉：相对 A 下降 **≤ 1%**。
- **动态更新**：D 组在"新术语"子集上提升 **≥ +5%**，且**零重训**（毫秒级写入）。
- 门控有效性：可视化 α 在命中领域术语时显著高于随机（复刻原文 §6.5）。

---

## 6. 训练层面（可执行 recipe）

**阶段划分（对应你的四个目标）：**

- **Stage 0 · 底座准备**：下载 BGE-small-zh；冻结；抽领域语料三元组。
- **Stage 1 · 记忆预热（Memory Warm-up）**：
  - 训 Layer-B hashed N-gram 表 + Layer-A 初值 + 融合层（W_K/W_V/conv）。
  - backbone 冻结；InfoNCE；Adam；记忆表 lr=backbone_lr×5、wd=0；conv 零初始化。
  - 目标：把组 C 训到达标（垂域 nDCG 显著 > A）。
- **Stage 2 · 冷启动写入（新词/实体）**：
  - 抽训练未见术语；a 方案聚合 backbone 隐状态 → 写 slot；c 方案对该行做数十步梯度微调。
  - 目标：组 D 提升，验证动态更新能力。
- **Stage 3 · 部署前固化**：分层缓存（热 N-gram 上 GPU）、导出门控统计、快照 Registry。

**关键工程点（照搬原文）：**
- 记忆表用 Adam + 放大 lr + 无 weight decay（防遗忘/正则漂移）。
- conv 零初始化 → 初始为恒等映射，训练稳定。
- 哈希用素数表 + 乘法-XOR，确定性 → 可预取、可复现。
- 梯度：backbone 冻结时记忆表梯度独立、显存友好。

---

## 7. 卡资源与训练时间（**本机实测，必须现实**）

### 7.1 本机实测（刚刚在 `D:\engram_embedding_model` 环境探测）
- GPU：**无**（`nvidia-smi` 不存在；仅 Intel UHD 620 核显，1GB）。
- CPU：Intel i7-8550U，4 核 8 线程 @1.8GHz（老款低压）。
- 内存：23.9 GB。
- 磁盘：D: 仅剩 **17.1 GB**，C: 仅剩 **1.9 GB**（紧张）。
- Torch：**2.6.0+cpu**，`cuda=False`；transformers 4.57.1；st 3.0.1 已装。
- **实测 GEMM 吞吐 ≈ 142 GFLOP/s (fp32, 4 线程)**。

### 7.2 由此推出的可行性边界（重要）
- ✅ **本机完全可做**：
  - BGE-small-zh（33M 级）**推理**（编码百万级文本，CPU 可接受）。
  - **DE 的完整架构 + 全部消融的"小规模可复现实验"**（小 backbone + 小记忆表 +
    几万条训练三元组）。这类实验 CPU 上是**小时级**，足以验证机制有效性、
    跑通 §5 的 4 组对照与消融、复现门控可视化。
  - Registry 的 CRUD 增删改查、动态更新测试（毫秒级写入，纯 CPU 轻松）。
- ⚠️ **本机吃力/不可做**：
  - 用 bge-base/m3 等大 backbone 做**训练**（CPU 训练 ~天级，不现实）。
  - MTEB 全量评测（数据量大、耗时长）。
  - 复刻原文 27B 规模（不可能，本机无卡、内存/磁盘也远不足）。
- 🔧 **推荐落地策略（二选一）**：
  1. **本机 CPU 路线（保真原型）**：bge-small-zh + DE，小语料，小时级跑通全流程，
     产出"机制有效 + 动态更新有效"的完整证据链与代码。这已足以支撑"方案可行"。
  2. **云/租卡路线（出结果）**：租 1 张 A10/A100/4090（≈¥1.5~8/小时），
     用 bge-base/m3 + 中等规模语料，1~2 天内跑出高质量垂域检索数字与完整 MTEB 子集。
     成本很低（几十~百元级）。

### 7.3 粗略训练时间预算（CPU，本机）
以 BGE-small-zh（≈33M 参）冻结 backbone、只训记忆+融合（可训参 ≪ backbone）、
几万条 InfoNCE 三元组、单 epoch 计：
- 吞吐量级：CPU（142 GFLOP/s）下，小模型+短序列，**约 10^4–10^5 样本/分钟**（保守）。
- 几万条三元组 × 少量 epoch ≈ **0.5 ~ 数小时**。做 4 组对照 + 若干消融（并行/串行）
  ≈ **一天内**可拿到完整 §5 表格。→ 可行。
（具体以 Stage 1 首跑实测为准，用 Profiler 校准。）

---

## 8. 里程碑与交付物

| 里程碑 | 内容 | 验收 |
|---|---|---|
| M0 环境 | 仓库脚手架、依赖、数据下载脚本 | `python -m de.demo` 跑通 |
| M1 架构 | DE 模块（压缩tokenizer+多哈希+门控+卷积+Registry）单元测试通过 | hash 确定性、CRUD 单测绿 |
| M2 Baseline | BGE-small-zh 在评估集上的 A 组分数 | 出基准 nDCG/Recall |
| M3 主实验 | 训出 C 组（DE 完整） | 垂域 nDCG 显著 > A；通用不掉 |
| M4 动态更新 | 插新术语出 D 组 + 消融 + 门控可视化 | D>C 且零重训；消融排序符合预期 |
| M5 报告 | 完整实验表 + 复现代码 + 云端放大方案 | 文档 + 可运行 pipeline |

---

## 9. 风险与对策

| 风险 | 概率 | 影响 | 对策 |
|---|---|---|---|
| 记忆表稀疏学不动（长尾 N-gram 梯度稀疏） | 中 | 中 | 哈希表 Adam + 放大 lr；冷启动用聚合初始化而非纯随机 |
| 领域适配过拟合，通用能力塌陷 | 中 | 高 | 通用数据混入 + MTEB 守门 + 通用指标不达标就回退 Stage1 超参 |
| 门控学不出"选择性"（α 恒开/恒关） | 中 | 高 | 门控是原文消融第②大项，必保留；可视化 α 监控；必要时加稀疏/温度正则 |
| CPU 太慢，实验拖期 | 高 | 中 | 用 small 模型 + 小语料做保真原型；结果阶段再租卡放大 |
| 磁盘只剩 19GB 不够 | 中 | 中 | 清理 C:；数据集只下子集；模型用 base/small |
| CRUD 与哈希寻址冲突（不知道 slot 在哪） | — | — | 已用**双层记忆**（Registry 显式 slot + Hashed 只读）解决 |

---

## 10. 与 Engram 原文的关系（诚实声明）

- **借鉴**：条件记忆/哈希 N-gram 寻址、上下文感知门控、轻量卷积、确定性预取、
  "存储-计算解耦"、消融优先级。均引用自 arXiv:2601.07372（CC BY 4.0）与官方 Apache-2.0 demo。
- **我们的增量（Engram 未做）**：
  1. 从 next-token LM → **对比学习 Embedding**（InfoNCE + 检索）；
  2. **显式可 CRUD 的 Registry 记忆层**（Engram 原表不可手工增删改）；
  3. **冷启动写入机制**（聚合初始化 + 定点微调）与**动态更新评测协议**；
  4. 领域术语/N-gram 的挖掘与预算分配的小消融。
- 定位：这是**把 Engram 的条件记忆范式迁移到垂域 Embedding 的工程化改造 + 验证**，
  非复现 27B 预训练。

---

## 附：关键引用

- Cheng et al., *Conditional Memory via Scalable Lookup* (Engram), arXiv:2601.07372v2.
- deepseek-ai/Engram 官方代码（Apache-2.0）: `engram_demo_v1.py`（压缩 tokenizer /
  乘法-XOR 多头哈希 / 门控 / ShortConv 实现）。
- BGE (Chen et al.) / E5 (Wang et al.) 作为基础 encoder 与通用检索评测参照。
- 领域数据：PubMedQA / FinQA / CaseHOLD / Amazon Reviews 等公开集。
# EVAL.md — Domain Engram 评测规范

> 编写日期：2026-10-10
> 上游文档：`docs/PLAN.md`（本文件是 PLAN §3.3 / §5 / M4 里程碑的可执行细化）
> 适用代码：`de/evaluation.py`、`de/registry.py`、`de/cold_start.py`
> 硬件假设：**单台 GB10（DGX Spark / 同级，128 GB LPDDR5X、273 GB/s、≤1 PFLOP FP4）**
> 环境约束：本仓库沙箱代理**屏蔽 huggingface.co 下载**。本文所有 HF dataset id 均通过
> `https://huggingface.co/api/datasets/<id>` 与 `https://datasets-server.huggingface.co/size?dataset=<id>`
> **在线核验**（这两个端点只返回 JSON，不下载数据），核验时间 2026-10-10。
> 凡未能核验的条目标记 **UNVERIFIED**，不做任何猜测。

---

## 0. 结论速览（TL;DR）

| 目标 | 落点 | 一句话结论 |
|---|---|---|
| 通用能力不掉 | §2 Guardrail（8 任务，≈ 8 min/模型） | 用 `MTEB(cmn, v1)` 检索子集 + `MTEB(eng, v2)` 的 4 个小语料检索任务 |
| 中文医疗 | §3.1 **CMIRB**（10 任务，`CMIRB/*`）+ `mteb/CmedqaRetrieval` | CMIRB 是目前唯一成体系的中文医疗检索套件，全部 Apache-2.0 或继承 C-MTEB |
| 中文金融 | §3.2 **FinMTEB**（`FinanceMTEB/*`，64 数据集 7 任务，EMNLP 2025） | 检索类只有 5 个，其中 AlphaFin/FinTruthQA 规模可用 |
| 中文法律 | §3.3 **LeCaRDv2**（`mteb/LeCaRDv2`）+ **LEAD**（`JamesChengGao/LEAD`） | ⚠️ MTEB 版只有 **159 query**，**无法分辨 3% 相对差**；必须补构造集 |
| 中文电商 | §3.4 `mteb/EcomRetrieval` + `mteb/MMarcoRetrieval` | 成熟，100k 语料 / 1k 与 7k query |
| 冷启动 / 动态更新 | §4 **自建评测协议**（无现成 benchmark），核心贡献 | WRITE/CONTROL/RANDOM-ng 三分 + 剂量曲线 + 回滚等效检验 + 门控抑制 |
| 统计功效 | §5 | 3% 相对 nDCG@10、p<0.05、power 0.8 → **n ≈ 1,300–1,900 queries**；设计取 **≥ 2,500** |
| 门控选择性 | §6 | **Term-Final AUROC** + matched-pair McNemar + **Gate-Mass Enrichment Ratio** |

---

## 1. 先修评测代码：现有 `de/evaluation.py` 的三处与标准 benchmark 不兼容

这一节必须先做，否则 §2/§3 的所有分数都不可信。

### 1.1 当前是「小候选池」评测，不是全语料排序

```python
# de/evaluation.py:100-108
g = torch.Generator().manual_seed(0)
distract = torch.stack([torch.randperm(C.shape[0], generator=g) for _ in range(sim.shape[0])])
cand = torch.cat([idx.unsqueeze(1), distract], dim=1)    # (nq, N+1)
cand_sim = torch.gather(sim, 1, cand)
out = retrieval_metrics_from_scores(cand_sim, gold_col=0)
```

* 全语料相似度矩阵 `sim` 已经算出来了，但**只 gather 了 1 gold + N 个随机 distractor**。
* `N = C.shape[0]`（整个语料！），所以实际上随机候选数 = 语料大小 − 1，只是被压缩成一张
  11 列（`retrieval_metrics_from_scores` 只用 `ranks`，等价于在全语料上排名后再报告 top-k）。
  → **这一点其实是好的**，等价于 full-corpus ranking。
* ⚠️ **但有三个真问题**：

| # | 问题 | 影响 | 修法 |
|---|---|---|---|
| P1 | `retrieval_metrics()` 用 `sim.diag()` 取 gold | 只在 **query i ↔ corpus i** 的 1:1 场景成立，无法用于真实 benchmark | 新增 `full_corpus_rank(sim, pos_idx)`：`(sim > sim[i, pos_idx[i]]).sum(1)` |
| P2 | `retrieval_metrics_from_scores` 的 `idcg` 用 `arange(ranks.shape[0])`，即 **候选数**（≈语料大小）而不是 `min(k, n)` | `ndcg@10` 实际是 `nDCG@N`，与 BEIR/MTEB 口径不一致，分数会偏低且不可比 | `idcg` 截断到 `k=10`（单 gold 时 `idcg` 恒为 1），与 BEIR 一致 |
| P3 | 每个 query 独立调 `randperm` 且 `seed=0`，但**同一 query 的候选集对所有模型是固定的** | 无法做 §5 的 paired bootstrap（需要逐 query 分数） | 增加 `return_per_query=True`，导出 `{qid, rank, ndcg10, recall1/5/10}` |

### 1.2 必须新增：`full_corpus` 模式

```python
@torch.no_grad()
def evaluate_full_corpus(model, corpus_texts, query_texts, gold_idx,
                         batch_size=256, max_len=256, device=None,
                         suppress_memory=None, return_per_query=False):
    """在【全语料】上对每条 query 排名，返回 MTEB/BEIR 口径的 Recall@k / MRR / nDCG@10。

    - gold_idx[i] = corpus 中 query i 的唯一（或主）正例下标。
    - 采用 tie-aware 的 "rank = #(score_strictly_greater)"，与 MTEB/pytrec_eval 一致。
    - 若基准是分级相关性（如 LeCaRD），改传 qrels: Dict[qid, Dict[docid, rel]]，
      用标准 DCG/IDCG 计算，不要复用单 gold 分支。
    """
```

### 1.3 与标准 benchmark 的「形状」不匹配清单

| 现象 | 出现在 | 是否适配我们的 1-gold 形状 | 必须做的适配 |
|---|---|---|---|
| **BEIR / MTEB 是 TREC run 格式**（`qrels` + 可多正例 + 分级相关性） | 全部 Retrieval 任务 | ✗ | 走 `full_corpus` 路径 + `pytrec_eval` 的 `ndcg_cut_10`；**不要**用 `retrieval_metrics_from_scores` |
| 一 query 多正例 | `mteb/scifact`, `mteb/nq`, `mteb/quora`, `CMIRB/DXYDiseaseRetrieval` 等 | ✗ | 取 relevance>0 的第一个作 gold 做「1-gold 视图」，并**额外**报告标准多 gold 口径；两套数都要进报告 |
| 分级相关性 (gain ≠ 1) | `mteb/LeCaRDv2`, `mteb/arguana`(部分), `mteb/quora` | ✗ | 用分级 DCG，或二值化后走单 gold 分支 |
| query 数极少 (<200) | `mteb/trec-covid` (50), `mteb/LeCaRDv2` (159), `FinanceMTEB/FinEvaRetrieval` (175) | ✗ **无法分辨 3% 差** | 只当**塌陷冒烟测试**（catastrophic-collapse detector），不当决策指标。见 §5.4 |
| 官方指标不是 nDCG@10 | `mteb/LeCaRDv2` → NDCG@{10,20,30}, P@{5,10}, MAP, Fleiss-KAPPA | ✗ | 报告 LeCaRD 官方 6 个指标 + 我们的统一 nDCG@10 |

> **一句话**：我们的 `evaluation.py` 形状（1 gold + distractors）适合 §4 自建集和
> §5 的显著性检验；接外部 benchmark 时**必须**走 full-corpus 分支，二者不可混用。

---

## 2. 通用能力守门集（Guardrail）

### 2.0 吞吐假设与标定方法

`bge-small-zh-v1.5` 实测 config（已核验）：`hidden=512, layers=4, heads=8, vocab=21128, maxpos=512`。
每 token 前向 ≈ **25.7 MFLOP**（4 层 × 3.15 M 参数 × 2 + attention 项）。

```
texts_per_sec ≈ THROUGHPUT / (25.7e6 × avg_tokens)
```

GB10 规划取值：**THROUGHPUT ≈ 10 TFLOP/s (BF16, 有效值)**，`avg_tokens = 180`（中文 ≈ 1 token/字，截断 256）
→ **≈ 2,200 texts/s**。保守起见本文件所有时间表按 **1,000 texts/s** 估（bf16 + batch 256 + FP32 gather 余量）。

> ⚠️ **务必先跑 5 分钟标定**：
> ```python
> import time, torch
> from de.model import DomainEngram            # 按实际 API 调整
> # 编码 10,000 条随机中文串，报告 texts/s，并把该数字写进 docs/EVAL.md 的运行时间表
> ```
> 本文所有 `≈` 时间都是 **规划值，不是实测值**，标定后请就地更新。

### 2.1 GUARDRAIL 档（推荐，9 个任务，单模型 ≈ 8–10 min）

选点原则：语料 ≤ 30 万、query ≥ 1,000（能做统计）、中英双语覆盖、包含 3 个「弱模型容易塌」的任务。

| # | MTEB task name | HF dataset id（已核验） | corpus | queries | License（HF 声明） | 语料/查询编码时间 |
|---|---|---|---|---|---|---|
| 1 | `SciFact` | `mteb/scifact` | 5,183 | 1,109 | `unknown`（原数据集 CC BY-NC 2.0） | ≈ 6 s |
| 2 | `NFCorpus` | `mteb/nfcorpus` | 3,633 | 3,237 | 未声明 | ≈ 7 s |
| 3 | `ArguAna` | `mteb/arguana` | 8,674 | 1,406 | `cc-by-sa-4.0` | ≈ 10 s |
| 4 | `SciDocs` | `mteb/scidocs` | 25,657 | 1,000 | `cc-by-sa-4.0` | ≈ 27 s |
| 5 | `FiQA2018` | `mteb/fiqa` | 57,638 | 6,648 | `unknown` | ≈ 64 s |
| 6 | `T2Retrieval` 🇨🇳 | `mteb/T2Retrieval` | 118,605 | 22,812 | `apache-2.0` | ≈ 142 s |
| 7 | `DuRetrieval` 🇨🇳 | `mteb/DuRetrieval` | 100,001 | 2,000 | 未声明 | ≈ 102 s |
| 8 | `EcomRetrieval` 🇨🇳 | `mteb/EcomRetrieval` | 100,902 | 1,000 | 未声明 | ≈ 102 s |
| 9 | `CmedqaRetrieval` 🇨🇳 | `mteb/CmedqaRetrieval` | 399,901 | 3,999 | 未声明 | ≈ 404 s |

> 合计编码量 ≈ 721k texts ≈ **12 min/模型**（4 个对照组 A/B/C/D 各跑一次 ≈ 48 min；可接受）。

**追加 3 个「秒级」STS 守门（推荐，成本可忽略）**，检测表征空间是否整体漂移：

| task name | HF id（已核验） | test pairs | license |
|---|---|---|---|
| `STS22` (zh 子集) | `mteb/STSB`（中文翻译版 STS-B） | 1,361 | 未声明 |
| `BQ`（银行问句相似） | `mteb/BQ` | 10,000 | 未声明 |
| `ATEC` | `mteb/ATEC` | 20,000 | 未声明 |

> 主指标：Spearman/Pearson（STS）与 `ndcg_at_10`（Retrieval）。
> **守门阈值（PLAN §5）**：相对组 A，Guardrail 平均 nDCG@10 相对下降 **≤ 1%**。
> 用 §5 的 paired bootstrap 做判定，不看绝对值。

### 2.2 FULL 档（论文/最终报告用，单模型 ≈ 2.5–3 h）

在 GUARDRAIL 之上追加：

| task name | HF id（已核验） | corpus | queries | license | 备注 |
|---|---|---|---|---|---|
| `QuoraRetrieval` | `mteb/quora` | 522,931 | 15,000 | `unknown` | 9 min；近重复检测，对退化敏感 |
| `TRECCOVID` | `mteb/trec-covid` | 171,332 | **50** | 未声明 | ⚠️ 只做冒烟 |
| `Touche2020` | `mteb/touche2020` | 382,545 | **49** | 未声明 | ⚠️ 只做冒烟 |
| `MedicalRetrieval` 🇨🇳 | `mteb/MedicalRetrieval` | 100,999 | 1,000 | 未声明 | 阿里医疗搜索 |
| `CovidRetrieval` 🇨🇳 | `mteb/CovidRetrieval` | 100,001 | 949 | `unknown` | |
| `MMarcoRetrieval` 🇨🇳 | `mteb/MMarcoRetrieval` | 106,813 | 6,980 | 未声明 | 含大量电商 query |
| `VideoRetrieval` 🇨🇳 | `mteb/VideoRetrieval` | 见 ModelScope | 1,000 | — | 可选 |
| STS 补全 🇨🇳 | `mteb/LCQMC` (12,500) / `mteb/PAWSX` (2,000) / `mteb/AFQMC` (3,861) / `mteb/QBQTC` (5,000) | — | — | 未声明 | 全部已核验存在 |
| 分类守门 | `mteb/banking77` (3,076 test) | — | — | 未声明 | 逻辑回归探针，≈ 2 min |

MTEB benchmark 名（已核验自 `mteb/benchmarks/benchmarks/benchmarks.py`）：

```python
import mteb
GUARD = mteb.get_tasks(tasks=["SciFact","NFCorpus","ArguAna","SciDocs","FiQA2018",
                              "T2Retrieval","DuRetrieval","EcomRetrieval",
                              "CmedqaRetrieval","STSB","BQ","ATEC"])
# 或者按 language/task_type 过滤：
mteb.get_tasks(task_types=["Retrieval"], languages=["cmn-Hans"])
```

### 2.3 明确**排除**的任务（GB10 单机不划算 / 被 gate）

| 任务 | HF id（已核验） | corpus | 排除理由 |
|---|---|---|---|
| `MSMARCO` | `mteb/msmarco` | 8,841,823 | ≈ 2.5 h 编码，单任务吃掉整晚 |
| `NQ` | `mteb/nq` | 2,681,468 | ≈ 45 min |
| `FEVER` | `mteb/fever` | 5,416,568 | ≈ 90 min，且 qid 数 123k |
| `HotpotQA` | `mteb/hotpotqa` | 5,233,329 | ≈ 87 min |
| `ClimateFEVER` | `mteb/climate-fever` | 5,416,593 | ≈ 90 min |
| `DBpedia` | `mteb/dbpedia` | 4,635,922 | ≈ 77 min |
| `CQADupstack` | `mteb/cqadupstack` | ≈ 457k × 12 | **HTTP 401（gated）** + 12 个子论坛 |
| `Signal1M` | `mteb/signal1m` | — | **HTTP 401（gated）** |
| `STS12`~`STS17`, `STSBenchmark`(en), `SICK-R`, `AmazonPolarity`, `AmazonCounterfactual`, `ThuNewsClustering*`, `TNews-classification` | `mteb/*` | — | **HTTP 401（gated）**，需 HF token + 手动同意条款 |

> ⚠️ **操作提示**：`mteb/` 组织下有相当一部分数据集是 **gated**（匿名请求返回 401，
> 例如 `mteb/sts12-sts13`、`mteb/sts22`、`mteb/stsbenchmark`、`mteb/sickr`、
> `mteb/amazon-polarity`、`mteb/cqadupstack`、`mteb/signal1m`、`mteb/ThuNewsClusteringS2S`、
> `mteb/TNews-classification`）。
> **执行前先 `huggingface-cli login`**，否则评测跑到一半才 401。
> 本文守门集**刻意全部选用非 gated 任务**，以免复现者卡住。

### 2.4 守门判定协议

1. 每个模型跑一次 Guardrail，导出 `results/{model}/guardrail.json`（含逐 query 分数）。
2. 对每对模型（A vs C、A vs D）做 **paired bootstrap（10,000 次，§5）**，得到 Δ 与 95% CI。
3. **Holm–Bonferroni** 校正跨 9 个任务的 p 值。
4. 额外报告 **win/tie/loss**（逐 query 比较 nDCG@10，tolerance = 1e-9）。
5. 任一任务相对下降 > 1% 且 CI 不含 0 → **判定为塌陷，阻断发布**。

---

## 3. 四大垂域中文检索基准

> 全部 HF id 于 2026-10-10 通过 HF API + datasets-server `/size` 核验；
> 「size」列为 `corpus` / `queries` / `qrels(test)` 三个 config 的真实行数（**不是官方宣称的样本数**，
> 官方宣称的 test query 数常常和实际 config 行数对不上，以 config 为准）。

### 3.1 医疗 🏥

**首选：CMIRB（Chinese Medical Information Retrieval Benchmark）** — 来源论文
*AutoMIR: Effective Zero-Shot Medical Information Retrieval without Instruction*（arXiv:2410.20050），
由其发布 CMIRB：5 类任务、10 个数据集，来自真实中文医疗网站。

| task name（HF id 全部已核验） | qrels(test) | queries | corpus | License |
|---|---|---|---|---|
| `CMIRB/MedExamRetrieval` | 697 | 697 | 27,871 | `apache-2.0` |
| `CMIRB/DuBaikeRetrieval` | 318 | 318 | 56,441 | `apache-2.0` |
| `CMIRB/DXYDiseaseRetrieval` | 1,255 | 1,255 | 54,021 | `apache-2.0` |
| `CMIRB/DXYConsultRetrieval` | 943 | 943 | 12,577 | `apache-2.0` |
| `CMIRB/CSLCiteRetrieval` | 573 | 573 | 36,703 | `apache-2.0` |
| `CMIRB/CSLRelatedRetrieval` | 439 | 439 | 36,758 | `apache-2.0` |
| `CMIRB/IIYIPostRetrieval` | 789 | 789 | 27,570 | `apache-2.0` |
| `CMIRB/CovidRetrieval` | 959 | 949 | （复用 C-MTEB，100,001） | **未声明**（`CMIRB/CovidRetrieval` 无 license tag） |
| `CMIRB/MedicalRetrieval` | 1,000 | 1,000 | 100,999 | **未声明** |
| `CMIRB/CmedqaRetrieval` | 7,449 | 3,999 | **399,901** | **未声明** |

* 数据格式：标准 MTEB 三 config（`default`=qrels, `corpus`, `queries`），`corpus.jsonl` 字段 `{id, text}`。
* 指标：MTEB/BEIR 口径 `ndcg_at_10`；qrels 含多正例（`CmedqaRetrieval` 7,449 条 qrels / 3,999 query ≈ 1.9 正例/query）。
* **推荐子集（≈ 6 min/模型）**：`MedExamRetrieval` + `DuBaikeRetrieval` + `DXYDiseaseRetrieval` +
  `CSLCiteRetrieval` + `IIYIPostRetrieval`（共 ≈ 3,400 query / 200k corpus），
  **单独**跑 `CmedqaRetrieval`（399k 语料 ≈ 7 min）做医疗主力指标。

**补充（阿里系，C-MTEB 源）**：`mteb/MedicalRetrieval`（100,999 / 1,000，未声明 license）、
`mteb/CovidRetrieval`（100,001 / 949，`unknown`）。

> MTEB 官方也把 `CmedqaRetrieval` 放进了 `MTEB(Medical, v1)`
> （已核验：任务列表含 `NFCorpus, TRECCOVID, TRECCOVID-PL, SciFact, SciFact-PL, CUREv1,
> PublicHealthQA, MedicalQARetrieval, MedrxivClustering*, CmedqaRetrieval, CMedQAv2-reranking`）。

### 3.2 金融 💰

**首选：FinMTEB**（arXiv:2502.10990，EMNLP 2025；GitHub `yixuantt/finmteb`，
HF 组织 `FinanceMTEB`，leaderboard `FinanceMTEB/FinanceMTEB_Leaderboard`）。
**64 个金融数据集 × 7 种任务，中英双语**。经核验，其 HF 组织下的**检索类**任务如下：

| task name（HF id 已核验） | qrels | queries | corpus | License |
|---|---|---|---|---|
| `FinanceMTEB/AlphaFin` | 10,000 | 10,000 | 10,000 | **未声明**（AlphaFin 论文，NeurIPS 2024） |
| `FinanceMTEB/DuEE-fin` | 19,906 | 19,906 | 19,906 | **未声明** |
| `FinanceMTEB/FinTruthQA` | 5,999 | 5,999 | 5,999 | **未声明** |
| `FinanceMTEB/FinEvaRetrieval` | 175 | 175 | 175 | **未声明**（`Fin-Eva`，中文金融） |
| `FinanceMTEB/DISCFinLLM-Retrieval` | 168 | 61 | 168 | **未声明** |

非检索但极便宜的中文金融守门任务（同组织，已核验存在）：
`FinanceMTEB/FinSTS` (STS, 370 test)、`FinanceMTEB/FinFE` (test 1,000)、
`FinanceMTEB/FNS` (1,000)、`FinanceMTEB/AlphaFin` 之外还有
`FinanceMTEB/CCKS2019/2020/2022`（各 10 test，样本太少，只能冒烟）、
`FinanceMTEB/HPC3-reranking` (1,000 test)、`FinanceMTEB/MInDS-14-zh` (182)。

> ⚠️ **诚实的结论**：中文金融**检索**公开基准偏弱。最有用的是 **AlphaFin（10k）** 和
> **DuEE-fin（19.9k）**——两者都是「query 与其唯一文档配对」的天然 1-gold 结构，
> **和我们的 `evaluation.py` 形状天然契合**，无需 `full_corpus` 改造。
> `FinEvaRetrieval`（175）与 `DISCFinLLM-Retrieval`（61）**样本量不足**，只能冒烟。
> 金融维度的**结论**应主要由 §4.2 自建集承担，FinMTEB 用来证明「没塌」。

### 3.3 法律 ⚖️

**现状（诚实话术）：中文法律检索没有「大而全」的公开 benchmark，只有两个「小而精」的。**

| 名称 | HF id（已核验） | queries | corpus | License | 官方指标 |
|---|---|---|---|---|---|
| **LeCaRDv2** | `mteb/LeCaRDv2`（同内容镜像 `Anxion/LeCaRDv2`） | **159** | 3,795 | **`mit`** | MTEB `ndcg_at_10`；原始 LeCaRD 官方为 NDCG@{10,20,30} / P@{5,10} / MAP / Fleiss-κ |
| **LeCaRD (v1)** | `github.com/myx666/LeCaRD`（107 query / ~43,000 候选，**不在 HF**） | 107 | ~43,000 | 未声明 | 同上 |
| **LEAD** | `JamesChengGao/LEAD` | **100,060 合成 query-key 对** | 见仓库 `raw_data`（中国裁判文书网） | **`mit`** | LeCaRD / CAIL2022 官方评测脚本 |

**关键事实（必须写进论文 limitation）**：
1. LeCaRDv2 **原论文规模是 800 query / 55,192 候选**，但 MTEB 发布的 `mteb/LeCaRDv2`
   **只保留了 159 query / 3,795 corpus**（已核验 config 行数）。MTEB 的 `MTEB(Law, v1)`
   benchmark 用的就是这个缩水版（已核验任务列表含 `LeCaRDv2`）。
2. LeCaRD 是**分级相关性**（rel ∈ {1,2,3}，只有 rel=3 算「高度相关」），
   **不是 1 gold**。直接套我们的单 gold 分支会失真。
3. LEAD 用 LLM 合成 query，**存在生成器偏差**；训练/评测必须按 query 来源切分，且
   **LEAD 不能同时当训练集和评测集**。

**建议做法**：
* 用 `mteb/LeCaRDv2` 做 **「法律域没有塌陷」的冒烟测试**（159 query 只能检出 >15% 的崩）。
* 法律域的**主要证据**来自 §4.2 的自建法律集，构造 recipe：

```
法律自建集 recipe（LCR-Synth）
  语料源   : 中国裁判文书网公开刑事案件全文（与 LEAD 同源，MIT）或 CAIL2018/CAIL2022
             → github.com/china-ai-law-challenge/cail2018（HF id 已核验存在）
  query    : 从案情（SS/事实认定）里 LLM 抽取「关键事实要素」→ 去掉人名/地名/时间 →
             合成 1–3 句检索式 query（完全复刻 LEAD 的做法，MIT，可直接引）
  candidate: 全量案件库（约 43k–55k）
  label    : 按「罪名 + 主法条 + 刑期」三级匹配（LEAD 的 Similar Case Match 规则）
  切分     : 按 **案件** 切分 train/dev/test（严禁按 query 切，否则同一案件泄漏）
  规模     : test ≥ 2,500 query（全量候选池），用于 §5 的显著性检验
  指标     : MTeB ndcg@10 + LeCaRD 官方 NDCG@{10,20,30}/P@{5,10}/MAP
```

### 3.4 电商 🛒

| 名称 | HF id（已核验） | queries | corpus | License | 备注 |
|---|---|---|---|---|---|
| **EcomRetrieval** | `mteb/EcomRetrieval` / `C-MTEB/EcomRetrieval` | **1,000** | **100,902** | 未声明 | 阿里电商搜索引擎段落检索，C-MTEB 主力电商任务 |
| **MMarcoRetrieval** | `mteb/MMarcoRetrieval` / `C-MTEB/MMarcoRetrieval` | **6,980** | **106,813** | 未声明 | mMARCO 中文，query 大量来自商品搜索 |
| **OnlineShopping**（分类，非检索） | `mteb/OnlineShopping-classification` | 1,000 test | — | 未声明（但 **HTTP 401 gated**） | 只在能 login 时用 |
| **JDReview**（分类） | `mteb/JDReview-classification` | 533 test | — | 未声明（**401 gated**） | 同上 |

> 电商是四个域里**公开评测最成熟**的（阿里直接开源了 E-com 搜索评测集）。
> 但 `EcomRetrieval` 只有 1,000 query → **单独用无法分辨 3% 相对差**（见 §5.4）。
> 解法：与 `MMarcoRetrieval` 合并成 **7,980 query 的电商池**再做显著性检验。

### 3.5 垂域汇总表（可直接贴进论文附录）

| 域 | 首选任务 | queries | corpus | license | 与我们 1-gold 形状的适配 |
|---|---|---|---|---|---|
| 医疗 | `CMIRB/DXYDiseaseRetrieval` + `CMIRB/CmedqaRetrieval` | 1,255 / 3,999 | 54k / 400k | apache-2.0 / 未声明 | 需 full-corpus；qrels 多正例 |
| 金融 | `FinanceMTEB/AlphaFin`、`FinanceMTEB/DuEE-fin` | 10,000 / 19,906 | 10k / 20k | 未声明 | ✅ **天然 1:1，直接可用** |
| 法律 | `mteb/LeCaRDv2`（冒烟）+ 自建 LCR-Synth | 159 / ≥2,500 | 3.8k / ≥43k | mit / 自建 | 分级相关性，需 dedicated 指标 |
| 电商 | `mteb/EcomRetrieval` + `mteb/MMarcoRetrieval` | 1,000 + 6,980 = 7,980 | 100k + 107k | 未声明 | 合并池后满足功效 |

---

## 4. 冷启动 / 动态更新评测设计（核心贡献，无现成 benchmark）

> 对应 PLAN §5 的 **D vs C**、§3.3 的「更新/删除测试」、M4 里程碑。
> 这是本项目**唯一无法用现成 benchmark 证明**的能力，因此必须自建并**完整公开协议**，
> 让第三方能一键复现。

### 4.0 总体设计

```
                      ┌───────────── Registry 状态 ─────────────┐
   WRITE (70%)  ──────┤  cold_start() 写入  → 模型 D
   CTRL-UNW (15%) ────┤  不写           → 检验「写入是否泛化」= 泄漏探针
   RANDOM-NG (15%) ───┤  写随机 N-gram → 检验「随便写点东西是否也涨」
                      └──────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   4.2 剂量-响应曲线      4.3 删除/回滚        4.4 污染/门控抑制
   ΔnDCG@10 vs K         D−P ≈ D（等效检验）   gate 关掉坏记忆
```

**单位：query。** 所有 bootstrap 重采样都以 query 为单位（§5）。
每个 query 必须携带 `set ∈ {WRITE, CTRL_UNW, RANDOM_NG, GENERAL}` 标签。

### 4.1 步骤 A：构造「保证训练时未见」的留出术语集

#### A.1 语料切分（先做，否则后面全不成立）

| 语料 | 内容 | 用途 |
|---|---|---|
| `C_train` | PLAN §3.1 训练三元组的**全部**原文（含 query/pos/neg 文本） | 训练 + 未见性检查的**唯一**参照 |
| `C_future` | **来源或时间上与 `C_train` 不相交**的语料（如不同网站 / 训练截止日之后的文章 / 另一个年份的年报） | 挖掘留出术语 + 构造评测集 |
| `C_eval` | 评测集文档（不参与术语挖掘的统计量） | 最终检索评测 |

> ⚠️ 硬性要求：`C_train` 与 `C_future` 的**文档级**去重（MinHash / SimHash，Jaccard < 0.8 视为重复）。
> 金融年报、法律判例有大量模板化重复，不去重会直接毁掉「未见性」。

#### A.2 候选挖掘 + 6 层硬过滤

挖掘流水线与 PLAN §3.2 完全一致（规则法 + 词典法 + PMI/凝固度），产出候选术语池后逐层过滤：

| 过滤器 | 定义 | 为什么必须 |
|---|---|---|
| **F1 Token 级未现** | 用 `de/registry.py: PhraseMatcher` 对 `C_train` 全量文本跑一遍，term 的 canonical id 元组**命中数 = 0** | 最直接的保证。必须真的跑，不能靠 tokenize 后 `in` 判断 |
| **F2 组成级未现** | term **不是** `C_train` 中任何已注册短语（长度 ≥ \|term\|）的子串 | 挡住「XX型糖尿病」已注册 ⇒「糖尿病」已被间接编码 |
| **F3 Layer-B 未现** | term 的所有 2/3/4-gram 在 `C_train` 的加权 TF-IDF **全部低于**域内 p99 | 挡住「哈希 N-gram 表其实见过」 |
| **F4 Backbone 未掌握（headroom）** | 组 A（冻结 backbone、memory off）下，含该 term 的 query 的 gold **不在 top-10**（或 nDCG@10 < 0.5） | 没有 headroom 就没有提升空间，测出来是 0 不是「无收益」 |
| **F5 频次带** | `df(C_future) ∈ [10, 200]` 且 `df(C_train) = 0` | 排除一次性噪声，也排除 backbone 已经靠上下文猜得出的高频词 |
| **F6 无泄漏** | term 不出现在 `C_eval` 的 **distractor** 侧（否则等于把答案写进题面） | 防止「写入 = 抄答案」 |

#### A.3 三分切分（分层随机）

按 **term token 长度（2–3 / 4–6 / 7+） × 子领域** 分层抽样后切分：

| 集合 | 占比 | 处理 | **期望 ΔnDCG@10** |
|---|---|---|---|
| `WRITE` | 70% | `cold_start()` 写入 registry | **≥ +5% 相对**（PLAN §5 门槛） |
| `CTRL_UNW` | 15% | **不写** | **≈ 0**；若显著 > 0 ⇒ 泄漏，必须报告 |
| `RANDOM_NG` | 15% | 写入等量**随机 N-gram**（长度分布匹配） | **≈ 0**；若 > 0 ⇒ 门槛太松 |

#### A.4 规模（每域）

| 资源 | 数量 | 依据 |
|---|---|---|
| 候选术语池（过完 F1–F6） | **≥ 3,000** | 保证三分后 WRITE 仍有 ~2,100 |
| 写入 term 数 | **≥ 2,000** | §5 功效；也是「运营一次导入一个词表」的合理规模 |
| **query 总数** | **≥ 2,500** | §5：3% 相对 nDCG@10、power 0.8 需 ≈1,300–1,900，留 30% 余量 |
| 其中 `WRITE` 子集 | ≥ 1,750 | |
| 其中含 **2–3 个** 新 term 的 query | ≥ 400 | 组合泛化测试 |
| 语料池（docs） | 10,000–50,000 | 与 `evaluation.py` 的 distractor 规模对齐 |

#### A.5 Query 构造（保证「1 gold + distractors」形状）

```
query_i  = 领域问句模板(症状/条款要素/商品属性)  ⊕  term ∈ WRITE/CTRL/RANDOM_NG
gold_i   = 真实文档片段，**必然包含 term_i**
distract = 同域近邻（共享领域背景但不含 term_i） × 9
约束    : term_i 必须出现在 ≥1 个 distractor 的 *上下文* 里但不能直接区分 gold
          —— 否则模型靠词面即可解决，测不到「记忆」
```

> 构造完成后，用组 A 跑一遍作为 **A'（baseline-prime）**；`WRITE` 子集上 A' 的
> nDCG@10 应落在 **0.30–0.60** 区间。太高说明任务太简单，太低说明噪声太大。

### 4.2 步骤 B：测量「写入后**不重训**的 Δ」（D vs C）

**执行序列（严格按序，每步只做 Registry 写入）：**

```
C  = 训完的 DE（Layer-A + Layer-B + 融合层都定型）
D  = C + cold_start(term, contexts)  ∀term ∈ WRITE      # 零梯度步，闭式解
D' = C + cold_start(term, [])       ∀term ∈ WRITE      # 对照：不给 support context
```

> `de/cold_start.py: solve_slot_closed_form` 已经是闭式岭回归解（无梯度），
> `refine_slot`（step 3，20 步梯度）是**可选**的。**主实验必须报两者**：
> `D`（闭式，O(1)/term）与 `D'`（闭式 + 20 步单行微调）。

#### 4.2.1 必报指标

| # | 指标 | 期望 | 统计方法 |
|---|---|---|---|
| M1 | `Δ_nDCG@10` on `WRITE` 子集 | **≥ +5% 相对** | paired bootstrap 10k，95% BCa CI |
| M2 | `Δ_nDCG@10` on `CTRL_UNW` | ≈ 0，CI 含 0 | paired bootstrap |
| M3 | `Δ_nDCG@10` on `RANDOM_NG` | ≈ 0，CI 含 0 | paired bootstrap |
| M4 | `Δ` on Guardrail（§2.1 9 任务） | 相对下降 ≤ 1% | paired bootstrap + Holm 校正 |
| M5 | **剂量-响应曲线**：K ∈ {0, 25, 50, 100, 200, 400, 800, 1600, 2100} 逐点写 K 个 term 后测 `Δ_nDCG@10(WRITE)` | 单调饱和 | 拟合饱和曲线，报 **K\*（达到 90% 最大增益所需 term 数）**与初始斜率 |
| M6 | **写入延迟**：per-term p50 / p95 / p99（闭式解）；batch 导入 2,100 term 的总墙钟 | 毫秒级 | 计时 |
| M7 | **Time-to-Update**：从 `registry.add()` 到该 term 可被服务（matcher 重建完成）的端到端延迟 | ms 级 | 计时（注意 `PhraseMatcher.maybe_rebuild` 的 O(T·\|lengths\|) 代价） |
| M8 | **组合泛化**：含 2–3 个新 term 的 query 子集上的 `Δ` | > 0 | paired bootstrap |
| M9 | **`D' vs D`**：闭式 vs 闭式+20 步微调 | `D' ≥ D` | paired bootstrap |
| M10 | **零支持对照**：`contexts=[]` 的纯随机初始化写入 | ≈ 0 或 < 0 | paired bootstrap（证明 aggregate-init 才是关键，而非「写进去就行」） |
| M11 | **槽位健康**：写入后 `len(registry)`、`n_used_slots`、内存占用 | 线性增长、无泄漏 | 断言 |

#### 4.2.2 「不重训」的硬证据

必须附上：

* **参数量快照**：写入前后 `sum(p.numel() for p in model.parameters())` **完全相等**；
  backbone `requires_grad` 全 False；Layer-B 表 `requires_grad=False`。
* **梯度快照**：`loss.requires_grad == False`（闭式解路径）。
* **写入前后 hash**：backbone `state_dict` 的 SHA256 不变（只有 `registry_table.emb` 的新行变化）。
* **训练器未实例化**：整个 D 组评测期间不创建 `torch.optim`。

### 4.3 步骤 C：删除 / 回滚测试（rollback）

#### C.1 构造 4 类「坏条目」

| 代号 | 构造方式 | 数量 | 模拟的真实事故 |
|---|---|---|---|
| **P1 错义写入** | term 在**语义不符**的 context 里 solve（如同名异义、跨域同名） | 200 | 运营录入时选错语境 |
| **P2 随机向量** | `registry_table.reset_slot()` 后不 solve，保留 N(0, σ²) | 200 | 上游 embedding 故障 |
| **P3 缩放污染** | 真解 × s，s ∈ {2, 5, 10} | 100（每个 s 一组） | 写入时范数未归一 |
| **P4 近邻对抗** | 对 gold term 做 1-char 替换/插入/删除生成伪 term，写入**另一个 term** 的向量 | 100 | **typo-squatting / 投毒** |

#### C.2 执行序列

```
S0  测 nDCG@10(D)                                ← 基准
S1  写入 K 个坏条目，K ∈ {1, 10, 50, 100, 200, 500}
      → 测 nDCG@10(D+P)  以及  门控统计（见 §6）
S2  registry.delete(phrase) ∀坏条目               ← tombstone + free-list，O(1)，不重训
      → 测 nDCG@10(D−P)
S3  recovery_gap = nDCG@10(D−P) − nDCG@10(D)
```

#### C.3 通过标准

| 标准 | 阈值 | 检验方法 |
|---|---|---|
| **回滚恢复** | `\|recovery_gap\| < 1.0` 绝对 nDCG 点，且 95% CI **含 0** | **TOST 等效性检验**，equivalence margin δ = 1% 绝对点；再补 paired bootstrap |
| **部分回滚** | 删 50% → 恢复量单调（报告曲线） | 同上 |
| **slot 复用安全** | 释放的 slot 上写入新 term 后，**其它所有条目的 slot_id / 向量比特位 / 匹配结果完全不变** | 逐条 diff `registry.entries` + `registry_table.emb.weight` 前后的哈希 |
| **matcher 缓存一致性** | 删除后必须 `sync_matcher()`；断言未调用时 matcher 仍返回旧 slot（**这是回归测试**，因为 `PhraseMatcher._built_version` 缓存了旧索引） | 单测 |
| **回滚延迟** | 删除 600 个条目 + matcher 重建的墙钟 | 计时 |

> ⚠️ **本仓库特有的实现风险（必须写进测试）**：
> `PhraseMatcher._rebuild` 只在 `registry.version` 变化时触发；
> `cold_start()` 里调了 `sync_matcher()`，但**外部直接调 `registry.delete()` 不会**。
> C.3 的最后一条就是专门抓这个 bug 的。

### 4.4 步骤 D：污染 / 对抗测试（验证门控抑制）

> 依据：PLAN §1.3（Engram 原文 §6.5 的门控是**选择性**的，矛盾时 α→0）。

#### D.1 门控层（per-term）

对每个污染 term `t`，构造 **≥ 200 对 matched / mismatched context**：

* **matched ctx**：term 出现且**语境与写入向量一致**（对应 P1 的正确语境）。
* **mismatched ctx**：term 出现但**语境相矛盾**（或完全无关的句子）。

| 指标 | 定义 | 期望 |
|---|---|---|
| `α_match` | term-final 位置的 α（matched ctx 平均） | 高 |
| `α_mismatch` | 同上（mismatched ctx 平均） | 显著低 |
| **SR（抑制比）** | `SR(t) = 1 − α_mismatch / α_match` | `SR > 0`；报 `#{t : SR > 0.5} / N` |
| **Context-AUROC(t)** | term 内 α 区分 match/mismatch 的 AUROC | 中位数 ≥ 0.7，IQR 报告 |

#### D.2 检索层（整体）

| 指标 | 定义 | 期望 |
|---|---|---|
| **hijack rate** | `(D+P)` 下 top-1 文档从 gold 改变的 query 占比 | ≤ 1%（K = 10×正常 term 数） |
| **robustness** | `nDCG@10(D+P) / nDCG@10(D)` | ≥ 0.99 |
| **污染剂量曲线** | K ∈ {0, 10, 100, 1k, 5k} 时 nDCG@10(D+P) | 平坦（门控选择性 ⇒ 应平坦） |
| **P4 特化** | 100 个 typo 伪 term 存在时，`WRITE` 子集 nDCG@10 相对 S0 的下降 | ≤ 1% |
| **容量压力** | 写满 `cfg.max_registry_slots` 后再写 → 期望抛 `RuntimeError("phrase registry exhausted")` | 单测覆盖 |

#### D.3 污染测试样本量

| 资源 | 数量 | 依据 |
|---|---|---|
| 污染 term（P1+P2+P4） | 500 | ≥ query 数 / 5，使「命中率」有意义 |
| 每个污染 term 的 matched/mismatched ctx 对 | ≥ 200 | 使 α 均值的 SE ≈ σ_α/√200 ≈ 0.03，可分辨 0.1 的 SR |
| ctx 总数 | 500 × 200 × 2 = **200,000** | 编码 ≈ 200k texts ≈ 200 s |
| 检索层复用 §4.1 的 ≥ 2,500 query 集 | 同上 | 保持 paired |

### 4.5 冷启动评测的「完整结果表」模板

| 配置 | nDCG@10 (WRITE) | Δ vs C | 95% CI | nDCG@10 (CTRL) | Δ | nDCG@10 (RANDOM) | Δ | Guardrail Δ |
|---|---|---|---|---|---|---|---|---|
| A. Baseline (frozen) | | — | — | | — | | — | — |
| B. DE-RandomMem | | | | | | | | — |
| C. DE (full) | | — | — | | — | | — | — |
| **D. DE + dynamic write** | | | | | | | | — |
| D' = D + refine_slot | | | | | | | | — |

---

## 5. 统计严谨性

### 5.1 为什么用 paired bootstrap 而不是 paired t-test

| 维度 | paired t-test | **paired bootstrap（推荐）** |
|---|---|---|
| 分布假设 | nDCG@10 差值需近似正态 | **无假设** |
| 离群点 | 1 个 query 从 rank 1 → rank 10000 会主导结果 | **稳健** |
| 指标通用性 | 只对可求均值的指标 | 任意指标（Recall、MRR、nDCG、hijack rate…）都能 bootstrap |
| 分层 | 无法直接做分位数 bootstrap | 可对子集分别 bootstrap |
| 可解释 | 给 t 值，无分布 | 给 **win/tie/loss + Δ 的经验分布** |

> 唯一例外：**Recall@k / top-1 翻转这类二值指标，McNemar 检验最有力**（见 5.3），可与 bootstrap 并列报告。

### 5.2 需要多少 query 才能检出 ~3% 相对 nDCG@10 提升（p<0.05）

规划公式（配对、双侧 α=0.05）：

```
n = (z_{1-α/2} + z_{1-β})² · σ_d² / Δ²
```

其中 `Δ = 0.03 × nDCG_baseline`（绝对提升），`σ_d = sd_i(nDCG_i^A − nDCG_i^C)`（**逐 query 差值的标准差**）。

`σ_d` 经验关系：`σ_d ≈ σ_nDCG · sqrt(2(1−ρ))`，`ρ` 是两系统的逐 query 相关。
取 `σ_nDCG ≈ 0.40`（典型单 gold、候选池 ≫10 的 nDCG@10 分布）。

#### 功效表（power = 0.80，`z=1.96+0.8416=2.8016`，`z²=7.849`）

| baseline nDCG@10 | Δ（绝对） | σ_d=0.20 | σ_d=0.25 | **σ_d=0.30** | σ_d=0.35 | σ_d=0.40 |
|---|---|---|---|---|---|---|
| 0.50 | 0.015 | 1,395 | 2,180 | **3,139** | 4,274 | 5,581 |
| 0.60 | 0.018 | 969 | 1,514 | **2,180** | 2,968 | 3,876 |
| **0.65**（规划值） | 0.0195 | 826 | 1,290 | **1,858** | 2,528 | 3,303 |
| 0.70 | 0.021 | 712 | 1,112 | **1,602** | 2,181 | 2,848 |
| 0.80 | 0.024 | 545 | 852 | **1,226** | 1,669 | 2,180 |

#### power = 0.90（`z=1.96+1.2816`，`z²=10.507`），baseline 0.65

| σ_d=0.20 | 0.25 | 0.30 | 0.35 | 0.40 |
|---|---|---|---|---|
| 1,105 | 1,727 | **2,487** | 3,385 | 4,421 |

#### 📌 结论

> **规划值：`σ_d = 0.30`、baseline nDCG@10 ≈ 0.65、power 0.80 → 需要 ≈ 1,860 query；
> 取整后每域评测集设计为 ≥ 2,500 query（留 ~35% 余量给过滤损耗与离群）。**
> power 0.90 需 ≈ 2,490（设计取 3,000）。

**同时必须诚实说明 3 条推论：**

1. **3% 是很小的效应量。** 若 `ρ` 较低（两系统逐 query 相关 < 0.7），`σ_d` 会升到 0.31+，
   需求逼近 2,500。**强烈建议把主要终点设为 5% 相对提升**（Δ=0.0325，n 减少约 30%），
   3% 作为「有信号」的参考线。
2. **预注册（pre-register）**：在跑实验前把 `n`、`K` 网格、指标、bootstrap 次数、校正方式写进
   `docs/EVAL.md` 本文件并 commit。任何事后改 n 都必须记为探索性分析。
3. **σ_d 必须实测**：第 1 次跑完 A vs C 后立刻算 `σ_d = std(ndcg_A − ndcg_C)`，
   把真实值回填到本表并重算所需 n。**不要用假设值做最终报告。**

### 5.3 Recall@k 翻转：McNemar 检验（更省样本）

Recall@k / top-1 是二值的。令 `b` = 仅 A 正确、`c` = 仅 C 正确（discordant pairs），
`n_bc = (z_{α/2} + z_β)² (b + c) / Δ²`。

* 假设 10% 的 query 发生翻转（`b+c = 0.10`），要检出 **5pp 绝对** 的 Recall@1 差异：
  `n = 7.849 × 0.10 / 0.0025 = **314**`。
* 即 **≈ 314 query 就能检出 5pp 的 Recall@1 变化**，比 nDCG@10 便宜 6 倍。

> **实践建议**：用 McNemar 做 **Recall@1 的快速守门**（n≥600 足够），
> 用 paired bootstrap 做 **nDCG@10 的正式判定**（n≥2,500）。

### 5.4 哪些基准「样本量不足」——必须写进 limitation

| 任务 | query 数 | 能分辨的相对差（σ_d=0.30, power 0.8, baseline 0.65） | 定位 |
|---|---|---|---|
| `FinanceMTEB/DISCFinLLM-Retrieval` | 61 | ≈ 32% | **冒烟** |
| `mteb/trec-covid` | 50 | ≈ 39% | **冒烟** |
| `mteb/touche2020` | 49 | ≈ 40% | **冒烟** |
| `mteb/LeCaRDv2` | 159 | ≈ 22% | **冒烟** |
| `FinanceMTEB/FinEvaRetrieval` | 175 | ≈ 20% | **冒烟** |
| `CMIRB/DuBaikeRetrieval` | 318 | ≈ 11% | 弱 |
| `CMIRB/CSLRelatedRetrieval` | 439 | ≈ 8% | 弱 |
| `mteb/EcomRetrieval` | 1,000 | ≈ 3.5% | 勉强 |
| **`§4 自建集`** | **≥ 2,500** | **≈ 2.2%** | ✅ **主战场** |

> **原则：小样本任务只回答「有没有塌」（>15% 级），不回答「涨没涨 3%」。**

### 5.5 Bootstrap 实施规范

```python
def paired_bootstrap(scores_a: np.ndarray, scores_b: np.ndarray,
                     B: int = 10_000, seed: int = 0, alpha: float = 0.05):
    """scores_*: 逐 query 的指标值（同一顺序、同一 query 集）。返回 (delta, ci_lo, ci_hi, p)."""
    rng = np.random.default_rng(seed)
    n = len(scores_a)
    d = scores_a - scores_b
    idx = rng.integers(0, n, size=(B, n))          # 重采样【query】，绝不重采样 document
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100*alpha/2, 100*(1-alpha/2)])
    # 双侧 p 值：bootstrap 分布跨过 0 的比例，向 0 翻倍
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return d.mean(), lo, hi, min(p, 1.0)
```

| 参数 | 取值 | 理由 |
|---|---|---|
| **B（重采样次数）** | **10,000**（正式）/ 2,000（快速迭代） | B=10,000 时 p 的蒙特卡洛 SE ≈ `sqrt(0.05·0.95/10000)=0.0022`，最小可分辨 p ≈ 0.0005，对 p<0.05 绰绰有余；B 再大无收益 |
| CI 类型 | **BCa**（正式）/ percentile（快速） | BCa 对偏度更稳健 |
| 重采样单位 | **query** | query 是独立观测单位；document 之间强相关，重采样 document 会严重高估显著性 |
| 随机种子 | 固定 0 | 可复现 |
| 多重比较 | **Holm–Bonferroni**（跨任务）/ **Benjamini–Hochberg**（探索性） | 9 个守门任务 × 多个指标必然有假阳性 |

> **经验法则**：有效的不是 B，而是 **query 数**。B=10,000 × n=2,500 是 2.5 亿次索引操作，
> numpy 下约 2 秒；瓶颈永远在 query 数上。

### 5.6 必报的效应量（不只报 p 值）

* `Δ`（绝对）与其 **相对提升 %**
* **Cohen's d_z = Δ / σ_d**
* **win/tie/loss** 计数（tolerance 1e-9）
* **AUC-style 提升**：PRNG 意义上的 `P(C > A)`

---

## 6. 门控选择性定量指标（复刻 Engram §6.5，但不只做热力图）

> Engram 原文只给了 α 的可视化。**热力图不可证伪**，必须给出标量指标 + 显著性。

### 6.0 我们白送的标签：位置级二分类

`PhraseMatcher.match()` 的输出 `slots[t, 0] != 0` **精确等于**「位置 t 是某个已注册短语的末位」。
这给了我们一个**零标注成本**的、逐 token 的二分类标签：

```
正样本 y=1 : 位置 t 是某个 domain term 的末 token（registry slot != 0）
负样本 y=0 : 位置 t 是有效内容 token 且 slot == 0（排除 [CLS]/[SEP]/[PAD]）
打分 s_t   : α_t（上下文感知门控的输出）
```

### 6.1 主指标：**Term-Final AUROC**

```
AUROC = P(s_pos > s_neg) + 0.5·P(s_pos = s_neg)
```

| 报法 | 说明 |
|---|---|
| 点估计 + 95% CI | CI 用 **bootstrap（B=2,000，按 *句子* 重采样**，不是按 token——同一句内的 token 不独立） |
| **零假设基线** | **标签置换**：把正/负标签在句内随机打乱 200 次，报置换分布的 AUROC 均值 ± 2σ。实测 AUROC 必须显著高于该基线 |
| 分层报告 | in-domain / out-of-domain 文本分别报（门控应该对 domain 文本更敏感） |

> **通过门槛**：AUROC ≥ 0.70 且置换基线 ≈ 0.50，95% CI 下界 > 0.55。

### 6.2 配套指标（全部必报，构成完整证据链）

| # | 指标 | 定义 | 为什么 |
|---|---|---|---|
| G1 | **Term-Final AUROC** | §6.1 | 主指标 |
| G2 | **Matched-Pair Accuracy + McNemar** | 对每个 term 出现，从**同一句**随机抽 1 个非 term 位置配对；统计 `α_term > α_other` 的比例，用 **McNemar 检验** | 消除句子难度的混淆（句子级配对） |
| G3 | **Gate-Mass Enrichment Ratio** ⭐ | `ER = [Σ_{t∈term} α_t·‖ỹ_t‖] / [Σ_{all t} α_t·‖ỹ_t‖] ÷ (|term| / |tokens|)` | **直接对应 Engram §6.5 的「记忆质量集中在术语位置」**，比热力图强一个数量级。ER ≥ 1 说明记忆确实集中在术语上 |
| G4 | **Point-biserial / rank-biserial 相关** | `corr(slot_hit ∈ {0,1}, α_t)` | G1 的效应量版本 |
| G5 | **Cliff's δ** | `α_term` vs `α_other` 的非参效应量 | 对离群稳健 |
| G6 | **分类别均值表** | `α` 在 4 类位置上的均值：{registry-hit, hash-only, in-domain plain, out-of-domain plain} | 证明门控不是「恒开/恒关」 |
| G7 | **门控动态范围** | `p95(α) − p5(α)`；以及 `frac(α>0.5)` | 门控若坍缩到常数，G1 会假性变高但 G7 会暴露 |
| G8 | **Context-AUROC / SR** | §4.4 D.1，区分 matched vs mismatched 语境 | 把 §6.1 从「位置级」升级到「语境级」选择性 |

### 6.3 采样与规模

| 资源 | 数量 |
|---|---|
| 评测文本 | ≥ 2,000 条（1,000 in-domain + 1,000 out-of-domain 对照） |
| 有效 token 位置总数 | ≥ 400,000 |
| 正样本（term 末位）数 | ≥ 20,000 |
| 置换零假设次数 | 200 |
| 门控统计的采集方式 | 在 `de/model.py` 的融合层加 `collect_gates()` hook，导出 `(position, α, slot_hit, token_id, text_id)` |

### 6.4 「塌缩检测」三件套（防止指标被作弊）

门控可能通过**恒开**（α≡1）或**随机噪声**骗过 AUROC。必须同时满足：

1. `G7 动态范围 > 0.3`（α 不能是常数）
2. `G3 ER ≥ 1.2`（记忆质量要真的富集在术语上）
3. 标签置换零假设 ≈ 0.50（排除标签泄漏——例如「slot≠0」与 α 在实现里共用同一个变量）

---

## 7. 复现清单（Runbook）

### 7.1 前置

```bash
pip install -U "mteb>=1.14" sentence-transformers pytrec_eval-terrier
huggingface-cli login          # 必需：mteb 组织下部分数据集 gated（见 §2.3）
```

### 7.2 执行顺序（对应 PLAN 里程碑）

```
M2  A 组    : §1.2 full_corpus 改造 → §2.1 Guardrail(9) → §3 主域检索 → 落 results/A.json
M3  C 组    : 同上 → 落 results/C.json
             → §6 门控选择性报告 G1–G8
M4  D 组    : §4.1 建集（WRITE/CTRL_UNW/RANDOM_NG 各 ≥ 2,500 / 375 / 375 query）
             → §4.2 剂量曲线 K ∈ {0,25,50,100,200,400,800,1600,2100}
             → §4.3 回滚 S0–S3（含 slot 复用 diff + matcher 缓存回归测试）
             → §4.4 污染 P1–P4 + hijack rate + Context-AUROC
             → 全程 paired bootstrap B=10,000
M5  报告    : §4.5 结果表 + §2.4 守门判定 + §5.4 limitation 表 + §5.6 效应量
```

### 7.3 产物目录

```
results/
  A/{guardrail.json, domain.json, per_query.parquet}
  C/{...}
  D/{...}
  coldstart/{write.tsv, dose_response.csv, rollback.json, pollution.json}
  gates/{auroc.json, enrichment.json, per_position.parquet}
  bootstrap/{B10000_seed0.npz}       # 存全部 Δ 分布，便于事后画 CI 而不重编码
```

`per_query.parquet` 必须包含：`task, split, qid, gold_id, rank, ndcg10, recall1/5/10, mrr, set ∈ {WRITE,CTRL_UNW,RANDOM_NG,GENERAL}`
—— §5 的所有 bootstrap 都在这个文件上离线跑，**不必重跑模型**。

---

## 8. 已核验 / 未核验清单（诚实声明）

### ✅ 已核验（HF API + datasets-server/size，2026-10-10）

`mteb/scifact` · `mteb/nfcorpus` · `mteb/arguana` · `mteb/scidocs` · `mteb/fiqa` · `mteb/quora` ·
`mteb/trec-covid` · `mteb/touche2020` · `mteb/banking77` · `mteb/msmarco` · `mteb/nq` · `mteb/fever` ·
`mteb/hotpotqa` · `mteb/climate-fever` · `mteb/dbpedia` · `mteb/biosses-sts` ·
`mteb/T2Retrieval`(apache-2.0) · `mteb/DuRetrieval` · `mteb/EcomRetrieval` · `mteb/MedicalRetrieval` ·
`mteb/CovidRetrieval` · `mteb/CmedqaRetrieval` · `mteb/MMarcoRetrieval` · `mteb/VideoRetrieval` ·
`mteb/ATEC` · `mteb/BQ` · `mteb/LCQMC` · `mteb/PAWSX` · `mteb/STSB` · `mteb/AFQMC` · `mteb/QBQTC` ·
`mteb/LeCaRDv2`(mit) · `Anxion/LeCaRDv2`(mit) · `mteb/CLSClusteringS2S` ·
`CMIRB/*` 全部 10 个 · `C-MTEB/*`（40 个中文字段级 config） ·
`FinanceMTEB/*` 全部 64 个 id 已列出核验 · `JamesChengGao/LEAD`(mit) ·
`china-ai-law-challenge/cail2018`（license `unknown`） ·
`BAAI/bge-small-zh-v1.5`(512/4层) · `BAAI/bge-base-zh-v1.5`(768/12层) · `intfloat/multilingual-e5-small`(384/12层) · `BAAI/bge-m3`

### ❌ 核验结果 = HTTP 401（gated，需 HF token + 同意条款）

`mteb/cqadupstack` · `mteb/signal1m` · `mteb/sts12-sts13` · `mteb/sts14-sts15` · `mteb/sts16-sts17` ·
`mteb/sts22` · `mteb/stsbenchmark` · `mteb/sickr` · `mteb/amazon-polarity` · `mteb/amazon-counterfactual` ·
`mteb/ThuNewsClusteringS2S` · `mteb/ThuNewsClusteringP2P` · `mteb/TNews-classification` ·
`mteb/IFlyTek-classification` · `mteb/waimai-classification` · `mteb/OnlineShopping-classification` ·
`mteb/JDReview-classification`

### ⚠️ UNVERIFIED（**不要在论文中当作已核验引用**）

| 条目 | 说明 |
|---|---|
| `mteb/VideoRetrieval` 的 corpus 行数 | 本轮未取 size 数据；只有 `C-MTEB` 侧宣称的 1,000 test queries |
| `C-MTEB/*` 各任务的 license | HF 上**未声明**；原 ModelScope 页的 license 需另行确认 |
| `CMIRB/CmedqaRetrieval` / `CMIRB/MedicalRetrieval` / `CMIRB/CovidRetrieval` 的 license | HF 无 tag；CMIRB 源自 C-MTEB（`CMIRB` 版仅 `MedExam`/`DuBaike`/`DXY*`/`CSL*`/`IIYIPost` 6 个为 `apache-2.0`） |
| `FinanceMTEB/*` 全部 license | HF 上**全部未声明**；FinMTEB 论文未逐数据集标 license |
| `LeCaRD v1`（107 query）的 HF id | **不存在 HF 官方版本**，仅 `github.com/myx666/LeCaRD` |
| LEAD 的 test 划分 | 仓库未提供官方 train/test 切分，**需自行定义**并公开 |
| `mteb/cqadupstack` 的 corpus 大小 | gated，未核验 |
| GB10 在本任务上的实测吞吐 | **全部时间为规划值**，必须按 §2.0 标定后回填 |
| Engram (arXiv:2601.07372) §6.5 的原始 α 数值 | 依 PLAN.md 的描述（术语/实体位置 α 显著更高）；**未独立核验论文原文** |

---

## 9. 关键引用

* Muennighoff et al., *MTEB: Massive Text Embedding Benchmark*, arXiv:2210.07316 / EACL 2023.
* Chen et al., *C-MTEB: Chinese Massive Text Embedding Benchmark*, ACL 2024.
* Enevoldsen et al., *MMTEB: Massive Multilingual Text Embedding Benchmark*, arXiv:2502.13595.
* **AutoMIR**（CMIRB 出处）, arXiv:2410.20050.
* Tang & Yang, *FinMTEB: Finance Massive Text Embedding Benchmark*, arXiv:2502.10990, **EMNLP 2025**；
  GitHub `yixuantt/finmteb`（**仓库未声明 license** — 已核验，引用前需确认）；leaderboard `FinanceMTEB/FinanceMTEB_Leaderboard`.
* Li et al., *LeCaRDv2: A Large-Scale Chinese Legal Case Retrieval Dataset*, arXiv:2310.17609.
* Ma et al., *LeCaRD: A Legal Case Retrieval Dataset for Chinese Law System*, SIGIR 2021（评测脚本 `github.com/myx666/LeCaRD/metrics.py`）。
* Gao et al., *Enhancing Legal Case Retrieval via Scaling High-quality Synthetic Query-Candidate Pairs (LEAD)*,
  EMNLP 2024, arXiv:2410.06581（数据 `JamesChengGao/LEAD`, MIT）。
* Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*, NeurIPS 2021.
* **Cheng et al., *Conditional Memory via Scalable Lookup* (Engram), arXiv:2601.07372v2** — §6.5 门控选择性；CC BY 4.0。
* NVIDIA, *DGX Spark / GB10 Grace Blackwell Superchip* 规格页（128 GB LPDDR5X、273 GB/s、≤1 PFLOP FP4）。
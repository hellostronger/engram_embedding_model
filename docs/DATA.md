# DATA.md — Domain Engram 训练数据规格（Training Data Specification）

> 状态：**已核验规格**（verified specification）
> 核验日期：2026-10-10
> 核验方式：全部 HF dataset id 通过 `huggingface.co/api/datasets/{id}` 与
> `datasets-server.huggingface.co/{first-rows,rows,splits}` 实测返回确认存在；
> 字段名（field names）一律来自 `first-rows` 的实测 schema，**不是抄 README 的推测**。
> 未能核验的一律显式标注 `UNVERIFIED`。
>
> 环境约束：沙箱代理屏蔽 huggingface.co 下载。本文件只给**规格**，不含下载代码。
>
> 相关文档：[PLAN.md](PLAN.md)（总体方案）。代码契约见 `de/data.py`（JSONL loader + collator）。

---

## 0. 阅读约定

### 0.1 术语
| 符号 | 含义 |
|---|---|
| **Anchoring / 锚定** | 通用域（general-domain）数据，作用是防止垂域适配导致通用检索能力塌陷 |
| **In-domain raw / 域内原文** | 领域原始文本（摘要、年报、判决书、商品描述），用于挖 N-gram + 领域术语 → 喂 Layer-B / Layer-A |
| **Triplet / 三元组** | `(query, positive, hard-negative[])`，直接进 InfoNCE |
| **Registry term / 注册词** | 进 Layer-A CRUD 显式短语表的实体/术语字符串 |
| **Cold start / 冷启动** | 训练截止时间之后才出现的实体（新品型号、新法规、新 ICD 码、新证券简称） |

### 0.2 Token 估算规则（本文统一使用，可复现）
不依赖下载，按下式估算，**所有 token 数均为估算值**：

```
tokens ≈ download_MB × 0.26 × 1e6      # ~4 bytes/token，Parquet 压缩后的经验值
```

误差范围经验值 ±25%。实际计数请在子采样后用目标 tokenizer（BGE-small-zh 用
`bert-base-chinese` 词表）跑一次精确统计并回填本文表格。

### 0.3 许可证红线
- `ODC-By v1.0`（FineWeb / FineWeb-2）：可用，**但受 CommonCrawl ToU 约束**，需归因。
- `CC-BY-SA-3.0 / GFDL`（Wikipedia）：**copyleft**，改了要开源。评测/内部检索可接受，做商业模型权重需注意。
- `CC-BY-NC-*`：**禁止商用**。仅可用于内部研究/评测，**不要进训练语料**。
- `pile-of-law`：`CC-BY-NC-SA-4.0` → **仅评测，禁止训练**。
- **SNOMED CT**：IHTSDO 许可，**美国境内免费、非美国需授权**。禁止进商业训练集。
- `gated: auto` / `gated: manual` 的数据集需要先在网页接受条款，见各条目标注。

---

# 第一部分：通用域锚定语料（所有 4 个领域共用）

作用：(a) Stage 1 对比学习的通用 query→passage 骨架；(b) 领域 N-gram 挖掘时的
"低普适性背景"（用于算领域词的 TF-IDF 分母）。

| # | HF dataset id | 规模 | License | 为什么选它 / 如何取 5–20M pairs |
|---|---|---|---|---|
| G1 | `Shitao/bge-m3-data` | **22.9 GB 单个 `bge-m3-data.tar.gz`**（实测 22,870 MB），172,094 行/文件条目，`size_categories:100K<n<1M` | **UNVERIFIED**（卡片未声明；内容继承 MS MARCO / NQ / mMARCO / MIRACL 等各自条款，混合体无法给出单一许可） | BGE-M3 的官方微调数据，已含 zh(en)：`msmarco`, `nq`, `hotpotqa`, `triviaqa`, `squad`, `coliee`, `pubmedqa`, `nli`, `dureader`, `mmarco-zh`, `t2ranking`, `law-gpt`, `cmedqav2`, `nli-zh`, `lecardv2`, `mtydi`, `miracl`, `mldr`。**注意：是一个 22.9 GB 的 tar.gz，无法按源单独下载**——要么全下，要么不用。**建议：GB10 最小配方不使用；扩规模时一次性抽取成 per-source jsonl 再删 tar。** |
| G2 | `HuggingFaceFW/fineweb` | 15T tokens（全量，45 TB）；ODC-By v1.0；`size_categories:10B<n<100B` | `odc-by` | 英文通用 web 原文，**字段带 `date` + `dump`**（CC 快照 ID）→ 冷启动时间切分的主源（见 §5）。**取子集方式**：按 `dump` 过滤到 2–3 个 CommonCrawl 快照，`date < T0`，`stream=True` 抽 N=3M docs → 3M docs × ~450 tok ≈ **1.3B tokens**，直接远超 20M 对需求。若只需 5–20M 对，抽 200K docs + 每个 doc 自生成 50 个 query 变体即可。 |
| G3 | `HuggingFaceFW/fineweb-2` | 15T tokens（全量 44 TB）；ODC-By；1,870 个语言 config | `odc-by` | **多语言版本，含中文。** ⚠️ **实测纠正：config 名是 `cmn_Hani`（不是 `cmn_Hani_train`），splits = `train` / `test`；`train` 实测 636,058,984 行，`test` 33,296 行**。**实测 schema（`first-rows`，config=`cmn_Hani`, split=`train`）**：`text, id, dump, url, date, file_path, language, language_score, language_script, minhash_cluster_size, top_langs` —— **`dump` 与 `date` 都在**，这是"中文 + 时间戳"冷启动切分的关键源。取子集：只下 `cmn_Hani` 的少数 parquet shard（官方仓库该目录巨大，建议用 `huggingface_hub` 的 allow_patterns 逐文件拉）。 |
| G4 | `HuggingFaceFW/finewiki` | 全库 184.7 GB parquet（实测）；zh 5.1 GB / en 35.1 GB（官方表）；`size_categories:10M<n<100M` | `cc-by-sa-4.0` + `gfdl` | **Wikipedia 带时间戳版**。实测 schema（`first-rows`）：`text, id, wikiname, page_id(int64), title, url, date_modified, in_language, wikidata_id, bytes_html, wikitext, version(int64), infoboxes, has_math(bool)`。按语言分 config：`en/train`、`zh/train`、`zh_classical`、`zh_min_nan`、`zh_yue`。**2025-08 dump，只保留每个 page 的最新 revision**——`date_modified` 是最后修改时间，不是创建时间（见 §5.3 的补偿方案）。 |
| G5 | `wikimedia/wikipedia` | `size_categories:10M<n<100M`（全语言合计）；`20231101.en` ≈ 6.4M articles（官方 card 自述） | `cc-by-sa-3.0` + `gfdl` | 干净、稳定的通用中文/英文锚定文本；**注意：无逐条时间戳字段**（只有 dump 级 `20231101`），**不能做时间切分**，只能做通用锚定。取子集：`stream=True` 抽 300K articles ≈ **1.5B tokens**，或按文章标题做 query/doc 自监督对。 |
| G6 | `opencsg/chinese-fineweb-edu` | `size_categories:10M<n<100M`（310 文件） | `apache-2.0` | 中文高质量教育级 web 语料。**实测 schema 只有单字段 `text`（无时间戳、无 url）** → 只能做通用锚定，**不能做冷启动切分**。取子集：`stream=True` 抽 200K docs。 |
| G7 | `BAAI/IndustryCorpus2`（及 30 个子集） | zh 1 TB + en 2.4 TB（官方表） | `apache-2.0`（**父仓库 gated=auto，需登录接受条款；30 个子集仓库 ungated**） | 中英双语 + **11 个中文行业分类标签**。子集实测目录结构 `chinese/{high,middle,low}`、`english/{high,middle,low}`，parquet shard 独立可单文件下载（见 §2）。 |
| G8 | `BeIR/msmarco` | 1.53 GB parquet（实测），corpus 8.84M passages | `other`（BEIR 统一标注；上游 MS MARCO 为 MSR-LA 条款） | 通用英文检索的事实标准锚定集。子集：`train` qrels 含少量硬负例；**只取 corpus 前 1M passages + train/dev queries** 即可得到 ~1M 对。 |
| G9 | `C-MTEB/T2Retrieval` | corpus **118,605**（实测），queries 24,832；repo **150.3 MB** | **未声明**（卡片无 license 字段 → UNVERIFIED，继承 T2Ranking 原条款） | **通用中文锚定的首选**（T2Ranking 是中文搜索日志真实 query→passage）。实测 corpus 字段 `{id: str, text: str}`，queries 字段 `{id, text}`。 |
| G10 | `C-MTEB/DuRetrieval` | corpus **100,001**（实测），queries 4,000；repo **61.5 MB** | 未声明（UNVERIFIED） | 通用中文第二锚定源（百度搜索日志），格式同 G9。 |
| G11 | `C-MTEB/CovidRetrieval` | corpus **100,001**（实测），queries 949；repo **62.1 MB** | 未声明（UNVERIFIED） | 中文新闻域，兼作通用锚定 + "新闻体"锚定。 |
| G12 | `mteb/scifact` | 8 文件 / **7.9 MB**（实测） | `unknown`（上游 `allenai/scifact` = CC-BY-NC-2.0，**NC**） | 科学文献 claim→摘要检索。**仅评测用途**，不要进训练集（NC 条款）。schema：corpus `{_id, title, text}`、queries `{_id, text}`、qrels `{query-id, corpus-id, score}`。 |

### 1.1 通用锚定的子采样策略（→ 5–20M pairs）

推荐**不自造 query**，而是从上面带真 qrels 的集合直接转 triplet：

```
target = 8M pairs
├─ zh 60% ≈ 4.8M
│   ├─ T2Ranking (G9)      : 24,832 queries × 1 pos × 4 hard-neg  →  24.8k triplets
│   ├─ DuRetrieval (G10)   :  4,000 × 5                            →   4.0k
│   ├─ CovidRetrieval (G11):    949 × 5                            →   0.9k
│   └─ wikipedia.zh 自监督 (G5): 标题 → 首段，词级 BM25 去重后 4.7M
└─ en 40% ≈ 3.2M
    ├─ MS MARCO (G8)       : 1M queries（train+dev）
    └─ FineWeb 标题/首句自监督 (G2): 2.2M
```
> 关键点：**每条 query 配 1–4 个真 qrels 负例作为 hard-negative，其余负例由
> in-batch 提供**。C-MTEB 的 `-qrels` 仓库（**已全部核验存在**：
> `C-MTEB/T2Retrieval-qrels`、`DuRetrieval-qrels`、`CovidRetrieval-qrels`、
> `CmedqaRetrieval-qrels`、`MedicalRetrieval-qrels`、`EcomRetrieval-qrels`，
> 以及 `BeIR/fiqa-qrels`；split 名 `dev`，字段 `{query-id, corpus-id, score}`）
> **必须单独下载**，否则拿不到负例。
>
> ⚠️ **一个必须知道的事实**：C-MTEB 在 HF 上的 corpus 是**被截断过的子集**，
> 不是论文里的全量。实测行数：`T2Retrieval` 118,605（论文全量约 438K）、
> `DuRetrieval` / `CovidRetrieval` / `CmedqaRetrieval` 各 100,001、
> `MedicalRetrieval` 100,999、`EcomRetrieval` 100,902。
> **这不影响训练**，但**不能把这些行数当成论文原值引用**。

---

# 第二部分：分领域数据

## 2.1 医疗 / Medical

### 2.1.A 域内原文（挖 N-gram + 术语）

| HF dataset id | 规模 | License | 说明 |
|---|---|---|---|
| `MedRAG/pubmed` | **2,209,839** 行（实测 `num_rows_total`）；repo **52.8 GB / 999 文件**（实测） | **未声明（UNVERIFIED）** | PubMed 摘要，S2ORC 切分。**实测 schema**：`{id: str, title: str, content: str, contents: str, PMID: int64}`——`content` = 首 chunk，`contents` = 全 chunk 列表。**无时间戳字段**。取子集：**按 parquet 文件逐个下**（999 个 shard 单文件下载），取 40 个 shard ≈ 2 GB ≈ 90K abstracts ≈ 90M tokens，足够挖医学术语。 |
| `openlifescienceai/medmcqa` | 5 文件 / **0.08 GB**（实测）；标准划分 train 182,822 / val 8,033 / test 4,183（**行数 UNVERIFIED-ON-HUB**：`rows` 接口对 `train` config 返回 404） | `apache-2.0` | 印度医考选择题，**schema UNVERIFIED-ON-HUB**（上游为 `question, opa, opb, opc, opd, cop, choice, exp`）。**用途：天然 hard-negative 源**（四个干扰选项即负例）→ **使用前必须 dump 首行确认字段名**。 |
| `ImperialCollegeLondon/health_fact` | 3 文件（实测）；`size_categories:10K<n<100K` | `mit` | PUBHEALTH 事实核查，`claim + review(evidence) + label`。用途：claim→evidence 检索对 + 医学实体。 |
| `BAAI/IndustryCorpus2_biomedicine` | repo **32.1 GB**（实测），79 parquet；`chinese/high 0.7 GB`、`chinese/middle 4.5 GB`、`english/high 15.4 GB` | `apache-2.0`（**ungated**） | **中文生物医药原文的首选**。取子集：`chinese/middle` 下 5 个 shard ≈ 1.5 GB。 |
| `BAAI/IndustryCorpus2_medicine_health_psychology_traditional_chinese_medicine` | 官方表 271.7 GB | `apache-2.0`（ungated） | 中文医疗健康/中医/心理学大规模原文。取子集：`chinese/middle` 5–10 shard。 |
| `mteb/scifact` (G12) | 5,189 corpus / 300 queries | `cc-by-nc-2.0` **NC** | 仅评测。 |

### 2.1.B 指令 / QA / 相似度 → (query, pos, hard-neg)

| HF dataset id | 规模 | License | **精确字段名（实测）** | triplet 构造 |
|---|---|---|---|---|
| `qiaojin/PubMedQA` | config `pqa_artificial` **211,269** 行 / 233 MB；`pqa_unlabeled` **61,249** / 66 MB；`pqa_labeled` **1,000** / 1.1 MB。repo **286.6 MB**（实测） | **`mit`** | `pubid:int32`, `question:str`, `context:{contexts:list[str], labels:list[str], meshes:list[str], (+reasoning_*_pred)}, long_answer:str`, `final_decision:str ∈ {yes,no,maybe}` | **query** = `question`；**positive** = `long_answer`（短）或 `" ".join(context.contexts)`（长）；**hard-neg** = 另一篇的 `long_answer` / 其他 contexts。**额外收获：`context.meshes` 就是官方 MeSH 术语表，直接进 Layer-A Registry。** |
| `bigbio/med_qa` | 45 个 config；`med_qa_en_source/train` = **10,178** 行（实测），`med_qa_zh_source/train` = **27,400** 行（实测） | 卡片 `unknown`（上游 PubMedQA = MIT） | `meta_info:str`, `question:str`, `answer_idx:str`(A/B/C…), `answer:str`, `options:list[{key:str, value:str}]` | **中文医考数据，是最好用的中文医疗 triplet 源**。query = `question`；positive = `options[answer_idx].value`；**hard-neg = 其余 4 个 options 的 value**（同主题、同术语，极硬）。 |
| `C-MTEB/CmedqaRetrieval` | corpus **100,001**（实测）/ queries **3,999** / repo **58.5 MB** | 未声明（UNVERIFIED） | corpus/queries 均 `{id: str, text: str}`；qrels 在 `C-MTEB/CmedqaRetrieval-qrels`（`{query-id, corpus-id, score}`） | 中文在线医疗咨询语料，query=问诊文本，positive=qrels 命中的回答文本（**多条**，天然多正例）。 |
| `C-MTEB/MedicalRetrieval` | corpus **100,999** / queries **1,000** / repo **23.9 MB** | 未声明（UNVERIFIED） | 同上 | 阿里医疗搜索日志，**商用中文查询风格**，专测 query-passage 泛化。 |
| `mteb/MindSmallReranking` | repo **2.14 GB / 90 文件**（实测）；`size_categories:100M<n<1B` | `other` | 官方 MTEB 封装：`corpus`、`queries`、`relevant_docs`、`top_ranked`（MTEB v2 结构） | 英文新闻，作为通用 anchor + rerank 蒸馏信号。**冷启动见 §5.4**（原始 `yjw1029/MIND` 才有逐条时间戳）。 |

### 2.1.C 实体 / 术语资源（Layer-A Registry）

| 资源 | 获取方式 | License | Registry 用途 |
|---|---|---|---|
| **MeSH（医学主题词表）** | ① 直连 NLM：<https://www.nlm.nih.gov/databases/download/mesh.html>（XML / ASCII / MARC21 / RDF，年度发布）。② 免下载捷径：`qiaojin/PubMedQA` 的 `context.meshes` 字段（**MIT，已核验**） | NLM Terms & Conditions（免费，含少量商标声明）；PubMedQA 镜像为 MIT | 规范医学概念名（"Neoplasms"、"Immunosuppressive Agents"…）→ 通用医疗术语 slot |
| **ICD-10-CM（疾病编码）** | 权威源：NLM/CDC 每年 XML（<https://www.cdc.gov/nchs/icd/icd-10-cm/files.html>）+ CMS ZIP（<https://www.cms.gov/medicare/coding-billing/icd-10-codes>）。**HF 镜像（已核验存在）**：`Gokul-waterlabs/ICD-10-CM`，10.1 MB，MIT，`size_categories:10K<n<100K`，实测 schema `{input:str, output:str, instruction:str}`（`output` = ICD 码，`input` = long description） | ICD-10-CM 在**美国境内免费使用**；WHO 版 ICD-10 有授权要求 ⚠️ | 编码 ↔ 描述双向 slot；也是**天然冷启动源**（每年 10 月 1 日新增约 200–300 个码） |
| **SNOMED CT** | `awacke1/SNOMED-CT-Code-Value-Semantic-Set.csv`（已核验存在，MIT，`size_categories:100K<n<1M`） | ⚠️ **上游 IHTSDO 许可，非美国需授权；US Edition 仅美国境内免费**。HF 上标 MIT 的镜像**不代表原始许可** | **默认不进训练/部署**，仅在受控研究环境使用 |
| 中文医学术语 | 《国家医保药品目录》《国家基本药物目录》（国家医保局 / 国家卫健委官网，Excel 公开）；《中国药典》名称表 | 政府公开数据（UNVERIFIED-条款，逐份核对） | 中文药品名 / 剂型 / 适应症 slot |

---

## 2.2 金融 / Finance

### 2.2.A 域内原文

| HF dataset id | 规模 | License | 说明 |
|---|---|---|---|
| `PleIAs/SEC` | repo **20.2 GB / 34 文件**（实测，1993–2024 各年 1 个 parquet） | **`cc0-1.0`**（实测；SEC 原始文件为美国政府公有领域） | **实测 schema**：`{filename:str, id:str, year:str, cik:str, text:str, word_count:int64, character_count:int64}`。**带 `year` → 时间切分用**（见 §5.2）。取子集：只下 `2015.parquet`…`2024.parquet` 共 10 个文件 ≈ 6 GB。 |
| `TeraflopAI/SEC-EDGAR` | **590 GB**，8,055,455 filings，**43.7B tokens** | `apache-2.0`（实测） | 全量 EDGAR（10-K/10-Q/8-K/4/S-1…）。**太大**，只做备份源；按 `(form, year)` 单 shard 下载。 |
| `eloukas/edgar-corpus` | 87 文件；`size_categories:100K<n<1M` | `apache-2.0`（实测） | EDGAR 全文语料，字段通常含 `cik, company, filing_date, form_type, ...`（**字段名 UNVERIFIED**）。 |
| `BAAI/IndustryCorpus2_finance_economics` | repo **84.2 GB / 146 parquet**（实测）：`chinese/high 8.3 GB`、`chinese/middle 22.4 GB`、`chinese/low 7.3 GB`、`english/high 18.6 GB`、`english/middle 24.1 GB`、`english/low 3.5 GB` | `apache-2.0`（ungated） | **中文财经原文的首选**。取子集：`chinese/middle` 下 6 个 shard ≈ 1 GB。 |
| `BAAI/IndustryCorpus2_news_media` | 官方表 51.0 GB | `apache-2.0` | 中文财经新闻的上位语料（金融新闻多数落在此）。 |
| `mteb/MindSmallReranking` / `yjw1029/MIND` | 2.14 GB | `other` / 未声明 | 通用新闻语料，兼作金融新闻上位域。 |

### 2.2.B 指令 / QA → triplet

| HF dataset id | 规模 | License | **精确字段名** | triplet 构造 |
|---|---|---|---|---|
| `BeIR/fiqa` | 4 文件 / **26.7 MB**（实测）；corpus 58,286（MTEB v1 论文表实测统计；FiQA 原论文为 57,638 posts），queries 6,500（train 5,500 / dev 500 / test 500） | `cc-by-sa-4.0`（实测） | corpus `{_id, title, text}`；queries `{_id, text}`；qrels `BeIR/fiqa-qrels` `{query-id, corpus-id, score}` | **英文金融问答检索的首选**。query = 用户提问；positive = qrels 命中的帖子；**hard-neg = 同帖 top-1 BM25 但未被判相关者**。每 query 平均 ~8.2 个正例。 |
| `next-tat/TAT-QA` | 6 文件 / **0.02 GB**（实测）；train 13,395 / val 1,909 / test 2,193；2,757 contexts，16,552 QA | `cc-by-4.0`（实测） | 实测 `first-rows` 在 `default/train` 返回 **500（parquet 抽取失败）→ 字段名 UNVERIFIED**。上游 TAT-QA 权威字段为 `table:list[list[str]]`, `paragraphs:list[{order, text}]`, `questions:[{uid, table_id, paragraph_ids, question, answer, scale, derivation, answer_type, ...}]` | query = `question`；positive = 对应 `paragraphs` 拼接文本（**表名/列名本身就是金融术语 → 顺手喂 Registry**）；hard-neg = 同段落集合的其他问题的答案 + 随机段落。 |
| `ibm-research/finqa` | 卡片 CC-BY-4.0；FinQA 本体 8,281 QA / 2,789 报表 | `cc-by-4.0`（实测） | ⚠️ 该仓库**只剩 `finqa.py` 加载脚本**（实测 files: `.gitattributes, README.md, finqa.py`），从 GitHub 拉原始 JSON；`datasets>=3.0` 已不支持脚本式数据集，需自行 `git clone https://github.com/czyssrs/FinQA`。上游字段：`id, pre_text, post_text, table, qa:{question, answer, program, gold_inds}` | query = `qa.question`；positive = `pre_text + post_text + table`；hard-neg = 同报表其他问题的答案。**小（8K），做微调信号足够。** |
| `orgrctera/uda_tat_qa_qa` | 14,703 QA | 卡片未声明（UNVERIFIED） | 实测（README）：`input:str`, `expected_output:str`(JSON)，`metadata:struct` | TAT-QA 的 UDA 封装，字段规整，可直接做 `input → expected_output` 配对。 |
| `TableQAKit/FinQA_retrieval` | train/validation/test | 未声明（UNVERIFIED） | **实测 first-rows 返回 500 → 字段名 UNVERIFIED** | 名义上已是 retrieval 版，但 schema 未核验，**使用前必须先 dump 一行确认**。 |
| `appQQQ/FinanceComplexQA` | 576 例（4 个场景 × zh/en）；**经 web 核验存在，HF API 未逐字段核验** | 未声明（UNVERIFIED） | 未核验 | 中英双语复杂金融 QA。**规模太小 → 仅作评测集，不进训练。** |
| **中文金融 triplet：⚠️ 缺口** | — | — | — | **HF 上没有规模足够、字段干净的中文金融 (query,pos,neg) 检索集。** 中文金融 triplet **必须自建**，见下方方案。 |

**中文金融 triplet 自建方案（推荐，已被 BGE 系验证的范式）**
1. 语料：`BAAI/IndustryCorpus2_finance_economics` 的 `chinese/middle` shard（保留段落切分前的换行/标题层级）。
2. 抽 query：在该 shard 内跑 TF-IDF / KeyBERT 取标题级短语 + 数字型短语（"2024年净利润"、"毛利率"、"北向资金"、"可转债"）。
3. 抽 positive：包含该短语且 `TF-IDF(term)` 最高的段落（长度 80–256 token）。
4. 抽 hard-negative：**同一 shard 内 BM25 top-50 中排名 2–10 且不含 query 短语的段落**（保证"相关但不答"，比随机负例硬 10 倍）。
5. 英文侧用 `BAAI/IndustryCorpus2_finance_economics/english/high` 做同样流程。

### 2.2.C 实体 / 术语资源（Layer-A Registry）

| 资源 | 获取方式 | License | 用途 |
|---|---|---|---|
| **A 股证券代码 + 简称** | 上交所 <https://www.sse.com.cn>、深交所 <https://www.szse.cn> 的"证券列表"页下载 xls/csv；北交所 <https://www.bse.cn> 同 | 交易所公开数据（免费，供内部使用） | "中芯国际 688981"、"宁德时代 300750" 一类**高频低歧义实体**，是 Registry 最佳种子（UNVERIFIED-是否可直接再分发，商用前需核对交易所条款） |
| **美股 ticker + 公司名** | SEC `company_tickers.json`（<https://www.sec.gov/files/company_tickers.json>，公有领域）；或 `eloukas/edgar-corpus` 的 `cik` 字段 | 公有领域 | 同上 |
| **SIC 行业码表** | SEC SIC 列表（<https://www.sec.gov/corpfin/division-of-corporation-finance-standard-industrial-classification>），约 1,000 条 | 公有领域 | 行业实体（"Semiconductors 3674"） |
| **财报科目 / XBRL 标签** | 从 `PleIAs/SEC` 的 10-K 正文正则抽取科目名（"Revenue", "Gross Profit", "Operating Income"） | `cc0-1.0` | 英美财报术语表 |
| 缩写词典 | 语料内自建：对 `IndustryCorpus2_finance_economics` 抽"首字母缩写（2–6 个大写字母）+ 全称"共现对 | `apache-2.0` | "EPS/ROE/PEG/CPI/PPI/MLF/LPR/OMO" |

---

## 2.3 法律 / Legal

### 2.3.A 域内原文

| HF dataset id | 规模 | License | 说明 |
|---|---|---|---|
| `coastalcph/lex_glue` | 23 文件 / **537.9 MB**（实测）。`case_hold`：train **45,000** / validation **3,900** / test **3,600**（官方 README） | **`cc-by-4.0`**（实测） | Harvard Caselaw 判决书 + holding。**实测 schema（`case_hold` config）**：`context:str`, `endings:list[str]`（**恰好 5 条**）, `label:ClassLabel(0..4)`。**注意**：官方 README 描述字段名为 `holdings`，但**加载器实际输出的是 `endings`** —— 以实测为准。 |
| `mteb/LeCaRDv2` | 5 文件 / **150.4 MB**（实测）。800 queries，55,192 candidate cases，avg 20.89 relevant/query | **`mit`**（实测） | **中文法律案例检索，MIT 许可可商用**。**实测 schema**：`corpus/corpus` → `{_id: str, title: str, text: str}`；`queries/queries` → `{_id: str, text: str}`；`default/test` → TREC 格式 qrels。 |
| `yefd/C3RD` | 17 文件 / **91.1 MB**（实测）。1,146 queries × 100 candidates，92,656 unique 民事案，13,096 relevant | `mit`（实测）⚠️ **gated=auto，需登录接受条款** | 中文民事案例检索。config：`documents` / `candidates` / `queries`（README 描述；**实测 datasets-server 返回 401，字段名 UNVERIFIED**）。 |
| `twang2218/chinese-law-and-regulations` | 4 文件 / **152.8 MB**（实测）；**22,552** 条（实测 `num_rows_total`，两个 config 均同） | **`apache-2.0`**（实测） | **中文法规原文 + 元数据，冷启动的黄金源**。**实测 schema（`default/train`）**：`publish_date:timestamp[ns]`, `effective_date:timestamp[ns]`, `type:str`（如 `宪法`/`法律`/`司法解释`）, `status:str`（`有效`/`已修改`/`已废止`）, `title:str`, `office:str`, `office_level:str`, `office_category:str`, `effective_period:str`, `content:str`。另有一个 `metadata/train` config（22,552 行，不含全文）。 |
| `pile-of-law/pile-of-law` | `size_categories:10M<n<100M` | **`cc-by-nc-sa-4.0`** ⚠️ **NC-SA：禁止商用，禁止进训练集** | 仅评测 / 参考。RA-Legal Pile。 |
| `BAAI/IndustryCorpus2_law_judiciary` | 官方表 238.5 GB | `apache-2.0`（ungated） | **中文法律语料首选**（比 NC 的 pile-of-law 可商用）。取 `chinese/middle` 5 shard。 |
| `ShengbinYue/DISC-Law-SFT` | 6 文件 / **0.54 GB**（实测）；~403K 条 | **`apache-2.0`**（实测） | **实测 schema（`default/train`）**：`id:str`, `input:str`, `output:str`。含 `DISC-Law-SFT-Pair.jsonl`、`DISC-Law-SFT-Triplet-QA-released.jsonl`、`DISC-Law-SFT-Triplet-released.jsonl` 等 5 个文件（文件名实测）。论文 §3.4 说明 Triplet 子集为 `<input, output, reference>`，**JSON 键名 UNVERIFIED（需 dump 首行确认）**。 |

### 2.3.B 指令 / QA → triplet

| 数据集 | 规模 | License | **精确字段名** | triplet 构造 |
|---|---|---|---|---|
| `coastalcph/lex_glue` (case_hold) | 45,000 train | `cc-by-4.0` | `context:str`, `endings:list[str]`(5), `label:int` | **query** = `context`（判决书节选，含被遮蔽的 holding 引用）；**positive** = `endings[label]`；**hard-neg = 其余 4 条 endings**（法域一致、话题一致 → 极硬负例，且天然带噪声/错法域负例）。**这是全语料里质量最好的法律对比学习源。** |
| `mteb/LeCaRDv2` | 800 q / 55,192 cases | `mit` | `_id, title, text` / `_id, text` / qrels | query = 查询案情；positive = 每个 relevant case（avg 20.9 个 → 多正例采样）；hard-neg = **同 `charge`/同罪名但未判相关的 case**（要联原始 LeCaRDv2 的 `relevence.trec` 做同罪名过滤）。 |
| `yefd/C3RD` | 1,146 q / 114,600 pairs | `mit`（gated） | 见 README（UNVERIFIED） | 同上；13,096 relevant + 101,504 non-relevant，**非相关候选按"同案由"分层采样**。 |
| `ShengbinYue/DISC-Law-SFT` (Pair) | ~256K pair + 93K legal QA | `apache-2.0` | `id, input, output` | query = `input`；positive = `output`；hard-neg = 同一 task-id 前缀下的其他 `output`。**中文法律问答主力。** |
| `ShengbinYue/DISC-Law-SFT` (Triplet) | ~99K | `apache-2.0` | 论文：`<input, output, reference>`（键名 UNVERIFIED） | query = `input`；**positive = `reference`（法条/判例原文）**；hard-neg = 其他条目的 `reference`。**这是唯一现成的中文"问题→法条原文"检索数据，价值极高。** |

### 2.3.C 实体 / 术语资源（Layer-A Registry）

| 资源 | 获取方式 | License | 用途 |
|---|---|---|---|
| **中国法律法规数据库（权威）** | <https://flk.npc.gov.cn>；2025-08 完成二期升级，旧 `/api/` 已废弃，现为 `/law-search/*` JSON 接口；含 `title / type / sxx(时效性) / flxz(法律性质) / office / content` 等字段；**限流约 20 次请求** | 中国政府公开数据 | 法条全称 + 法条编号 + 制定机关 → Registry slot。**已通过 `twang2218/chinese-law-and-regulations` 离线镜像（Apache-2.0）获得，无需爬虫。** |
| `twang2218/chinese-law-and-regulations` 的 `title` 字段 | 同上，22,552 条 | `apache-2.0` | 直接抽取：《民法典》《证券法》《个人信息保护法》… 全称与简称 |
| 同上 `content` 字段 + 正则 | 条号抽取 `第[一二三四五六七八九十百千零〇\d]+条` | `apache-2.0` | 8 万+ 条法条编号（"民法典第577条"）→ Registry |
| 司法解释专有名词 | 同上 `type == "司法解释"` 子集 | `apache-2.0` | "最高人民法院关于……的解释" |
| 美国法条 / 案名 | CaseHOLD `endings` 中的 `18 usc 2113a`、`346 F.3d 205` 等引用串；或 govinfo（US 公有领域） | `cc-by-4.0` / 公有领域 | 案名 + 引证号 slot |

---

## 2.4 电商 / E-commerce

### 2.4.A 域内原文

| HF dataset id | 规模 | License | 说明 |
|---|---|---|---|
| `McAuley-Lab/Amazon-Reviews-2023` | 全库 **750 GB**（实测）；571.54M reviews，54.51M users，48.19M items，30.14B review tokens，**May 1996 – Sep 2023** | **卡片未声明 license（UNVERIFIED）**——引用前需按 McAuley Lab 页面条款自行确认 | **实测字段**：`raw_review_*` → `{rating:float64, title:str, text:str, images:list[{attachment_type, large_image_url, medium_image_url, small_image_url}], asin:str, parent_asin:str, user_id:str, timestamp:int64(ms), helpful_vote:int64, verified_purchase:bool}`；`raw_meta_*` → `{main_category, title, average_rating, rating_number, features:list[str], description:list[str], price, images:{hi_res,large,thumb,variant}, videos:{title,url,user_id}, store:str, categories:list[str], details:str(JSON), parent_asin, bought_together, subtitle, author}`。**带 `timestamp` → 冷启动主源（§5.1）。** 取子集：`raw_review_All_Beauty` **139 MB / 701,528 行** + `raw_meta_All_Beauty` **807 MB / 801,446 items**（两个 download_size 均为官方 README 实测值）。 |
| `C-MTEB/EcomRetrieval` | corpus **100,902**（实测）/ queries **1,000** / repo **8.1 MB**（实测） | 未声明（UNVERIFIED） | 阿里电商搜索日志。**实测 schema**：corpus `{id: str, text: str}`（样例："红棉优级小粒老黄冰糖1.2kg大罐炖煮煲汤…"）、queries `{id: str, text: str}`（样例："大落地窗"）。**中文电商 query→商品短标题检索，域内价值最高的小数据集。** |
| `amazon_reviews_multi` | 3 文件；`size_categories:100K<n<1M` | `other` | ⚠️ **实测语言标签仅 `de/en/es/fr`——不含 zh/jp**（与部分论文所述的 6 语版本不同，**不要按"含中文"使用**）。用于多语泛化测试。 |
| `TaoBao-MM/Taobao-MM` | 大规模淘宝推荐序列（含 item embedding） | 需核验（UNVERIFIED） | 仅推荐任务，非检索；`train_samples.parquet` 含 `user_id, item_id, label(click)`。可用于 item 侧实体统计。 |
| KuaiSearch | 快手电商搜索日志：Relevance 子集 46,422（`query, title, brand name, seller name, attribute, score`）+ Ranking 子集 81,401,477 | arXiv:2602.11518，**HF repo id 未核验 → UNVERIFIED** | 若能拿到，是最好的中文电商 (query, title, brand, attribute) 四元组。 |

### 2.4.B 指令 / QA / 相似度 → triplet

| 数据集 | 规模 | License | **精确字段名** | triplet 构造 |
|---|---|---|---|---|
| `C-MTEB/EcomRetrieval` | 1,000 q / 100,902 docs | 未声明 | `{id, text}` × 2 | query = 用户 query；positive = qrels 命中的商品短标题（多正例）；**hard-neg = 同 `id` 区间 / 同词形商品**（需靠 qrels 的非相关集 + BM25 top-k） |
| `McAuley-Lab/Amazon-Reviews-2023` 自建 | 见上 | UNVERIFIED | 见 §2.4.A | **query** = review `title`（短、带型号/品牌，最像 query）；**positive** = `text` 前 192 token + 对应 `raw_meta.title` + `features`；**hard-negative = 同一 `main_category` 下 `average_rating` 相近但 `parent_asin` 不同的其它商品 meta**（同品类、同价位 → 极硬） |
| 同上 · 用户意图对 | 同上 | 同上 | 同上 | **query** = 同 `user_id` 的 review `title` 序列（上一件 → 下一件）；**positive** = 下一件的 meta title；**hard-neg** = 同用户同期浏览过的其它商品（用 `timestamp` 邻近窗口约束） |
| 同上 · 品牌/型号实体 | 同上 | 同上 | `raw_meta.store`(品牌), `raw_meta.details`(属性 JSON), `raw_meta.categories` | 直接进 Layer-A Registry（见 §2.4.C） |

### 2.4.C 实体 / 术语资源（Layer-A Registry）

| 资源 | 获取方式 | License | 用途 |
|---|---|---|---|
| Amazon 类目树 + 品牌 + 属性 | `McAuley-Lab/Amazon-Reviews-2023` 的 `raw_meta_*` 配置：`categories:list[str]`（层级路径）、`store:str`（品牌）、`details`（**JSON 字符串**，内含 `Material / Brand / Item model number / Size / Package Dimensions / UPC`） | UNVERIFIED（随上游） | **结构化程度最高、最省事**的英文电商术语源。`details` 是 JSON 字符串，用 `json.loads` 解析；`item model number` 就是型号实体（冷启动主力）。 |
| 中文类目/品牌/属性词典 | ⚠️ **缺口**：淘宝/京东的类目与属性表没有官方开放下载 | — | 退而求其次：① 从 `C-MTEB/EcomRetrieval` 100,902 条商品短标题里跑 **凝固度（PMI）+ 领域 TF-IDF** 抽短语；② 从 `IndustryCorpus2` 的 `other_manufacturing` / `other_information_services_information_security` 子集挖"规格/型号"模式（正则 `\d+\s*(mm|cm|kg|L|W|V|寸|匹|码)` 前后 3 token 窗口）；③ 若合规，允许在自有业务日志上做同样的抽取（这是最准的来源）。 |

---

# 第三部分：长尾 / 新实体 —— COLD START 评测数据

## 3.0 核心方法：如何保证"训练时没见过"

这是本文最关键的一节。**光有带时间戳的数据不够，必须构造严格的保证（guarantee）**：

```
T0 = 训练截止时间（例如 2024-06-30）

① 切分：把"同一来源的同一形态"的数据按时间戳切成
     TRAIN = { d : timestamp(d) < T0 }
     COLD  = { d : timestamp(d) ≥ T0 }

② 训练约束（必须全部满足，否则保证失效）：
   a. Layer-B（Hashed N-gram 表）只在 TRAIN 上做梯度更新；
   b. Layer-A（Registry）的 phrase→slot 索引只用 TRAIN 挖掘的术语构建；
   c. 通用锚定语料（G2/G3/G4）也必须按同一 T0 切分，不能混入 COLD 文档；
   d. N-gram 统计表 / TF-IDF 分母 / 词表全部只用 TRAIN 拟合。

③ 新实体定义（guaranteed unseen）：
   NEW_TERMS = { t : first_occurrence(t) 在整个语料中 ≥ T0 }
   实现：对 TRAIN ∪ COLD 全量语料做一次 suffix-N-gram 计数，
        统计每个 n-gram 的 **首次出现文档的 timestamp**；
        取 min(timestamp) ≥ T0 的 n-gram 集合。
        由于 TRAIN 内该 n-gram 计数严格为 0，
        → Layer-B 该槽位梯度为 0（保持初始化）
        → Layer-A 无对应 slot（matcher 不命中）
        ⇒ 该 term 在 DE 中 **一定未被记忆**，cold-start 增益完全归因于
          写入 Registry + 冷启动初始化（cold_start.py 的 closed-form ridge），
          而不是"碰巧在训练里见过"。

④ 评测：T2Ranking/LeCaRDv2/FiQA/Amazon 等评测集本身不含时间戳，
   **不要拿它们做 cold-start 评测**——要另建 COLD 评测集：
   从 COLD 文档里复用已有检索集的 query 生成方式（见 §3.5 方案 B）。
```

## 3.1 源 1：Amazon Reviews 2023（电商，最干净的冷启动源）

| 项 | 值 |
|---|---|
| HF id | `McAuley-Lab/Amazon-Reviews-2023`（已核验存在） |
| 时间字段 | `timestamp:int64`（**毫秒级 unix 时间**），实测 schema 确认 |
| 时间跨度 | May 1996 – Sep 2023 |
| 大小 | 全库 750 GB；单 config：`raw_review_All_Beauty` 139 MB / 701,528 行；`raw_review_Video_Games`、`raw_review_Toys_and_Games`（5.08 GB / 16.26M 行）等 |
| License | **UNVERIFIED**（卡片无 license 标签） |

**冷启动构造（新商品型号 / 新品牌）**
```
T0 = 2023-01-01
TRAIN : timestamp <  T0     （1996-01 ~ 2022-12）
COLD  : timestamp >= T0     （2023-01 ~ 2023-09，约 571.54M 的最后 ~8%）

NEW_ASIN   = parent_asin 只出现在 COLD 的商品（→ 训练时商品从未出现）
NEW_MODEL  = raw_meta.details 里 "Item model number" 只出现在 COLD 的型号串
NEW_BRAND  = raw_meta.store 只出现在 COLD 的品牌

评测：query = 该商品在 COLD 期的一条 review.title
      positive = 该商品的 raw_meta.title + features 拼接
      hard-neg  = 同 main_category、同 average_rating±0.2、parent_asin≠gold 的商品
      （并按「该 ASIN 在 TRAIN 中出现次数 == 0」二次过滤）
```
**这个切分是 100% 可证的**：ASIN 是显式 ID，只要 TRAIN 里 count(asin)==0，模型就没在
训练数据里见过这个商品。

## 3.2 源 2：SEC 10-K（金融，按年切分）

| 项 | 值 |
|---|---|
| HF id | `PleIAs/SEC`（已核验，**CC0-1.0**） |
| 时间字段 | `year:str`（如 `"1993"`），实测 schema 确认；`id` = `cik_year` |
| 字段 | `filename, id, year, cik, text, word_count, character_count` |
| 大小 | 20.2 GB，1993–2024 每年 1 个 parquet（34 文件） |

**冷启动构造**
```
T0 = 2022
TRAIN : year <= 2022
COLD  : year >= 2023        （2023/2024 两个 parquet，约 1.2 GB）

NEW_CIK   = cik 只出现在 2023+ 的公司（IPO 新上市公司）
NEW_SEG   = 只在 2023+ 出现的业务分部名 / 产品线名（"Aether Rotor"、"Helios Compute"）
QUERY  = "2024 年 <CIK 简称> 的 <新分部/新产品> 业务收入是多少"  ← 需模板生成
POSITIVE= 该 CIK 2023/2024 年 10-K 中含该词段落
HARD-NEG = 同 industry、同 size 分档、year <= 2022 的 CIK 的相似段落
```
**注意**：`PleIAs/SEC` 只有 `year`（年粒度），若要月/日粒度，换用
`TeraflopAI/SEC-EDGAR`（Apache-2.0）或直接解析 EDGAR header 里的 `FILED AS OF DATE`。

## 3.3 源 3：中文法规（法律，唯一带"发布日期"的中文垂域金矿）

| 项 | 值 |
|---|---|
| HF id | `twang2218/chinese-law-and-regulations`（已核验，**Apache-2.0**，22,552 条） |
| 时间字段 | **`publish_date:timestamp[ns]`**、`effective_date:timestamp[ns]`、`effective_period:str` —— 实测 schema 确认 |
| 其他字段 | `type`（宪法/法律/行政法规/司法解释/部门规章…）、`status`（有效/已修改/已废止）、`title`、`office`、`office_level`、`office_category`、`content` |

**冷启动构造（最干净的中文新术语实验）**
```
T0 = 2022-01-01
TRAIN : publish_date <  T0   （≈ 21,000 条旧法）
COLD  : publish_date >= T0   （≈ 1,500 条新法/新司法解释）

NEW_STATUTE_TITLE = title 只在 COLD 出现（例："最高人民法院关于……的解释（2023）"）
NEW_CLAUSE       = 「第X条」编号在 COLD 才出现
NEW_TERM         = 对 COLD 文本做「新词发现（凝固度 + 未登录字/词）」，筛掉 TRAIN 词表里已有的

QUERY  = 新法标题中的核心义务/条件从句（模板或 LLM 抽取）
POSITIVE= 该条 content（含条号）
HARD-NEG = TRAIN 中主题最近似的旧法条款（同 type、同 office_category、TF-IDF/BM25 top-10 中随机 4）
```
**保证**：只要新法标题/条号不在 TRAIN 中，Registry 里就没有这些 slot
（`twang2218` 的 Registry 种子仅从 TRAIN 的 `title` + `content` 正则抽取）。

## 3.4 源 4：Web 快照（中英通用，"新实体"最丰富的冷启动源）

| 项 | 值 |
|---|---|
| HF id | `HuggingFaceFW/fineweb-2`（ODC-By，15T tokens，44 TB） |
| 时间字段 | **`dump:str`**（如 `"CC-MAIN-2024-42"`）+ **`date:str`**（ISO8601 抓取时间，如 `"2021-10-15T21:20:12Z"`）+ `url:str`。**已实测确认**（config `cmn_Hani`，split `train`，行数 636,058,984） |
| 中文子集 | **config `cmn_Hani`，splits `train`(636,058,984 行) / `test`(33,296 行)** |
| 许可 | `odc-by`，**但受 CommonCrawl ToU 约束** |

**冷启动构造（`dump` 级切分，最锐利）**
```
T0_dumps = {CC-MAIN-2023-*, CC-MAIN-2024-*}     # CommonCrawl 有明确快照边界
TRAIN     : dump ∈ {CC-MAIN-2019-* … CC-MAIN-2022-*}
COLD      : dump ∈ {CC-MAIN-2023-*, CC-MAIN-2024-*}

NEW_ENTITY 候选：
  ① 新公司/产品型号：url 的 host 在 TRAIN 中出现 0 次（COLD 中出现 ≥ 3 次）
  ② 新 n-gram：对 COLD 做 n-gram(=2..4) 计数，取在 TRAIN 中 count==0 且在 COLD 中
     count ≥ 5 的 n-gram（这就是 §3.0 ③ 的实现）
  ③ 新 SKU / 新药名 / 新法规名：正则 `\b[A-Z]{2,6}[-–]\d{2,6}\b`、
     `\d{4}年\d{1,2}月\d{1,2}日.*(公告|办法|条例)`、CAS/RXN 码等
```
**为什么 `dump` 比 `date` 好**：快照边界是硬隔离（同一 `dump` 内的文档绝不会被切到
两侧），而 `date` 是抓取时间，虽然单调但边界在同一 `dump` 内部。
**推荐两者并用**：`dump` 定边界，`date` 定桶。

## 3.5 源 5：Wikipedia 带时间戳（中文 + 英文，术语冷启动）

| 项 | 值 |
|---|---|
| HF id | `HuggingFaceFW/finewiki`（**CC-BY-SA-4.0 + GFDL**，184.7 GB parquet 实测） |
| 实测 schema | `text, id, wikiname, page_id:int64, title, url, date_modified, in_language, wikidata_id, bytes_html:int64, wikitext, version:int64, infoboxes, has_math:bool` |
| 中文 | config `zh/train`（1,295,955 页，5.1 GB）；另有 `zh_classical`、`zh_min_nan`、`zh_yue` |

**⚠️ 重要限制**：FineWiki 只保留每个 page 的**最新** revision（2025-08 dump），
`date_modified` = 最后修改时间，**不是创建时间**。因此：
- ✅ 可以说："这个实体在 2025-08 之前存在，但 T0 之后才被修改过" → **修饰型新术语**
  （如新商品名、新公司名、新产品线）
- ❌ 不能说："这个页面是 T0 之后新建的" → **新建页面不可用**

**补偿方案（务必执行）**：把 FineWiki 的 `date_modified` 当作**弱信号**，用
`wikidata_id` 的外部锚点补齐创建时间——或更简单：**用 FineWeb-2 的 `dump` 边界**
做主切分（§3.4），FineWiki 只用作**新术语词表的第二来源**。

**方案 B（推荐，成本最低）**：不自己造 COLD 评测集，而是**改造已有检索评测集**：
从 `T2Ranking` / `DuRetrieval` / `LeCaRDv2` / `FiQA` / `EcomRetrieval` 的 corpus 里，
用 FineWeb-2 的 `dump` 标签做交叉：如果某个 doc 的内容能在 2025 快照里找到而在
≤2022 快照里找不到，则该 doc 的关键实体就是"新实体"。
（实现成本高；第一轮实验建议直接用 §3.1/§3.2/§3.3 的显式 ID 时间切分，更干净。）

## 3.6 源 6：MIND 新闻（弱选项，仅用于"新事件名"）

| 项 | 值 |
|---|---|
| HF id | `yjw1029/MIND`（已核验存在）；封装版 `mteb/MindSmallReranking`（`other`） |
| 时间字段 | ⚠️ **原始 MIND 的 `behaviors.tsv` 有 `Time` 列（如 `11/13/2019 8:36:57 AM`）和 impression 时间；但 HF 镜像 `yjw1029/MIND` 的 schema 未核验，且 `gated=auto`、license 未声明 → 需先 dump 一行确认** |
| 规模 | 约 160k 英语新闻 + 15.78M impression logs，1M users |
| 价值 | impression 时间天然给出"查询 → 点击文章"的时间序，可做**行为层面的时间切分**（训练用 11/09–11/14，冷启动用 11/15） |

**结论：MIND 的价值在"新事件/新实体词"（如新出现的公司、产品、灾害名），
但 schema 需自行确认，优先级低于 §3.1–§3.4。**

## 3.7 中文 e-commerce 冷启动 ⚠️ 缺口

- `C-MTEB/EcomRetrieval` **无时间戳** → 不能做冷启动切分（只能做域内评测）。
- 替代方案（按优先级）：
  1. **`PleIAs/SEC` / Amazon 英文侧的冷启动结论外推到中文**（模型层共享；语言层不共享，需谨慎）。
  2. **KuaiSearch**（§2.4.A）：若能获取，其 ranking 日志天然带 `time index`，可做中文电商冷启动。**HF id 未核验 → UNVERIFIED。**
  3. **自建**：用 FineWeb-2 `cmn_Hani` 的 `dump` 切分，抓"新型号/新品牌"出现，构造中文商品 query。成本最高但完全可控。
  4. **务实方案（第一轮实验推荐）**：用 §3.3 的中文法规冷启动（`twang2218`）作为
     **主 cold-start 战场**（字段干净、Apache-2.0、时间戳精确、术语明确），
     电商/金融的中英冷启动放到第二轮。

---

# 第四部分：GB10（128 GB 统一内存）最小可跑配方

> 目标：**第一次训练 < 8 小时**（含数据准备 + Stage 1 + Stage 2 冷启动写入 + 评测）
> 假设 backbone = `BAAI/bge-small-zh-v1.5`（≈24M 参数，384 维），冻结 backbone，
> 只训 Layer-A/B + 融合层；序列长度 `max_len=128`（query/pos/neg 统一）。

## 4.1 时间预算（保守估计）

| 项 | 估算 |
|---|---|
| 单 triplet token 数 | query≈30 + pos≈128 + neg≈128 ≈ **287 token** |
| 前向 FLOPs/三流 | ≈ 2 × 25M × 287 ≈ **1.4 × 10¹⁰ FLOP/triplet**（backbone 冻结时仍需完整前向） |
| GB10 有效吞吐（bf16/AMP，20 核 Arm，`batch=64` triplets ≈ 192 seq） | 保守取 **80–150 triplets/s** |
| 850k triplets × 1 epoch | **≈ 1.6–3.0 小时** |
| 数据准备（下载 + 解析 + 抽三元组 + Registry 构建） | **≈ 1.5–2.5 小时** |
| Stage 2 冷启动写入（closed-form ridge，无梯度） | **≈ 10–20 分钟** |
| 评测（MTEB/C-MTEB 检索子集 + 自建 cold-start） | **≈ 1–2 小时** |
| **合计** | **≈ 4.5–8 小时** ✅ |

## 4.2 精确子集清单

| # | 用途 | HF dataset id | 下载量（实测） | 取用 | Triplets | ≈Tokens |
|---|---|---|---|---|---|---|
| A1 | 通用中文锚定 | `C-MTEB/T2Retrieval` | 150 MB | 24,832 queries 全用，corpus 118,605 全用 | 24,832 | 7.1 M |
| A2 | 通用中文锚定 | `C-MTEB/DuRetrieval` | 62 MB | 4,000 queries + 100,001 corpus | 4,000 | 1.1 M |
| A3 | 中文新闻锚定 | `C-MTEB/CovidRetrieval` | 62 MB | 949 queries + 100,001 corpus | 949 | 0.3 M |
| A4 | 通用英文锚定 | `BeIR/msmarco` | 1.53 GB | `train` qrels → 20 万三元组（子采样） | 200,000 | 57 M |
| **M1** | 医疗 en QA | `qiaojin/PubMedQA` (config `pqa_artificial`) | 233 MB | 采样 120,000 | 120,000 | 34 M |
| **M2** | 医疗 zh 医考 | `bigbio/med_qa` (`med_qa_zh_source/train`) | 小 | 27,400 全用（options 做 4 个 hard-neg） | 27,400 | 7.9 M |
| **M3** | 医疗 en 医考 | `bigbio/med_qa` (`med_qa_en_source/train`) | 小 | 10,178 全用 | 10,178 | 2.9 M |
| **M4** | 医疗 zh 检索 | `C-MTEB/CmedqaRetrieval` (+`-qrels`) | 59 MB | 3,999 queries + 100,001 corpus | 3,999 | 1.1 M |
| **M5** | 医疗 zh 搜索 | `C-MTEB/MedicalRetrieval` (+`-qrels`) | 24 MB | 1,000 queries + 100,999 corpus | 1,000 | 0.3 M |
| **F1** | 金融 en 检索 | `BeIR/fiqa` (+`-qrels`) | 27 MB | 5,500 train queries + 57,638 corpus | 5,500 | 1.6 M |
| **F2** | 金融 en 表文 | `next-tat/TAT-QA` | 20 MB | 13,395 train（先 dump 一行确认字段） | 13,395 | 3.8 M |
| **F3** | 金融 en 原文 | `PleIAs/SEC` | **6 GB**（`2015.parquet`…`2024.parquet`，10 文件） | 章节标题→段落自监督 100k | 100,000 | 29 M |
| **F4** | 金融 zh 原文 | `BAAI/IndustryCorpus2_finance_economics` | **1 GB**（`chinese/middle` 6 shard） | TF-IDF query + BM25 hard-neg 自监督 80k | 80,000 | 23 M |
| **L1** | 法律 en | `coastalcph/lex_glue` (config `case_hold`) | 538 MB（仅需 train） | 45,000 全用（`endings` 4 个 hard-neg） | 45,000 | 12.9 M |
| **L2** | 法律 zh 检索 | `mteb/LeCaRDv2` | 150 MB | 800 queries × 采样 20 个 (pos, hard-neg) 对 | 20,000 | 5.7 M |
| **L3** | 法律 zh 检索 | `yefd/C3RD` ⚠️gated | 91 MB | 1,146 queries + 13,096 relevant | 13,096 | 3.8 M |
| **L4** | 法律 zh QA | `ShengbinYue/DISC-Law-SFT` (Pair) | 540 MB | 采样 60,000（`input`→`output`） | 60,000 | 17 M |
| **E1** | 电商 zh 检索 | `C-MTEB/EcomRetrieval` (+`-qrels`) | 8 MB | 1,000 queries + 100,902 corpus | 1,000 | 0.3 M |
| **E2** | 电商 en 原文 | `McAuley-Lab/Amazon-Reviews-2023` | **946 MB**（`raw_review_All_Beauty` 139 MB + `raw_meta_All_Beauty` 807 MB） | review.title→meta/features 100k | 100,000 | 29 M |
| **E3** | 电商 en 自监督 | 同 E2 | 0 | (brand + model number) → features 60k | 60,000 | 17 M |
| **合计** | | | **≈ 11.0 GB 下载** | | **≈ 854,347** | **≈ 255 M** |

### 4.3 冷启动评测数据（第 4 阶段额外下载 ≈ 1.5 GB）

| 用途 | 数据 | 下载 | 切分 |
|---|---|---|---|
| **主战场（中文法律）** | `twang2218/chinese-law-and-regulations` | 153 MB | T0 = 2022-01-01（`publish_date`）→ 新法条/新司法解释 |
| **次战场（英文电商）** | `McAuley-Lab/Amazon-Reviews-2023` | 复用 E2 的 shard（需额外 `raw_meta_Video_Games` 约 1.9 GB，见备注） | T0 = 2023-01-01（`timestamp`）→ 新 ASIN / 新型号 |
| **第三战场（英文金融）** | `PleIAs/SEC` | 复用 F3 | T0 = 2022（`year`）→ 新 CIK / 新业务分部 |
| **通用冷启动（新实体 N-gram）** | `HuggingFaceFW/fineweb-2` config `cmn_Hani` | 3 个 parquet shard ≈ 3 GB | `dump` 边界：TRAIN ≤ CC-MAIN-2022-*，COLD ≥ CC-MAIN-2024-* |

> 备注：`raw_meta_Video_Games` download_size 未在本次核验中实测（`raw_review_Video_Games`
> dataset_size 5.08 GB / 16.26M 行，官方 README）。若时间紧，E2 的
> `All_Beauty`（1996–2023 全时段）已足够做冷启动实验。

## 4.4 训练超参（对应 PLAN §6 Stage 1）

```
backbone        = BAAI/bge-small-zh-v1.5   (frozen)
inject_layer    = 2                          (PLAN §1 证据 5)
max_len         = 128
pooling         = mean                       (PLAN §2 改造 1，先跑 mean，CLS/last 做消融)
batch_size      = 64 triplets (= 192 sequences/step)
hard_negatives  = 1 per triplet  (+ in-batch negatives)
lr              = 2e-5 (融合层 W_K/W_V/conv) ; 1e-4 (记忆表，= backbone_lr × 5)
weight_decay    = 0.0  (记忆表)  /  0.01 (融合层)
optimizer       = AdamW, cosine, warmup 5%
epochs          = 1                          (先 1 epoch 看曲线)
grad_ckpt       = 关闭（模型很小，128 GB 内存不需要）
precision       = bf16 AMP
Registry        = 从 §2.x.C 的术语 + TRAIN 语料的领域 TF-IDF 抽取，
                  上限 200,000 slots（按 PLAN §2 改造 3 的 d_registry 配置）
```

---

# 第五部分：JSONL Schema（训练三元组）

## 5.1 Collator 契约（`de/data.py` 实测）

`TripletDataset.from_jsonl()` 实际读取的字段：

```python
d = json.loads(line)
query      = d["query"]              # 必填，缺失会 KeyError
positive   = d.get("positive", d.get("doc", ""))   # positive 优先，回退 doc
negatives  = d.get("negatives", [])  # 必须是 list[str]；若是 str 会被包成单元素 list
negatives  = negatives[:max_negatives]             # 由 from_jsonl(max_negatives=N) 截断
```

`DECollator.__call__()` 把 `queries / pos / 逐条展开的 negatives` 三个流分别 tokenize，
并对**每个流**跑 `matcher.match_batch(ids)` 生成 `(B, T, R)` 的 registry slot 张量。

**⚠️ 已知实现细节（写数据时必须注意）**
1. `DECollator` 把所有 negative 展平成一个流并 **共享 batch 维度**。若各样本
   negative 数量不同，展平后数量 ≠ `B × max_neg`，loss 计算侧需按
   `de/losses.py` 的约定还原分组。**建议所有行统一 `max_negatives` 个负例。**
2. `negatives` 为空列表时，collator 不返回 `"negative"` 键 → loss 侧会走
   "no explicit negative" 分支。**训练文件里不要出现空的 `negatives`。**

## 5.2 规范 JSONL（推荐）

每行一个 JSON 对象，UTF-8，不转义（`ensure_ascii=False`），`\n` 结尾：

```json
{"query": "C-MTEB 查询文本", "positive": "命中文档文本", "negatives": ["硬负例1", "硬负例2"]}
```

**最小契约（`de/data.py` 直接可读）**：

```json
{"query":"<str, 必填>","positive":"<str, 必填>","negatives":["<str>","<str>"]}
```

**推荐完整契约**（前 3 个字段必填，后 4 个由下游脚本读取、被 loader 忽略）：

```json
{
  "query":      "string, 必填, 非空",
  "positive":   "string, 必填, 非空",
  "negatives":  ["string", ...],   // 必填, 长度 == max_negatives, 元素非空, 与 positive 不重复

  "domain":     "medical|finance|legal|ecommerce|general",
  "source":     "hf:<dataset_id>/<config>/<split>  或  <self_built_pipeline_id>",
  "hard_neg_type": "label_neighbor|bm25_topk|same_category|same_charge|random|inbatch"
}
```

## 5.3 校验规则（生成脚本必须自检）

| 规则 | 检查 |
|---|---|
| R1 | `query`、`positive` 非空字符串，`len(negatives) == max_negatives`（默认 1） |
| R2 | `len(negatives) > 0`（否则 collator 不产 negative 流） |
| R3 | `positive ∉ negatives`，`len(set(negatives)) == len(negatives)` |
| R4 | `query not in negatives`（防止把 query 当负例） |
| R5 | 统一 tokenize 后：`len(tok(query)) ≤ 64`，`len(tok(pos)) ≤ 128`，`len(tok(neg)) ≤ 128`；超长按 **句号/换行边界**截断，**不要**直接 `max_len` 硬切（会切掉实体尾部，正好是我们要测的东西） |
| R6 | `negatives` 必须是 **hard negative**（BM25/同标签/同类别），随机负例只允许 ≤ 20% 作为正则化 |
| R7 | 冷启动集单独文件 `cold_start.jsonl`，其所有 term 必须满足 §3.0 ③ 的 `min(timestamp) ≥ T0` |
| R8 | 每行必须能反查到 `source`（可审计、可复现、可 license 追溯） |

## 5.4 冷启动 Registry 写入文件（Stage 2）

`de/cold_start.py::cold_start(model, tokenizer, term, contexts, ...)` 需要
`(term, contexts[])`，建议用同样的 JSONL：

```json
{"term":"<待写入的新术语, string>",
 "contexts":["<该术语出现的支持句1>", "<支持句2>", "<支持句3>"],
 "refine_steps": 20, "refine_lr": 0.05, "ridge": 1e-3,
 "expect_slot_delta": 12345,          // 写成功后该 term 在评测集上的 nDCG@10 预期变化(可空)
 "source":"twang2218/chinese-law-and-regulations#publish_date>=2022-01-01"}
```

---

# 第六部分：核验状态总表

## 6.1 已核验（HF API 200 + schema 实测）
`Shitao/bge-m3-data` · `qiaojin/PubMedQA` · `bigbio/med_qa` · `MedRAG/pubmed` ·
`openlifescienceai/medmcqa` · `ImperialCollegeLondon/health_fact` ·
`coastalcph/lex_glue` · `mteb/LeCaRDv2` · `yefd/C3RD` · `twang2218/chinese-law-and-regulations` ·
`ShengbinYue/DISC-Law-SFT` · `BAAI/IndustryCorpus2` + `_biomedicine` +
`_medicine_health_psychology_traditional_chinese_medicine` + `_finance_economics` +
`_law_judiciary` + `_news_media` · `C-MTEB/{T2Retrieval,DuRetrieval,CovidRetrieval,
CmedqaRetrieval,MedicalRetrieval,EcomRetrieval}` + 全部 6 个 `-qrels` 仓库 ·
`BeIR/{msmarco,fiqa,trec-covid}` ·
`mteb/{scifact,msmarco,MindSmallReranking}` · `PleIAs/SEC` · `TeraflopAI/SEC-EDGAR` ·
`eloukas/edgar-corpus` · `McAuley-Lab/Amazon-Reviews-2023` · `amazon_reviews_multi` ·
`HuggingFaceFW/{fineweb,fineweb-edu,fineweb-2,finewiki}` · `wikimedia/wikipedia` ·
`opencsg/chinese-fineweb-edu` · `next-tat/TAT-QA` · `ibm-research/finqa` ·
`Gokul-waterlabs/ICD-10-CM` · `pile-of-law/pile-of-law` · `yjw1029/MIND`

## 6.2 UNVERIFIED（必须自行确认后再用）
| 项 | 未核验内容 |
|---|---|
| `Shitao/bge-m3-data` | License 未声明（混合来源，**建议不用**）；tar.gz 内 per-source 文件清单 |
| `McAuley-Lab/Amazon-Reviews-2023` | License 未在卡片声明 → 商用前须查 McAuley Lab 站点条款 |
| 所有 `C-MTEB/*` | 卡片无 license 字段 → 继承上游，**商用前须逐个核对** |
| `next-tat/TAT-QA` | 字段名（`first-rows` 返回 500）→ **使用前必须 dump 首行** |
| `ibm-research/finqa` | 仓库仅剩加载脚本，`datasets>=3.0` 不支持 → 需 `git clone` 上游 |
| `TableQAKit/FinQA_retrieval` / `orgrctera/uda_tat_qa_qa` | 字段名未核验（前者 `first-rows` 500） |
| `yefd/C3RD` | config/字段名（datasets-server 401，需登录后确认） |
| `ShengbinYue/DISC-Law-SFT` (Triplet) | JSON 键名（论文说是 `<input, output, reference>`，文件键名未确认） |
| `HuggingFaceFW/fineweb-2` | 字段名与中文 config **已实测**（见 G3）；但 `HuggingFaceFW/fineweb`（英文）的字段名来自官方论文，未在 Hub 直测 |
| `appQQQ/FinanceComplexQA` | 规模仅 576 → 只作评测 |
| KuaiSearch | **HF repo id 未核验** |
| `Alibaba-NLP/AliMe` | **HF API 返回 401（不存在或私有）——不要写进代码** |
| C-MTEB/OnlineShopping, C-MTEB/JDReview | **HF API 返回 401（不存在或私有）——不要写进代码** |
| A 股/美股证券代码表、ICD-10-CM 中文版、国家医保目录 | 官方源已定位（§2.2.C / §2.1.C），但**再分发条款未核验** |

## 6.3 已知缺口（第一轮实验必须绕开）
1. **中文金融 (query, pos, hard-neg) 检索三元组**——无现成公开集 → 按 §2.2.B 自建方案从
   `BAAI/IndustryCorpus2_finance_economics` 生成。
2. **中文电商类目/品牌/属性字典**——淘宝/京东不开放 → 从 `C-MTEB/EcomRetrieval`
   + `IndustryCorpus2` 挖掘，或用自有日志。
3. **中文电商时间戳**——`C-MTEB/EcomRetrieval` 无时间字段 → 第一轮用中文法规冷启动
   代替（§3.7）。
4. **SNOMED CT 商用**——许可受限 → 默认排除，只用 ICD-10-CM + MeSH。
5. **法语/德语等非中英垂域**——FineWeb-2 有，但本项目 backbone 为中文，暂不投入。

---

## 附：本文引用的实测命令（便于复核）
```
# 存在性 + license + gating
GET https://huggingface.co/api/datasets/{id}
# 字段名 / 行数（权威来源）
GET https://datasets-server.huggingface.co/splits?dataset={id}
GET https://datasets-server.huggingface.co/first-rows?dataset={id}&config={cfg}&split={split}
GET https://datasets-server.huggingface.co/rows?dataset={id}&config={cfg}&split={split}&offset=0&length=1
# 下载体积（用于"取哪些 shard"的决策）
GET https://huggingface.co/api/datasets/{id}/tree/main?recursive=true
```
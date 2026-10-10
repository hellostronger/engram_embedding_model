"""
make_synthetic_data.py — generate a small, fully self-contained dataset so the whole
pipeline (train -> eval -> cold-start -> CRUD) can be exercised with NO network access.

This is NOT for reporting numbers. It exists so that:
  * the data format is pinned down and testable,
  * the A/B/C/D experiment harness can be smoke-tested offline,
  * CI has something to run.

It mimics the structure of a real domain retrieval task: queries sharing a rare
"domain term" with exactly one relevant document, plus lexically-similar distractors.
That structure is what the Engram memory is supposed to exploit.

Usage:
    python make_synthetic_data.py --out data/synth --domain medical --n 2000
"""

from __future__ import annotations

import argparse
import json
import os
import random

DOMAINS = {
    "medical": {
        "terms": ["新型隐球菌脑膜炎", "糖尿病足溃疡", "视网膜静脉阻塞", "特发性肺动脉高压",
                   "肝豆状核变性", "巨细胞动脉炎", "慢性荨麻疹", "阻塞性睡眠呼吸暂停",
                   "自身免疫性肝炎", "侵袭性真菌感染"],
        "templates_q": ["{t}的治疗方案", "{t}有哪些临床表现", "{t}如何诊断", "{t}用什么药",
                        "{t}的预后如何", "关于{t}的最新指南"],
        "templates_p": ["{t}是一种需要规范管理的疾病，其治疗包括对症支持与病因治疗。",
                        "{t}的临床表现以慢性起病为主，诊断依赖实验室与影像学检查。",
                        "针对{t}，临床指南建议早期干预并定期随访评估疗效。"],
        "filler": ["患者", "症状", "检查", "药物", "剂量", "疗程", "并发症", "预后"],
    },
    "finance": {
        "terms": ["可转换公司债券", "资产证券化ABS", "量化中性策略", "永续债",
                   "同业存单", "结构性存款", "信用违约互换", "国债期货基差",
                   "私募股权二级份额", "跨境人民币结算"],
        "templates_q": ["{t}的定义是什么", "{t}如何定价", "{t}有哪些风险",
                        "{t}的投资逻辑", "如何理解{t}"],
        "templates_p": ["{t}是一种常见的金融市场工具，其核心特征体现在流动性与风险结构上。",
                        "关于{t}的定价机制，需要综合考虑利率、信用利差与市场情绪。",
                        "研究{t}时，应关注发行条款、评级变化与二级市场表现。"],
        "filler": ["利率", "利差", "评级", "流动性", "杠杆", "久期", "收益率"],
    },
    "legal": {
        "terms": ["代位求偿权", "善意取得", "表见代理", "情势变更原则", "不可抗力免责",
                   "定金罚则", "违约金调整", "诉讼时效中断", "格式条款解释", "让与担保"],
        "templates_q": ["{t}的构成要件", "{t}如何适用", "{t}有哪些限制",
                        "{t}的法律后果", "关于{t}的裁判规则"],
        "templates_p": ["{t}是合同法中的重要制度，其适用需满足严格的法律要件。",
                        "司法实践中，{t}的认定标准较为严格，需结合具体事实判断。",
                        "关于{t}的适用，司法解释提供了明确的判断依据。"],
        "filler": ["合同", "条款", "当事人", "法院", "判决", "证据", "时效"],
    },
    "ecommerce": {
        "terms": ["无线蓝牙耳机降噪", "人体工学椅护腰", "空气炸锅预热", "电动牙刷声波",
                   "登山鞋防滑", "婴儿纸尿裤透气", "智能手表血氧", "除螨仪紫外线",
                   "破壁机静音", "加湿器上加湿"],
        "templates_q": ["{t}推荐", "{t}哪个牌子好", "{t}多少钱", "{t}怎么选",
                        "{t}值得买吗"],
        "templates_p": ["这款{t}在同类产品中表现出色，用户口碑良好，值得考虑。",
                        "选购{t}时建议关注核心参数与实际使用场景的匹配度。",
                        "{t}属于热销商品，配送速度快，售后保障完善。"],
        "filler": ["销量", "好评", "折扣", "包邮", "旗舰店", "补货", "规格"],
    },
}


def build(out_dir: str, domain: str, n: int, seed: int = 0):
    rng = random.Random(seed)
    cfg = DOMAINS[domain]
    terms = cfg["terms"]

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.jsonl")
    eval_path = os.path.join(out_dir, "eval.jsonl")
    cold_path = os.path.join(out_dir, "cold_start.jsonl")

    # ---- held-out terms for cold start (never used in train/eval) ---------- #
    held_out = [f"新发疾病{i}" for i in range(1, 11)] if domain == "medical" else \
               [f"新兴概念{i}" for i in range(1, 11)]

    # ---- train: only seen terms ------------------------------------------ #
    with open(train_path, "w", encoding="utf-8") as f:
        for i in range(n):
            t = rng.choice(terms)
            q = rng.choice(cfg["templates_q"]).format(t=t)
            p = rng.choice(cfg["templates_p"]).format(t=t)
            # hard negatives: same domain, DIFFERENT term, so the memory for THIS term
            # is what disambiguates -> exactly the signal Engram should provide.
            negs = []
            for _ in range(2):
                t2 = rng.choice([x for x in terms if x != t])
                negs.append(rng.choice(cfg["templates_p"]).format(t=t2))
            f.write(json.dumps({"query": q, "positive": p, "negatives": negs},
                               ensure_ascii=False) + "\n")

    # ---- eval: corpus + queries, 1 gold doc each -------------------------- #
    corpus, queries, pos_idx = [], [], []
    with open(eval_path, "w", encoding="utf-8") as f:
        for i in range(400):
            t = rng.choice(terms)
            q = rng.choice(cfg["templates_q"]).format(t=t)
            gold = rng.choice(cfg["templates_p"]).format(t=t)
            queries.append(q)
            # build the row's corpus: gold + 5 distractors, then shuffle and record
            # the gold's LOCAL index within this row (eval_rows expects local idx)
            docs = [gold]
            for _ in range(5):
                t2 = rng.choice([x for x in terms if x != t])
                docs.append(rng.choice(cfg["templates_p"]).format(t=t2))
            rng.shuffle(docs)
            corpus.append((q, docs, docs.index(gold)))
        for q, docs, gi in corpus:
            f.write(json.dumps({"query": q, "corpus": docs, "gold": gi},
                               ensure_ascii=False) + "\n")

    # ---- cold-start: held-out terms with support contexts ------------------ #
    with open(cold_path, "w", encoding="utf-8") as f:
        for t in held_out:
            ctxs = [f"{t}的临床表现与治疗", f"关于{t}的诊断标准", f"{t}患者的管理策略"]
            f.write(json.dumps({"term": t, "contexts": ctxs}, ensure_ascii=False) + "\n")

    # ---- eval queries restricted to held-out terms (the D-vs-C test) ------- #
    cold_eval = os.path.join(out_dir, "eval_cold.jsonl")
    with open(cold_eval, "w", encoding="utf-8") as f:
        for i in range(200):
            t = rng.choice(held_out)
            q = rng.choice(cfg["templates_q"]).format(t=t)
            gold = rng.choice(cfg["templates_p"]).format(t=t)
            docs = [gold]
            for _ in range(9):
                t2 = rng.choice(held_out + terms)
                docs.append(rng.choice(cfg["templates_p"]).format(t=t2))
            rng.shuffle(docs)
            f.write(json.dumps({"query": q, "corpus": docs, "gold": docs.index(gold)},
                               ensure_ascii=False) + "\n")

    print(f"domain={domain}")
    print(f"  {train_path}  ({n} triplets)")
    print(f"  {eval_path}  (400 queries x 6 docs)")
    print(f"  {cold_path}  ({len(held_out)} held-out terms)")
    print(f"  {cold_eval}  (200 queries on UNSEEN terms)")


def _group(corpus, k):
    return [corpus[i:i + k] for i in range(0, len(corpus), k)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--domain", default="medical", choices=list(DOMAINS))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.out, a.domain, a.n, a.seed)
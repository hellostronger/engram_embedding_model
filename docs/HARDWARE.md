# 硬件支持矩阵与限制

> 结论：**代码不挑卡。** 任何能跑 PyTorch 的环境都能运行。
> 唯一真正的硬性限制是 **bf16 需要 sm_80+**，以及 **显存决定记忆表能开多大**。

## 1. 为什么说它不挑卡

代码里只用标准算子，**没有任何自定义 CUDA kernel / Triton**：

| 用途 | 算子 | 来源 |
|---|---|---|
| 记忆表查表 | `nn.Embedding` | PyTorch |
| 深度因果卷积 | `nn.Conv1d(groups=d)` | PyTorch |
| 归一化 | `nn.RMSNorm` | PyTorch |
| 哈希 | `bitwise_xor` / `remainder` | PyTorch |
| 冷启动闭式解 | `torch.linalg.solve` | PyTorch |
| backbone attention | SDPA（HF 默认） | PyTorch |

因此以下环境都能跑，代码**一行不改**：

- NVIDIA：A100 / H100 / L40S / RTX 4090 / 4080 / 3090 / 3060 / T4 / V100
- NVIDIA GB10 / DGX Spark（Blackwell sm_121，ARM64）
- AMD ROCm、Intel XPU、Apple MPS、纯 CPU

> GB10 的那些注意事项（sm_121 需要 CUDA≥12.8、aarch64 没有 flash-attn 轮子、
> 统一内存不要 pin_memory）全部属于**装 PyTorch** 的问题，与本仓库代码无关。

## 2. 硬性限制清单（按重要性排序）

### 2.1 精度分级 —— 唯一的硬门槛

| 计算能力 | 代表卡 | bf16 | fp16 | 可用最高精度 |
|---|---|---|---|---|
| ≥ sm_80 | A100/H100/L40S/4090/3090/GB10 | ✅ | ✅ | bf16 |
| sm_70–79 | V100 / T4 | ❌ | ✅ | fp16（需 GradScaler） |
| sm < 70 / CPU | 老卡 / CPU | ❌ | ❌ | fp32 |

`default_config(torch_dtype="float32")` 是默认值，所以**在任何设备上都能跑通**，只是慢。

自动探测已实现：`de/runtime.py::recommended_dtype()`。

### 2.2 显存 = 记忆表规模（真正的瓶颈）

这是 DE 与常规 embedding 模型最大的不同：**要 scaling 的是记忆表，不是 backbone**。

Adam(fp32) 每参数 16 字节 = param(4) + grad(4) + m(4) + v(4)：

| preset | 表参数 | 注册表参数 | 可训合计 | 优化器状态显存 |
|---|---|---|---|---|
| tiny | 1.1M | 0.2M | 1.2M | **0.02 GB** |
| small | 16.8M | 3.2M | 20.0M | **0.32 GB** |
| ref | 33.6M | 12.8M | 46.4M | **0.74 GB** |
| medium | 67.2M | 12.8M | 80.0M | **1.28 GB** |
| large | 268.6M | 64.0M | 332.6M | **5.32 GB** |
| xlarge | 537.0M | 128.0M | 665.0M | **10.64 GB** |

**按显存选卡：**

| 显存 | 建议 preset |
|---|---|
| 4 GB (GTX1650/3050) | `small` |
| 6 GB (3060 8G 勉强) | `ref` |
| 8 GB (4060Ti) | `large` |
| 12–16 GB (3060 12G / 4070 / A4000) | `large` |
| 24 GB (3090 / 4090 / L4 / A5000) | `large`，余量充足 |
| 40 GB (A100-40G / L40S) | `xlarge` |
| 80 GB (A100-80G / H100) | `xlarge`，可上调 n_head |
| 128 GB GB10 统一内存 | `xlarge` 仅占 8%，可继续放大 |

> 上表是**参数状态**显存，不含激活与 gather 临时张量。实际峰值需再留 1.5–2× 余量，
> 因此 `auto_preset(headroom=0.5)` 保守选择。

### 2.3 真正的时间瓶颈：dense Adam sweep（不是 gather）

- **gather（查表）成本与表大小无关**：每 token 只取
  `batch × seq × (n_orders × n_head)` 行，约 12 KB/token。
  这正是 Engram "O(1) lookup" 的含义，**表开到 100B 参数也不会让查表变慢**。
- **真正随表大小线性增长的是 Adam 的更新遍历**：
  每 step 要读写全部 `trainable_params × 16` 字节。

粗略交叉点（按 compute agent 估算）：

| 表参数量 | dense Adam 带宽 | 结论 |
|---|---|---|
| ~0.6B | ~35 TF/s | 与计算相当 |
| ~1.1B | ~18 TF/s | Adam 主导 |
| ~2.5B | ~8 TF/s | Adam 严重主导 |

→ **单卡上表开到 ~1B 参数后就该停了**，再大应改用稀疏/分片更新。

### 2.4 尚未实现的能力（明确声明，避免踩坑）

| 未实现 | 影响 | 说明 |
|---|---|---|
| 张量并行 / 多卡分片 | **单卡显存即硬上限** | Engram 原文训练时用 All-to-All 把表分到多卡；本仓库是单进程单卡 |
| CPU offload / 预取引擎 | 大表需降 preset | 论文 §6.4「100B 表 offload 仅 -2.8%」**未实现**，需要改推理引擎；GB10 因统一内存影响很小，24GB 卡上需手动缩表 |
| 自定义 CUDA kernel | 查表走通用 gather | 性能上不最优，但不影响正确性 |
| LoRA | 只能全冻结或全解冻 | `freeze_backbone=True` 默认；`EXPERIMENTS.md` 的 LoRA 基线待写 |

### 2.5 CPU 会先撞墙的地方

`PhraseMatcher` 在 numpy（CPU）执行，每个 batch 都要跑。已设 `num_workers=4` 缓解。
若 batch 大 / 序列长 / 注册表大，CPU matcher 会成为瓶颈。缓解方式：
- 调小 batch；
- 或把 matcher 移到 GPU（需重写，用 `searchsorted` + `scatter`）。

## 3. 复现与验证状态（诚实声明）

- ✅ `offline_test.py`：CPU、离线、tiny BERT，通过即可证明
  哈希确定性 / 逐表索引范围 / CRUD / NULL 惰性 / 门控范围 / 冷启动非零 / 训练降loss。
- ❌ **尚未在真实 GB10 / GPU 上跑过**。`docs/COMPUTE.md` 中所有吞吐数字均为**推算**，
  首次上机必须先用 `run_gb10.py --stage bench` 校准。
- ❌ 尚未跑过真实数据集（HF 下载在沙箱被代理拦截）。
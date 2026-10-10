"""
runtime.py — device capability detection and automatic configuration selection.

The Domain Engram code uses only standard PyTorch ops (nn.Embedding, nn.Conv1d,
RMSNorm, scaled_dot_product_attention, linalg.solve), so it runs on any CUDA device,
on CPU, and on GB10/Blackwell. What actually varies by GPU is:

  1. dtype support : bf16 needs compute capability >= 8.0 (Ampere). fp16 needs >= 7.0.
                     Anything older must use fp32.
  2. VRAM budget   : the Engram memory tables live in VRAM during training. A 24 GB
                     4090 and a 128 GB GB10 differ by ~5x in how large a table is sane.
                     (On unified-memory boxes such as GB10, "CPU offload" is free; on a
                     discrete GPU it is not, so we simply size tables to fit VRAM.)
  3. arch support  : nothing in this repo is arch-specific. flash-attn is NOT used.

This module probes the runtime and picks a safe configuration, so the same code runs
unchanged on: GB10 / DGX Spark, A100, H100, L40S, RTX 4090, RTX 3060, T4, or CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DeviceProfile:
    device_type: str          # "cuda" | "cpu"
    name: str
    capability: Optional[tuple]   # (major, minor) e.g. (12, 1) for sm_121
    total_mem_gb: float
    supports_bf16: bool
    supports_fp16: bool
    is_unified_memory: bool   # GB10 / DGX Spark class: CPU and GPU share the pool

    @property
    def sm(self) -> str:
        if self.capability is None:
            return "cpu"
        return f"sm_{self.capability[0]}{self.capability[1]}"


def detect_device(device: Optional[str] = None) -> DeviceProfile:
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        return DeviceProfile("cpu", "CPU", None, 0.0, False, False, False)

    cap = torch.cuda.get_device_capability()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    name = torch.cuda.get_device_name(0)
    major = cap[0]

    # GB10 / DGX Spark and Grace-Hopper parts expose very large "GPU" memory that is
    # really unified system memory; treat >=96 GB as unified for sizing purposes.
    unified = total_gb >= 96.0 or "GB10" in name or "DGX Spark" in name

    return DeviceProfile(
        device_type="cuda",
        name=name,
        capability=cap,
        total_mem_gb=total_gb,
        supports_bf16=(major >= 8),
        supports_fp16=(major >= 7),
        is_unified_memory=unified,
    )


def recommended_dtype(profile: DeviceProfile, prefer_bf16: bool = True):
    """bf16 on Ampere+, else fp16 on Volta+, else fp32. (fp16 needs a GradScaler.)"""
    import torch
    if profile.device_type == "cpu":
        return torch.float32
    if prefer_bf16 and profile.supports_bf16:
        return torch.bfloat16
    if profile.supports_fp16:
        return torch.float16
    return torch.float32


# --------------------------------------------------------------------------- #
# VRAM-aware preset selection
# --------------------------------------------------------------------------- #
# (table_size_per_ngram, d_head, n_head_per_ngram, n_layers, max_slots, d_registry)
PRESETS = {
    "tiny":   (4096,   16, 4, 2,   5_000,   32),
    "small":  (16384,  32, 8, 2,  50_000,   64),
    "ref":    (32768,  32, 8, 2, 200_000,   64),
    "medium": (65536,  32, 8, 2, 200_000,   64),
    "large":  (131072, 64, 8, 2, 500_000,  128),
    "xlarge": (262144, 64, 8, 2, 1_000_000, 128),
}


def _estimate_table_params(table_size: int, d_head: int, n_head: int, n_layers: int,
                           n_orders: int = 2) -> int:
    """Rows ~= n_layers * n_orders * n_head * (table_size + small prime slack)."""
    return n_layers * n_orders * n_head * (table_size + 64) * d_head


def estimate_vram_gb(preset: str, n_layers: int = 4) -> dict:
    """Rough VRAM estimate for a preset. Backbone activations are NOT included."""
    table_size, d_head, n_head, _n_inj, slots, d_reg = PRESETS[preset]
    n_inj = 2
    table_params = _estimate_table_params(table_size, d_head, n_head, n_inj)
    registry_params = slots * d_reg
    fusion_params = n_inj * (2 * (512 + table_size * 0 + 0))  # rough; see describe()
    total_trainable = table_params + registry_params
    # param + grad + Adam(m,v) = 16 bytes/param in fp32
    return {
        "table_params": table_params,
        "registry_params": registry_params,
        "trainable_params": total_trainable,
        "vram_gb_fp32_16bytes_per_param": total_trainable * 16 / 1e9,
        "vram_gb_bf16_master_fp32": total_trainable * 16 / 1e9,
    }


def auto_preset(profile: DeviceProfile, headroom: float = 0.5) -> str:
    """Pick the largest preset whose trainable-parameter state fits in VRAM.

    headroom leaves room for activations of the backbone + the gathered e_mem tensors.
    """
    if profile.device_type == "cpu":
        return "tiny"
    budget = profile.total_mem_gb * headroom
    order = ["tiny", "small", "ref", "medium", "large", "xlarge"]
    chosen = "tiny"
    for name in order:
        est = estimate_vram_gb(name)["vram_gb_fp32_16bytes_per_param"]
        if est <= budget:
            chosen = name
        else:
            break
    return chosen


def describe_profile(profile: DeviceProfile) -> str:
    if profile.device_type == "cpu":
        return (f"device=CPU  bf16={False} fp16={False}\n"
                f"  -> use preset 'tiny' (CPU smoke tests only; not for real training)")
    return (
        f"device={profile.name}\n"
        f"  arch            : {profile.sm} (capability {profile.capability})\n"
        f"  memory          : {profile.total_mem_gb:.0f} GB "
        f"({'unified' if profile.is_unified_memory else 'discrete'})\n"
        f"  bf16 / fp16     : {profile.supports_bf16} / {profile.supports_fp16}\n"
        f"  recommended dtype: {recommended_dtype(profile)}\n"
        f"  auto preset     : {auto_preset(profile)}\n"
    )
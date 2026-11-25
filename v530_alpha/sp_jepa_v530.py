# -*- coding: utf-8 -*-
"""
SP-JEPA: Self-Play Joint Embedding Predictive Architecture
======================================================================
Model Name: FengXiao (风小)
A single-file, hardware-adaptive, cross-platform training script.

Revision History:
- v1 ~ v4.4: (略，见历史文档)
- v5.2.0: 0号方案实验结论驱动的训练层修复（表征坍缩）:
        (a) NEW covariance_loss: 协方差非对角抑制（VICReg covariance
            term），直接正则二阶谱。0 号方案实测：一阶均值分量
            |mean|=0.97 贡献假阳性，但剩余二阶结构（EffRank
            ~24/128）无块语义——边缘正则（scale/shape）管不到
            维度间相关性，此为病灶。
        (b) VISRegLoss 新增 lambda_decorr=0.005（标准化 z 上计算，
            项值 ~0.055 与 shape 同量级；量级由 v5.1.1 checkpoint
            离线定标：cov_offdiag/D ≈ 11）。
        (c) 分量观测：last_scale/last_shape/last_decorr +
            epoch 级 Decorr 打印 + history["decorr"]。
        (d) 施加位置 pred_m（与现有 VISReg 同点），经 EMA 传导
            target_encoder（检索层用 target）。
- v5.0.0: Cognitive upgrade (multi-step + self-eval head).
- v5.0.1: Fixes after review:
        (a) make_span_mask rewritten.
        (b) Restored per-sample mask.
        (c) per_sample_err vectorized.
        (d) lambda_multi / lambda_conf default to 0.0.
        (e) confidence target uses exp(-err).
- v5.0.2: (P0) Fixed EMA silent collapse.
- v5.0.3: Three fixes:
        (a) multi-step loss DISABLED.
        (b) make_span_mask: exact mask count control.
        (c) update_ema docstring clarified.
- v5.1.0: (保守迭代) Targeted, behavior-preserving improvements:
        (a) confidence_head: batch-wise z-score normalized target,
            prevents signal collapse into a ~0.93 narrow band.
        (b) Explorer.choose: z-score normalize difficulty before
            softmax -> tau becomes a meaningful temperature.
        (c) mask ratio lower bound raised: uniform(0.3*r, r).
            Eliminates wasted zero-mask steps.
        (d) ema_health: switched from parameter-space (single tensor)
            to output-space probe on a fixed random batch.
        (e) history.json written every epoch (crash-safe).
        (f) New representation_health diagnostic (per-dim std +
            effective rank) to catch collapse beyond JEPA loss.
        (g) Removed unused num_workers from ENV.
        (h) All forward() return values are detached scalars.
- v5.1.1: Two surgical fixes after sandbox test (30E run, best val
        0.0608, resume verified 3x across process kills):
        (a) VISRegLoss now wired to CONFIG: lambda_scale /
            lambda_shape / num_projections were silently ignored
            (constructor used defaults).
        (b) representation_health: fixed-seed val subset, no
            global-RNG consumption. Diagnostics on/off no longer
            changes the training stream; EffRank comparable
            across epochs.

Status: EMA alive, per-sample scheduled masking, val locked, env-aware.
        Cognitive: self-eval head with normalized target;
                   multi-step permanently disabled.

Usage:
1. Place `input_v530.txt` (UTF-8) + `identity_zh_paraphrase_v530.tsv` in dir.
2. python sp_jepa_v530.py       # resume if checkpoint exists
   python sp_jepa_v530.py fresh # train from scratch

v5.3.0 (2026-11-18) — 对比学习版, 基于 sp_jepa.py v5.2.0 修改:
  [依据] 白化排除实验(CHANGELOG_rescue_exp_2026-11-18.md): 后处理无解,
         语义从未被编码 → 训练目标加"谁该靠近谁"。
  1. context_encoder dropout 0.0 -> contrast_dropout(0.1);
     target_encoder 保持 0.0(EMA/检索侧无噪声)。
  2. 对比项 contrast loss(检索形态): paraphrase 对(TSV 监督正例)
     + SimCSE 式 dropout 正例(同句两次前向) + batch 内负例,
     InfoNCE, 独立 backward 与 JEPA loss 梯度累加。
  3. 语料 input_v530.txt = 原 159K + 节气教师语料 + 145 对 paraphrase;
     vocab 2214 -> 2264(节气生僻字自动并入)。
  4. 每 sep_probe_every epochs 用 target_encoder 跑 sep probe
     (calibrate 同款查询组), 按 sep 存 best_sep checkpoint
     ——监督回路: sep 而非 val loss 是检索的裁判。
"""

import os
import random
import time
import json
import contextlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

__version__ = "5.3.0"


# =====================================================================
#  0. Environment & Hardware Detection
# =====================================================================
def detect_env():
    info = {
        "torch_version": torch.__version__,
        "cuda_built":     torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu", "device_name": "CPU",
        "vram_gb": 0, "ram_gb": 8,
        "use_amp": False,
        "batch_size": 8, "embed_dim": 192, "num_layers": 3, "seq_len": 96,
        "device_tag": "CPU",
    }

    try:
        import psutil
        info["ram_gb"] = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

    if info["cuda_built"] is not None and info["cuda_available"]:
        info["device"] = "cuda"
        info["device_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info["use_amp"] = True
        info["device_tag"] = "GPU"

        if info["vram_gb"] >= 20:
            info.update({"batch_size": 32, "embed_dim": 384, "num_layers": 6, "seq_len": 256})
        elif info["vram_gb"] >= 10:
            info.update({"batch_size": 24, "embed_dim": 384, "num_layers": 4, "seq_len": 192})
        elif info["vram_gb"] >= 5:
            info.update({"batch_size": 16, "embed_dim": 256, "num_layers": 4, "seq_len": 128})
        else:
            info.update({"batch_size": 8, "embed_dim": 192, "num_layers": 3, "seq_len": 96})

    elif info["cuda_built"] is not None and not info["cuda_available"]:
        info["device_tag"] = "CPU (CUDA build, no GPU)"
        print("[ENV] 检测到 CUDA 编译版 torch，但当前机器没有可用的 GPU。")
        print("      将回退到 CPU 训练。\n")
        if info["ram_gb"] >= 16:
            info.update({"batch_size": 8, "embed_dim": 192, "num_layers": 3, "seq_len": 96})
        else:
            info.update({"batch_size": 4, "embed_dim": 128, "num_layers": 2, "seq_len": 64})

    else:
        info["device_tag"] = "CPU (CPU-only build)"
        print("[ENV] 当前 torch 为 CPU-only 编译版。")
        print("      如想启用 GPU 训练，请到 pytorch.org 重装 CUDA 版。\n")
        if info["ram_gb"] >= 16:
            info.update({"batch_size": 8, "embed_dim": 192, "num_layers": 3, "seq_len": 96})
        else:
            info.update({"batch_size": 4, "embed_dim": 128, "num_layers": 2, "seq_len": 64})

    return info


ENV = detect_env()
HARDWARE = ENV
DEVICE = torch.device(ENV["device"])

CONFIG = {
    "data_path": "input_v530.txt",
    "save_dir": "./checkpoints_v530",
    "para_tsv": "identity_zh_paraphrase_v530.tsv",
    "lambda_contrast": 0.5,
    "contrast_tau": 0.05,
    "contrast_dropout": 0.1,
    "contrast_pairs": 8,
    "sep_probe_every": 5,
    # v5.3.0-alpha: 12 epochs 先验证 sep 趋势(对比信号收敛快于填空任务);
    # sep 转正后再跑完整训练
    "epochs": 12,
    "seq_len": ENV["seq_len"],
    "embed_dim": ENV["embed_dim"],
    "num_heads": max(4, ENV["embed_dim"] // 64),
    "num_layers": ENV["num_layers"],
    "batch_size": ENV["batch_size"],
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "max_grad_norm": 1.0,
    "mask_ratio": 0.3,
    "mask_ratio_min_factor": 0.3,   # v5.1.0: 实际 ratio 下界 = mask_ratio * 此值
    "val_ratio": 0.05,
    "seed": 42,
    "ema_start": 0.990,
    "ema_end": 0.996,
    "warmup_epochs": 5,
    "lambda_scale": 1.0,
    "lambda_shape": 1.0,
    "lambda_decorr": 0.0,
    "lambda_center": 0.0,
    "lambda_sent_center": 0.5,
    "lambda_sent_variance": 1.0,
    "lambda_sent_decorr": 0.05,
    "num_projections": 128,
    "explorer_tau": 0.5,
    "explorer_momentum": 0.7,
    "explorer_enabled": True,
    "use_amp": ENV["use_amp"],
    "log_interval": 10,
    "save_interval": 10,
    # --- 认知损失 ---
    # v5.0.3: lambda_multi 已废弃（实现不成立）。保持 0。
    #          lambda_conf 可用，建议先跑 0 基线，再开 0.2 对比。
    "lambda_multi": 0.0,
    "lambda_conf":  0.0,
    # v5.1.0: 表征健康度诊断开关（几乎零成本，建议开）
    "repr_health_enabled": True,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
#  1. Dataset (Global shuffle, then split)
# =====================================================================
class TextDataset:
    def __init__(self, path, seq_len, val_ratio=0.05):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if len(text) < seq_len * 10:
            raise ValueError("Text too short.")

        self.chars = sorted(list(set(text)))
        self.mask_token = len(self.chars)
        self.vocab_size = len(self.chars) + 1
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}

        data = torch.tensor([self.char2idx[ch] for ch in text], dtype=torch.long)
        n_seq = len(data) // seq_len
        data = data[:n_seq * seq_len].view(n_seq, seq_len)

        perm = torch.randperm(n_seq)
        data = data[perm]

        n_val = max(1, int(n_seq * val_ratio))
        self.train = data[n_val:]
        self.val = data[:n_val]
        print(f"[DATA] Chars: {len(text):,} | Vocab: {self.vocab_size} "
              f"| Train: {len(self.train)} | Val: {len(self.val)}")

    def batches(self, batch_size, split="train"):
        d = self.train if split == "train" else self.val
        indices = torch.randperm(len(d)) if split == "train" else torch.arange(len(d))
        for i in range(0, len(d) - batch_size + 1, batch_size):
            yield d[indices[i: i + batch_size]]


def make_span_mask(seq_len, mask_ratio):
    """
    随机 span 掩码，精确控制实际掩码数 = num_mask。
    v5.0.3: 逐个采样起点 + 1-2 长度扩展，若超量则截断。
    v5.1.0: 由调用方保证 mask_ratio > 0（见 train 循环）。
    """
    mask = torch.zeros(seq_len, dtype=torch.bool)
    num_mask = int(seq_len * mask_ratio)
    if num_mask <= 0:
        return mask

    count = 0
    max_attempts = seq_len * 2
    attempts = 0
    while count < num_mask and attempts < max_attempts:
        attempts += 1
        t = random.randint(0, seq_len - 1)
        if mask[t]:
            continue
        length = random.randint(1, 2)
        end = min(seq_len, t + length)
        allowed = num_mask - count
        end = min(end, t + allowed)
        newly = int((~mask[t:end]).sum().item())
        if newly == 0:
            continue
        mask[t:end] = True
        count += newly
    return mask


def make_span_mask_batch(B, seq_len, mask_ratio):
    """per-sample mask, shape (B, L)。"""
    return torch.stack([make_span_mask(seq_len, mask_ratio) for _ in range(B)])


# =====================================================================
#  2. VISReg Loss
# =====================================================================
def variance_loss(z, eps=1e-4):
    std_z = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std_z))


def covariance_loss(z):
    """
    v5.2.0: 协方差非对角抑制（VICReg covariance term）。
    z: (N, D)。推动维度间去相关 → 谱白化 → EffRank ↑。
    注意：本项只管 batch 内二阶结构；跨 batch 公共方向（一阶坍缩）
    由 center_loss 管（实测 v5.1.1: batch 均值 |mu|≈51 vs std≈2，
    8 批均值方向余弦 1.000——96% 能量在公共方向上）。
    标准化 z 上计算，量纲稳定。
    """
    N, D = z.shape
    if N < 2:
        return z.sum() * 0.0
    zc = z - z.mean(dim=0)
    cov = (zc.t() @ zc) / (N - 1)
    # 归一化为相关矩阵（对角=1），只罚非对角
    std = torch.sqrt(torch.diagonal(cov).clamp(min=1e-8))
    corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))
    off = corr - torch.diag(torch.diagonal(corr))
    return off.pow(2).sum() / D


def center_loss(z, eps=1e-4):
    """
    v5.2.0: 批均值 z-score 模长惩罚（一阶/跨 batch 坍缩）。
    直接压"每个 batch 的公共方向"——0 号方案实测 |mean|=0.97
    的训练层根源。若跨 batch 公共分量存在，各批标准化均值
    同向且模长大 → 逐 batch 惩罚有效（线性性：位置向量减公共
    分量后，mean pooling 句向量同样减去）。
    量级参考（v5.1.1 checkpoint 离线）：~4.5。
    """
    mu = z.mean(dim=0)
    sd = z.std(dim=0).clamp(min=eps)
    return (mu / sd).pow(2).mean()


def sliced_wasserstein_distance(z, num_projections=128):
    batch_size, dim = z.shape
    proj = torch.randn(dim, num_projections, device=z.device)
    proj = F.normalize(proj, dim=0)
    projections = z @ proj
    sorted_proj, _ = torch.sort(projections, dim=0)
    quantiles = torch.linspace(1, batch_size, batch_size, device=z.device) / (batch_size + 1)
    target = torch.erfinv(2 * quantiles - 1) * (2 ** 0.5)
    return F.mse_loss(sorted_proj, target.unsqueeze(1).expand_as(sorted_proj))


class VISRegLoss(nn.Module):
    def __init__(self, lambda_scale=1.0, lambda_shape=1.0, num_projections=128,
                 lambda_decorr=0.0, lambda_center=0.0):
        super().__init__()
        self.lambda_scale = lambda_scale
        self.lambda_shape = lambda_shape
        self.lambda_decorr = lambda_decorr
        self.lambda_center = lambda_center
        self.num_projections = num_projections
        # 分量观测（epoch 聚合/日志用；单步值，forward 后更新）
        self.last_scale = 0.0
        self.last_shape = 0.0
        self.last_decorr = 0.0
        self.last_center = 0.0

    def forward(self, z):
        loss_scale = variance_loss(z)
        z_norm = (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-6)
        loss_shape = sliced_wasserstein_distance(z_norm, self.num_projections)
        # v5.2.0a: 去相关项（batch 内二阶）。
        # v5.2.0b: 中心项（跨 batch 一阶）——batch 内中心化会放走
        # 公共方向（实测 |mu|/std ≈ 25:1），必须显式惩罚批均值。
        loss_decorr = covariance_loss(z_norm) if self.lambda_decorr > 0 else (
            z.sum() * 0.0)
        loss_center = center_loss(z) if self.lambda_center > 0 else (
            z.sum() * 0.0)
        self.last_scale = float(loss_scale.detach())
        self.last_shape = float(loss_shape.detach())
        self.last_decorr = float(loss_decorr.detach())
        self.last_center = float(loss_center.detach())
        return (self.lambda_scale * loss_scale
                + self.lambda_shape * loss_shape
                + self.lambda_decorr * loss_decorr
                + self.lambda_center * loss_center)


# =====================================================================
#  3. Explorer (Error-Driven Self-Play Sampler)
# =====================================================================
class Explorer:
    def __init__(self, num_seqs, tau=0.5, momentum=0.7):
        self.difficulty = torch.zeros(num_seqs)
        self.tau = tau
        self.momentum = momentum

    def choose(self, pool, k):
        if len(pool) <= 1:
            return pool[:k]
        pool_diff = self.difficulty[pool]
        # v5.1.0: z-score 归一化 —— 让 tau 在不同误差量级下语义稳定。
        # 归一化后，池内难度 z∈[-2, 2] 左右；tau=0.5 时最易/最难样本
        # 权重比 ≈ exp(8) ≈ 3000，区分度足够且不爆。
        d_mean = pool_diff.mean()
        d_std = pool_diff.std().clamp(min=1e-6)
        pool_diff_z = (pool_diff - d_mean) / d_std
        weights = torch.softmax(pool_diff_z / self.tau, dim=0)
        k = min(k, len(pool))
        return pool[torch.multinomial(weights, k, replacement=False)]

    def update(self, indices, per_sample_errors, rewards=None):
        if rewards is not None:
            per_sample_errors = per_sample_errors - rewards * 0.5
        old = self.difficulty[indices]
        self.difficulty[indices] = self.momentum * old + (1 - self.momentum) * per_sample_errors


# =====================================================================
#  4. SP-JEPA Model
# =====================================================================
class SP_JEPA(nn.Module):
    def __init__(self, vocab_size, mask_token, embed_dim, num_heads, num_layers, seq_len):
        super().__init__()
        self.mask_token = mask_token
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)

        def make_enc(dropout=0.0):
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 4, dropout=dropout,
                batch_first=True, norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=num_layers)

        # v5.3.0: context 侧开 dropout(SimCSE 正例来源),
        # target 侧保持 0(EMA/检索编码无噪声)
        self.context_encoder = make_enc(CONFIG["contrast_dropout"])
        self.target_encoder = make_enc(0.0)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # 自评头
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid(),
        )

        # v5.1.1: 接入 CONFIG（原先用默认值，改 CONFIG 不生效）
        # v5.2.0: lambda_decorr 同步接线（吸取 v5.1.1 教训，一次到位）
        self.visreg_loss = VISRegLoss(
            CONFIG["lambda_scale"], CONFIG["lambda_shape"],
            CONFIG["num_projections"], CONFIG["lambda_decorr"],
            CONFIG["lambda_center"])
        self.current_momentum = CONFIG["ema_start"]
        self._warned_multi = False

        # v5.1.0: EMA health 探针 —— 固定随机 token 序列（独立于全局种子）。
        # 放在 buffer 里（persistent=False 不进 state_dict）。
        probe_len = min(seq_len, 32)
        g = torch.Generator().manual_seed(12345)
        probe_x = torch.randint(0, max(vocab_size - 1, 1), (4, probe_len), generator=g)
        self.register_buffer("_probe_x", probe_x, persistent=False)

    def forward(self, x, mask):
        """
        x    : (B, L)
        mask : (B, L)  per-sample
        返回 : total, loss_jepa, loss_visreg, per_sample_err, loss_multi, loss_conf
        所有 loss 均为 detach 后的标量 tensor。
        """
        B, L = x.shape
        pos = self.pos_embed[:, :L, :]
        ctx_x = x.masked_fill(mask, self.mask_token)

        ctx_emb = self.context_encoder(self.token_embed(ctx_x) + pos)
        with torch.no_grad():
            tgt_emb = self.target_encoder(self.token_embed(x) + pos)

        pred_emb = self.predictor(ctx_emb)

        pred_m, tgt_m = pred_emb[mask], tgt_emb[mask]

        zero = pred_emb.sum() * 0.0
        if pred_m.size(0) == 0:
            return (zero, zero.detach(), zero.detach(),
                    torch.zeros(B, device=x.device), zero.detach(), zero.detach())

        # --- 1. 基础 JEPA loss ---
        loss_jepa = F.smooth_l1_loss(
            F.normalize(pred_m, dim=-1),
            F.normalize(tgt_m, dim=-1),
        )

        # --- 2. 多步一致性 loss (v5.0.3 起永久禁用) ---
        loss_multi = zero.detach()
        if CONFIG["lambda_multi"] > 0.0 and not self._warned_multi:
            print("[WARN] lambda_multi > 0 but multi-step loss is not implemented. "
                  "Ignoring. See v5.0.3 changelog.")
            self._warned_multi = True

        # --- 3. 自评 loss ---
        loss_conf = zero.detach()
        if CONFIG["lambda_conf"] > 0.0:
            conf = self.confidence_head(pred_m)
            with torch.no_grad():
                true_err = F.smooth_l1_loss(
                    F.normalize(pred_m, dim=-1),
                    F.normalize(tgt_m, dim=-1),
                    reduction="none",
                ).mean(dim=-1, keepdim=True)  # (N, 1)
                # v5.1.0: batch 内 z-score -> sigmoid(-z)，把目标从
                # [0.86, 0.96] 的窄带拉伸到 (0, 1) 全域，梯度信号显著。
                if true_err.size(0) >= 2:
                    err_mean = true_err.mean()
                    err_std = true_err.std().clamp(min=1e-6)
                    err_z = (true_err - err_mean) / err_std
                else:
                    err_z = torch.zeros_like(true_err)
                true_conf = torch.sigmoid(-err_z)  # (N, 1), ∈ (0, 1)
            loss_conf = F.smooth_l1_loss(conf, true_conf)

        # --- 4. VISReg（pred_m 位置级）---
        loss_visreg = self.visreg_loss(pred_m)

        # --- 4b. v5.2.0c 句级表征正则（检索同形态直接正则）---
        # 实验链：pred_m 位置级正则隔 pooling/normalize/EMA 三层，
        # 实测治不了句向量方向共线（R3: online Center=0.006 但
        # target |mean|=0.947）。改为直接对 context_encoder 输出
        # 的 mean pooling + normalize 句向量（= 检索管线的精确形态）
        # 施加 center/decorr。loss_sent_center 是 0 号方案 |mean|
        # 的训练时版本；梯度直达 context_encoder，经 EMA 参数级
        # 传导 target_encoder（检索侧）。
        sent_vec = ctx_emb.mean(dim=1)                    # (B, D) raw
        sent_n = F.normalize(sent_vec, dim=-1)             # 检索形态
        # v5.2.0d: 句级三件套 center+variance+decorr（VICReg 移植）。
        # variance 防零和作弊：句间 std 下界，强制句向量真张开。
        loss_sent_center = sent_n.mean(dim=0).norm()
        loss_sent_var = variance_loss(sent_vec)
        if B > 1 and CONFIG["lambda_sent_decorr"] > 0:
            loss_sent_decorr = covariance_loss(sent_n)
        else:
            loss_sent_decorr = zero
        # 挂到 visreg_loss 上统一观测（train 循环读取）
        self.visreg_loss.last_sent_center = float(loss_sent_center.detach())
        self.visreg_loss.last_sent_decorr = float(loss_sent_decorr.detach())
        self.visreg_loss.last_sent_var = float(loss_sent_var.detach())

        # --- 总 loss ---
        total = (
            loss_jepa
            + CONFIG["lambda_multi"] * loss_multi
            + CONFIG["lambda_conf"] * loss_conf
            + loss_visreg
            + CONFIG["lambda_sent_center"] * loss_sent_center
            + CONFIG["lambda_sent_variance"] * loss_sent_var
            + CONFIG["lambda_sent_decorr"] * loss_sent_decorr
        )

        # --- 5. per-sample error ---
        with torch.no_grad():
            diff = F.smooth_l1_loss(
                F.normalize(pred_emb, dim=-1),
                F.normalize(tgt_emb, dim=-1),
                reduction="none",
            ).mean(dim=-1)
            mask_f = mask.float()
            counts = mask_f.sum(dim=1).clamp(min=1.0)
            per_sample_err = (diff * mask_f).sum(dim=1) / counts

        return (total,
                loss_jepa.detach(),
                loss_visreg.detach(),
                per_sample_err,
                loss_multi.detach(),
                loss_conf.detach())

    @torch.no_grad()
    def update_ema(self, global_step, total_steps):
        """
        动量按全局步数 warmup（前 10% 步）。
        - global_step=0     → m = ema_start (0.990)
        - global_step=10%N  → m = ema_end   (0.996)
        - 之后恒定 0.996
        """
        t = min(global_step / max(total_steps * 0.1, 1), 1.0)
        m = CONFIG["ema_start"] + (CONFIG["ema_end"] - CONFIG["ema_start"]) * t
        self.current_momentum = m
        for p_t, p_c in zip(self.target_encoder.parameters(),
                            self.context_encoder.parameters()):
            p_t.mul_(m).add_(p_c, alpha=1 - m)

    @torch.no_grad()
    def ema_health(self):
        """
        v5.1.0: 输出空间 EMA 漂移探针。
        用固定随机 token 序列喂两个 encoder，比较归一化输出的 smooth-L1。
        - 训练初期 ≈ 0（两个 encoder 完全相同）
        - 若长期 → 0 则 EMA 未生效（collapse 静默）
        - 若过大（> 0.5）则 EMA 滞后严重
        """
        x = self._probe_x
        if x.device != self.pos_embed.device:
            x = x.to(self.pos_embed.device)
        L = x.shape[1]
        pos = self.pos_embed[:, :L, :]
        emb = self.token_embed(x) + pos
        ctx_out = self.context_encoder(emb)
        tgt_out = self.target_encoder(emb)
        return F.smooth_l1_loss(
            F.normalize(ctx_out, dim=-1),
            F.normalize(tgt_out, dim=-1),
        ).item()


# =====================================================================
#  5. AMP Compatibility Shim
# =====================================================================
def build_amp_handles(use_amp, device_type):
    if not use_amp:
        return contextlib.nullcontext(), None

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    try:
        ctx = torch.amp.autocast("cuda", enabled=True)
    except (TypeError, AttributeError):
        ctx = torch.cuda.amp.autocast(enabled=True)

    return ctx, scaler


# =====================================================================
#  6. Diagnostics
# =====================================================================
@torch.no_grad()
def representation_health(model, dataset, n_samples=128):
    """
    v5.1.0: 在未掩码的 val 序列上采样，看 context_encoder 输出的统计量。
    返回 (std_per_dim_mean, effective_rank, dim)。
    - effective_rank = exp(entropy(cov_eigenvalues))
      均匀分布 → ≈ dim；全塌缩 → ≈ 1。
    - std_per_dim_mean 逐维标准差均值，< 0.01 通常预示塌缩。
    """
    was_training = model.training
    model.eval()
    try:
        d = dataset.val
        n = min(n_samples, len(d))
        if n < 2:
            return 0.0, 0.0, model.pos_embed.shape[-1]
        # v5.1.1: 固定子采样种子 —— 诊断不再消耗全局 RNG（开/关诊断
        # 不再改变训练随机流），且 EffRank 估计逐 epoch 可比、不再抖动。
        g = torch.Generator().manual_seed(1234)
        idx = torch.randperm(len(d), generator=g)[:n]
        x = d[idx].to(DEVICE)
        L = x.shape[1]
        pos = model.pos_embed[:, :L, :]
        emb = model.token_embed(x) + pos
        out = model.context_encoder(emb)          # (N, L, D)
        seq_repr = out.mean(dim=1)                # (N, D)
        std_per_dim = seq_repr.std(dim=0).mean().item()

        centered = seq_repr - seq_repr.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / max(n - 1, 1)
        eigvals = torch.linalg.eigvalsh(cov).clamp(min=1e-10)
        p = eigvals / eigvals.sum()
        eff_rank = torch.exp(-(p * (p + 1e-12).log()).sum()).item()
        dim = seq_repr.shape[1]
    finally:
        if was_training:
            model.train()
    return std_per_dim, eff_rank, dim


# =====================================================================
#  7. Evaluation & Training
# =====================================================================
def evaluate(model, dataset, val_masks):
    model.eval()
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for i, x in enumerate(dataset.batches(CONFIG["batch_size"], split="val")):
            if i >= len(val_masks):
                break
            x = x.to(DEVICE)
            mask = val_masks[i].to(DEVICE)
            loss, _, _, _, _, _ = model(x, mask)
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


# =====================================================================
#  4b. v5.3.0 Contrastive components (检索形态监督)
# =====================================================================
_crng = random.Random(20261118)   # 独立 RNG: 采样 paraphrase 不污染训练流


def load_para_pool(tsv_path, chars, seq_len):
    """TSV 加载 paraphrase 对; 两端全在 vocab 且不超长才收录。"""
    vocab = set(chars)
    pool = []
    if not os.path.exists(tsv_path):
        print(f"[CONTRAST] WARN: {tsv_path} not found -> contrast disabled")
        return pool
    for raw in open(tsv_path, encoding="utf-8"):
        raw = raw.rstrip("\n")
        if not raw.strip():
            continue
        cols = raw.split("\t")
        if len(cols) != 2 or not cols[0].strip() or not cols[1].strip():
            continue
        a, b = cols[0].strip(), cols[1].strip()
        if len(a) > seq_len or len(b) > seq_len:
            continue
        if any(c not in vocab for c in a + b):
            continue
        pool.append((a, b))
    return pool


def contrast_forward(model, para_pool, ch2idx, cfg):
    """SimCSE dropout 正例 + paraphrase 正例 + batch 内负例 -> InfoNCE。

    检索形态: 无 mask 前向 -> 有效位置 mean pool -> normalize。
    返回标量 loss(None: 池空/关闭)。
    """
    if not para_pool or cfg["lambda_contrast"] <= 0:
        return None
    n = min(cfg["contrast_pairs"], len(para_pool))
    pairs = _crng.sample(para_pool, n)
    sents = [s for p in pairs for s in p]
    N = len(sents)
    if N < 4:
        return None
    device = next(model.parameters()).device
    lengths = torch.tensor([min(len(s), cfg["seq_len"]) for s in sents])
    maxlen = int(lengths.max())
    pad_id = model.mask_token
    xb = torch.full((N, maxlen), pad_id, dtype=torch.long)
    for r, s in enumerate(sents):
        ids = [ch2idx[c] for c in s if c in ch2idx][:cfg["seq_len"]]
        xb[r, :len(ids)] = torch.tensor(ids)
    xb = xb.to(device)
    pad_mask = (torch.arange(maxlen)[None, :] >= lengths[:, None]).to(device)
    keep = (~pad_mask).unsqueeze(-1).float()

    def sent_pass():
        emb = model.token_embed(xb) + model.pos_embed[:, :maxlen, :]
        h = model.context_encoder(emb, src_key_padding_mask=pad_mask)
        return F.normalize((h * keep).sum(1) / keep.sum(1), dim=-1)

    z1 = sent_pass()          # 两次前向, context dropout(0.1)提供差异
    z2 = sent_pass()

    tau = cfg["contrast_tau"]
    sim = z1 @ z2.t() / tau                       # (N, N)
    ar = torch.arange(N, device=device)
    partner = ar ^ 1                              # 偶奇配对 = paraphrase 伙伴
    pos = torch.stack([sim[ar, ar], sim[ar, partner]], dim=1)  # 两个正例
    logits = sim.clone()
    logits[ar, ar] = -1e9                         # 正例从分母移除
    logits[ar, partner] = -1e9
    all_logits = torch.cat([logits, pos], dim=1)
    return -(torch.logsumexp(pos, dim=1)
             - torch.logsumexp(all_logits, dim=1)).mean()


def sep_probe(model, dataset):
    """calibrate 同款查询组, target_encoder(检索侧) 编码。
    返回 (sep, min_hit, max_miss)。"""
    ch2idx = {c: i for i, c in enumerate(dataset.chars)}

    def enc(text):
        ids = [ch2idx[c] for c in text if c in ch2idx][:CONFIG["seq_len"]]
        if not ids:
            return None
        x = torch.tensor([ids], dtype=torch.long)
        pos = model.pos_embed[:, :x.size(1), :]
        with torch.no_grad():
            e = model.target_encoder(model.token_embed(x) + pos).mean(1)
        return F.normalize(e, dim=-1).squeeze(0)

    idx = {}
    for lang, path in (("zh", "identity_zh.txt"), ("en", "identity_en.txt")):
        vecs = []
        for l in open(path, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            v = enc(l)
            if v is not None:
                vecs.append(v)
        idx[lang] = torch.stack(vecs)

    hit_q = [("你是谁", "zh"), ("你叫什么名字", "zh"), ("风小是谁", "zh"),
             ("你的名字是什么", "zh"), ("Who are you", "en"),
             ("What is your name", "en"),
             ("介绍一下你自己", "zh"), ("你叫什么", "zh"),
             ("你的名字是啥", "zh"), ("你是什么模型", "zh"),
             ("who r u", "en"), ("your name", "en")]
    miss_q = [("今天天气如何", "zh"), ("1+1等于几", "zh"),
              ("明天会下雨吗", "zh"), ("怎么做红烧肉", "zh"),
              ("北京是哪个国家的首都", "zh")]

    def maxsim(q, lang):
        v = enc(q)
        return None if v is None else float((idx[lang] @ v).max())

    hits = [s for s in (maxsim(q, l) for q, l in hit_q) if s is not None]
    miss = [s for s in (maxsim(q, l) for q, l in miss_q) if s is not None]
    if not hits or not miss:
        return -float("inf"), 0.0, 0.0
    return min(hits) - max(miss), min(hits), max(miss)


def train(fresh=False):
    set_seed(CONFIG["seed"])

    print("=" * 68)
    print(f"  SP-JEPA v{__version__}  |  Model: FengXiao (风小)")
    print("=" * 68)
    print(f"  torch      : {ENV['torch_version']}")
    print(f"  CUDA build : {ENV['cuda_built'] if ENV['cuda_built'] else 'CPU-only build'}")
    print(f"  Device     : {ENV['device_name']}  [{ENV['device_tag']}]")
    if ENV["device"] == "cuda":
        print(f"  VRAM       : {ENV['vram_gb']:.1f} GB")
    print(f"  RAM        : {ENV['ram_gb']:.1f} GB")
    print(f"  AMP        : {CONFIG['use_amp']}")
    print(f"  Batch Size : {CONFIG['batch_size']}")
    print(f"  Embed Dim  : {CONFIG['embed_dim']}")
    print(f"  Layers     : {CONFIG['num_layers']}")
    print(f"  Seq Len    : {CONFIG['seq_len']}")
    print(f"  Lambda     : multi={CONFIG['lambda_multi']}, conf={CONFIG['lambda_conf']}")
    print("=" * 68)
    print()

    dataset = TextDataset(CONFIG["data_path"], CONFIG["seq_len"], CONFIG["val_ratio"])

    val_steps = len(dataset.val) // CONFIG["batch_size"]
    val_masks = []
    for _ in range(val_steps):
        val_masks.append(make_span_mask_batch(
            CONFIG["batch_size"], CONFIG["seq_len"], CONFIG["mask_ratio"]))

    model = SP_JEPA(
        dataset.vocab_size, dataset.mask_token,
        CONFIG["embed_dim"], CONFIG["num_heads"],
        CONFIG["num_layers"], CONFIG["seq_len"],
    ).to(DEVICE)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"[MODEL] Trainable params: {n_params / 1e6:.2f}M\n")

    optimizer = optim.AdamW(trainable, lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    amp_ctx, amp_scaler = build_amp_handles(CONFIG["use_amp"], ENV["device"])

    explorer = Explorer(
        len(dataset.train),
        CONFIG["explorer_tau"],
        CONFIG["explorer_momentum"],
    )

    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    best_val = float("inf")
    best_sep = -float("inf")
    para_pool = load_para_pool(CONFIG["para_tsv"], dataset.chars, CONFIG["seq_len"])
    ch2idx = {c: i for i, c in enumerate(dataset.chars)}
    print(f"[CONTRAST] paraphrase pairs: {len(para_pool)} "
          f"(lambda={CONFIG['lambda_contrast']}, tau={CONFIG['contrast_tau']}, "
          f"dropout={CONFIG['contrast_dropout']})\n")
    history = {
        "epoch": [], "train": [], "val": [],
        "jepa": [], "visreg": [], "multi": [], "conf": [],
        "decorr": [], "center": [],
        "sent_center": [], "sent_var": [],
        "ema_diff": [], "ema_m": [], "mask_ratio": [],
        "repr_std": [], "repr_eff_rank": [], "repr_dim": [],
        "contrast": [], "sep": [],
    }
    start_time = time.time()

    steps_per_epoch = len(dataset.train) // CONFIG["batch_size"]
    total_steps = CONFIG["epochs"] * steps_per_epoch
    global_step = 0
    start_epoch = 1

    history_path = os.path.join(CONFIG["save_dir"], "history.json")

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    resume_path = os.path.join(CONFIG["save_dir"], "fengxiao_resume.pth")
    if not fresh and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"])
        if "optimizer_state" in ck:
            try:
                optimizer.load_state_dict(ck["optimizer_state"])
            except Exception:
                print("[RESUME] optimizer state mismatch -> re-initialized")
        if ck.get("explorer_difficulty") is not None:
            try:
                explorer.difficulty = ck["explorer_difficulty"].to(
                    explorer.difficulty.device, explorer.difficulty.dtype)
            except Exception:
                print("[RESUME] explorer state mismatch -> re-initialized")
        start_epoch = ck["epoch"] + 1
        global_step = ck.get("global_step", 0)
        best_val = ck.get("best_val", float("inf"))
        best_sep = ck.get("best_sep", -float("inf"))
        history = ck.get("history", history)
        print(f"[RESUME] Continuing from epoch {start_epoch} "
              f"(best_val={best_val:.4f}, global_step={global_step})")

    if start_epoch > CONFIG["epochs"]:
        print(f"[TRAIN] Already trained {start_epoch - 1}/{CONFIG['epochs']} epochs. "
              f"Nothing to do. (use `python sp_jepa.py fresh` to retrain)")
        return

    print("[TRAIN] Starting...")
    print(f"[TRAIN] Explorer: {CONFIG['explorer_enabled']}")
    print("-" * 68)

    mask_lo = CONFIG["mask_ratio"] * CONFIG["mask_ratio_min_factor"]
    mask_hi = CONFIG["mask_ratio"]

    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        model.train()
        all_indices = torch.arange(len(dataset.train))
        steps = steps_per_epoch
        tr_loss, tr_jepa, tr_vis = 0.0, 0.0, 0.0
        tr_contrast = 0.0
        tr_decorr, tr_center = 0.0, 0.0
        tr_sent_center, tr_sent_var = 0.0, 0.0
        tr_multi, tr_conf = 0.0, 0.0
        epoch_mask_ratios = []

        for step in range(steps):
            if CONFIG["explorer_enabled"]:
                idx = explorer.choose(all_indices, CONFIG["batch_size"])
            else:
                idx = all_indices[torch.randperm(len(all_indices))[:CONFIG["batch_size"]]]

            x = dataset.train[idx].to(DEVICE)
            B, L = x.shape

            # v5.1.0: 下界抬到 mask_lo，避免 ratio≈0 的空掩码浪费步
            ratio = random.uniform(mask_lo, mask_hi)
            mask = make_span_mask_batch(B, L, ratio).to(DEVICE)

            epoch_mask_ratios.append(mask.float().mean().item())

            optimizer.zero_grad()
            with amp_ctx:
                loss, l_jepa, l_vis, per_sample_err, l_multi, l_conf = model(x, mask)

            # v5.3.0: 对比项(检索形态), 与 JEPA 主损失梯度累加
            l_contrast = contrast_forward(model, para_pool, ch2idx, CONFIG)

            if amp_scaler:
                amp_scaler.scale(loss).backward()
                if l_contrast is not None:
                    amp_scaler.scale(CONFIG["lambda_contrast"] * l_contrast).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, CONFIG["max_grad_norm"])
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss.backward()
                if l_contrast is not None:
                    (CONFIG["lambda_contrast"] * l_contrast).backward()
                torch.nn.utils.clip_grad_norm_(trainable, CONFIG["max_grad_norm"])
                optimizer.step()

            global_step += 1
            model.update_ema(global_step, total_steps)

            if CONFIG["explorer_enabled"]:
                explorer.update(idx, per_sample_err.cpu())

            tr_loss  += loss.item()
            tr_jepa  += l_jepa.item()
            tr_vis   += l_vis.item()
            tr_multi += l_multi.item()
            tr_conf  += l_conf.item()
            if l_contrast is not None:
                tr_contrast += l_contrast.item()
            # v5.2.0: decorr 分量观测（l_vis 是 VISReg 聚合，
            # last_decorr 是单步未加权分量）
            tr_decorr += model.visreg_loss.last_decorr
            tr_center += model.visreg_loss.last_center
            tr_sent_center += model.visreg_loss.last_sent_center
            tr_sent_var += model.visreg_loss.last_sent_var

        avg_tr = tr_loss / max(steps, 1)
        val_loss = evaluate(model, dataset, val_masks)
        elapsed = time.time() - start_time
        ema_diff = model.ema_health()
        avg_mask_ratio = sum(epoch_mask_ratios) / max(len(epoch_mask_ratios), 1)

        # v5.1.0: 表征健康度
        if CONFIG["repr_health_enabled"]:
            repr_std, repr_eff, repr_dim = representation_health(model, dataset)
        else:
            repr_std, repr_eff, repr_dim = 0.0, 0.0, CONFIG["embed_dim"]

        print(f"[Epoch {epoch:02d}] "
              f"Train: {avg_tr:.4f} "
              f"(JEPA: {tr_jepa/steps:.4f}, "
              f"Multi: {tr_multi/steps:.4f}, "
              f"Conf: {tr_conf/steps:.4f}, "
              f"VISReg: {tr_vis/steps:.4f}, "
              f"Decorr: {tr_decorr/steps:.4f}, "
              f"Center: {tr_center/steps:.4f}, "
              f"SentC: {tr_sent_center/steps:.4f}, "
              f"SentV: {tr_sent_var/steps:.4f}, Ctr: {tr_contrast/steps:.4f}) | "
              f"Val: {val_loss:.4f} | "
              f"EMA_m: {model.current_momentum:.4f} | "
              f"EMA_diff: {ema_diff:.4e} | "
              f"MaskR: {avg_mask_ratio:.3f} | "
              f"RepStd: {repr_std:.3f} | "
              f"EffRank: {repr_eff:.1f}/{repr_dim} | "
              f"Time: {elapsed:.1f}s")

        history["epoch"].append(epoch)
        history["train"].append(avg_tr)
        history["val"].append(val_loss)
        history["jepa"].append(tr_jepa / max(steps, 1))
        history["visreg"].append(tr_vis / max(steps, 1))
        history["decorr"].append(tr_decorr / max(steps, 1))
        history["center"].append(tr_center / max(steps, 1))
        history["sent_center"].append(tr_sent_center / max(steps, 1))
        history["sent_var"].append(tr_sent_var / max(steps, 1))
        history["multi"].append(tr_multi / max(steps, 1))
        history["conf"].append(tr_conf / max(steps, 1))
        history["ema_diff"].append(ema_diff)
        history["ema_m"].append(model.current_momentum)
        history["mask_ratio"].append(avg_mask_ratio)
        history["repr_std"].append(repr_std)
        history["repr_eff_rank"].append(repr_eff)
        history["repr_dim"].append(repr_dim)

        # v5.3.0: 对比项均值 + sep probe(检索的裁判, 按 sep 存 best)
        history["contrast"].append(tr_contrast / max(steps, 1))
        sep_val = None
        if epoch % CONFIG["sep_probe_every"] == 0 or epoch == CONFIG["epochs"]:
            sep_val, s_min_hit, s_max_miss = sep_probe(model, dataset)
            print(f"  [SEP] sep={sep_val:+.4f}  "
                  f"min(HIT)={s_min_hit:.4f}  max(MISS)={s_max_miss:.4f}")
            if sep_val > best_sep:
                best_sep = sep_val
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "sep": sep_val,
                    "vocab": dataset.chars,
                    "config": CONFIG,
                    "hardware": ENV,
                    "version": __version__,
                }, os.path.join(CONFIG["save_dir"], "fengxiao_best_sep.pth"))
                print(f"  -> BestSep saved (sep={best_sep:+.4f})")
        history["sep"].append(sep_val)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": best_val,
                "vocab": dataset.chars,
                "config": CONFIG,
                "hardware": ENV,
                "version": __version__,
            }, os.path.join(CONFIG["save_dir"], "fengxiao_best.pth"))
            print(f"  -> Best saved (val_loss={best_val:.4f})")

        if epoch % CONFIG["save_interval"] == 0 or epoch == CONFIG["epochs"]:
            ckpt_path = os.path.join(CONFIG["save_dir"], f"fengxiao_epoch{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "vocab": dataset.chars,
                "config": CONFIG,
                "hardware": ENV,
                "version": __version__,
            }, ckpt_path)
            print(f"  -> Checkpoint saved: {ckpt_path}")

        # -- per-epoch resume checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "explorer_difficulty": explorer.difficulty.cpu().clone(),
            "global_step": global_step,
            "best_val": best_val,
            "best_sep": best_sep,
            "history": history,
        }, resume_path)

        # v5.1.0: history 每 epoch 落盘（crash-safe）
        try:
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"  [WARN] history.json 写入失败: {e}")

    print("-" * 68)
    print(f"[DONE] Best val loss: {best_val:.4f}")
    print(f"[DONE] Checkpoints in: {os.path.abspath(CONFIG['save_dir'])}")
    print(f"[DONE] History: {history_path}")
    print("-" * 68)


if __name__ == "__main__":
    import sys
    fresh = len(sys.argv) > 1 and sys.argv[1] == "fresh"
    try:
        train(fresh=fresh)
    except KeyboardInterrupt:
        print("\n[STOPPED] Progress saved in fengxiao_resume.pth (rerun to continue)")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        try:
            input("\nPress Enter to exit...")
        except EOFError:
            pass
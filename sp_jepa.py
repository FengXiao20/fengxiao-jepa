# -*- coding: utf-8 -*-
"""
SP-JEPA: Self-Play Joint Embedding Predictive Architecture
======================================================================
Model Name: FengXiao (风小)
A single-file, hardware-adaptive, cross-platform training script.

Revision History:
- v1: Initial draft. Fake per-sample error (batch-average copy).
- v2: Real per-sample error framework. Fixed val masks. Modern AMP API.
- v3: Mask-only error. Locked val data + masks. Zero-loss with grad.
- v4: Global shuffle before split (identity corpus lands in TRAIN).
- v4.1: Docstring cleanup.
- v4.2: Filename stabilized to `sp_jepa.py`.
- v4.3: Restored per-step EMA (P0). Added EMA health check log.
        Kept scheduled masking (uniform in [0, mask_ratio]).
        Kept non-rejection make_span_mask.
- v4.4: Fixed batch mask shape bug (per-sample -> shared mask,
        unsqueeze+expand for indexing). External review credit.
        Added torch environment detection (CPU-only vs CUDA build).
        Added autocast/GradScaler version compatibility shim.

Status: EMA alive, scheduled masking live, val locked, env-aware.

Usage:
1. Place `input_augmented.txt` (UTF-8) in the same directory.
2. python sp_jepa.py
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


# =====================================================================
#  0. Environment & Hardware Detection
# =====================================================================
def detect_env():
    """
    检测 torch 安装类型 + 可用设备。
    三种组合:
      A) CPU-only 编译版 torch    -> 强制 CPU
      B) CUDA 编译版 + 无 GPU     -> 强制 CPU，并警告
      C) CUDA 编译版 + 有 GPU     -> 用 GPU + AMP
    """
    info = {
        "torch_version": torch.__version__,
        "cuda_built":     torch.version.cuda,          # None 表示 CPU-only 编译版
        "cuda_available": torch.cuda.is_available(),   # 是否有可用 GPU
        "device": "cpu", "device_name": "CPU",
        "vram_gb": 0, "ram_gb": 8,
        "use_amp": False, "num_workers": 0,
        "batch_size": 8, "embed_dim": 192, "num_layers": 3, "seq_len": 96,
        "device_tag": "CPU",
    }

    try:
        import psutil
        info["ram_gb"] = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

    # --- GPU 可用 ---
    if info["cuda_built"] is not None and info["cuda_available"]:
        info["device"] = "cuda"
        info["device_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info["use_amp"] = True
        info["device_tag"] = "GPU"

        if info["vram_gb"] >= 20:
            info.update({"batch_size": 32, "embed_dim": 384, "num_layers": 6, "seq_len": 256, "num_workers": 4})
        elif info["vram_gb"] >= 10:
            info.update({"batch_size": 24, "embed_dim": 384, "num_layers": 4, "seq_len": 192, "num_workers": 4})
        elif info["vram_gb"] >= 5:
            info.update({"batch_size": 16, "embed_dim": 256, "num_layers": 4, "seq_len": 128, "num_workers": 2})
        else:
            info.update({"batch_size": 8, "embed_dim": 192, "num_layers": 3, "seq_len": 96})

    # --- CUDA 编译版但没 GPU ---
    elif info["cuda_built"] is not None and not info["cuda_available"]:
        info["device_tag"] = "CPU (CUDA build, no GPU)"
        print("[ENV] 检测到 CUDA 编译版 torch，但当前机器没有可用的 GPU。")
        print("      可能原因: 驱动缺失 / 显卡不支持 / 被环境变量屏蔽。")
        print("      将回退到 CPU 训练。\n")
        if info["ram_gb"] >= 16:
            info.update({"batch_size": 8, "embed_dim": 192, "num_layers": 3, "seq_len": 96})
        else:
            info.update({"batch_size": 4, "embed_dim": 128, "num_layers": 2, "seq_len": 64})

    # --- CPU-only 编译版 ---
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
HARDWARE = ENV          # 向后兼容旧名字
DEVICE = torch.device(ENV["device"])


CONFIG = {
    "data_path": "input_augmented.txt",
    "save_dir": "./checkpoints_spjepa",
    "seq_len": ENV["seq_len"],
    "embed_dim": ENV["embed_dim"],
    "num_heads": max(4, ENV["embed_dim"] // 64),
    "num_layers": ENV["num_layers"],
    "batch_size": ENV["batch_size"],
    "epochs": 30,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "max_grad_norm": 1.0,
    "mask_ratio": 0.3,
    "val_ratio": 0.05,
    "seed": 42,
    "ema_start": 0.990,
    "ema_end": 0.996,
    "warmup_epochs": 5,
    "lambda_scale": 1.0,
    "lambda_shape": 1.0,
    "num_projections": 128,
    "explorer_tau": 0.5,
    "explorer_momentum": 0.7,
    "explorer_enabled": True,
    "use_amp": ENV["use_amp"],
    "log_interval": 10,
    "save_interval": 10,
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
    """随机 span 掩码（目标位置 + 后向扩展，无拒绝采样）。"""
    mask = torch.zeros(seq_len, dtype=torch.bool)
    num_mask = int(seq_len * mask_ratio)
    if num_mask <= 0:
        return mask
    targets = set(torch.randperm(seq_len)[:num_mask].tolist())
    i = 0
    while i < seq_len:
        if i in targets:
            length = random.randint(1, 4)
            end = min(seq_len, i + length)
            mask[i:end] = True
            i = end
        else:
            i += 1
    return mask


# =====================================================================
#  2. VISReg Loss
# =====================================================================
def variance_loss(z, eps=1e-4):
    std_z = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std_z))


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
    def __init__(self, lambda_scale=1.0, lambda_shape=1.0, num_projections=128):
        super().__init__()
        self.lambda_scale = lambda_scale
        self.lambda_shape = lambda_shape
        self.num_projections = num_projections

    def forward(self, z):
        loss_scale = variance_loss(z)
        z_norm = (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-6)
        loss_shape = sliced_wasserstein_distance(z_norm, self.num_projections)
        return self.lambda_scale * loss_scale + self.lambda_shape * loss_shape


# =====================================================================
#  3. Explorer (Error-Driven Self-Play Sampler)
# =====================================================================
class Explorer:
    def __init__(self, num_seqs, tau=0.5, momentum=0.7):
        self.difficulty = torch.zeros(num_seqs)
        self.tau = tau
        self.momentum = momentum

    def choose(self, pool, k):
        pool_diff = self.difficulty[pool]
        weights = torch.softmax(pool_diff / self.tau, dim=0)
        k = min(k, len(pool))
        return pool[torch.multinomial(weights, k, replacement=False)]

    def update(self, indices, per_sample_errors, rewards=None):
        # Optional reward: 高奖励 -> 难度下降（已学会）
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

        def make_enc():
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 4, dropout=0.0,
                batch_first=True, norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=num_layers)

        self.context_encoder = make_enc()
        self.target_encoder = make_enc()
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.visreg_loss = VISRegLoss()
        self.current_momentum = CONFIG["ema_start"]

    def forward(self, x, mask):
        B, L = x.shape
        pos = self.pos_embed[:, :L, :]
        ctx_x = x.masked_fill(mask, self.mask_token)

        ctx_emb = self.context_encoder(self.token_embed(ctx_x) + pos)
        with torch.no_grad():
            tgt_emb = self.target_encoder(self.token_embed(x) + pos)

        pred_emb = self.predictor(ctx_emb)

        mask_b = mask.unsqueeze(0).expand(B, -1)  # (B, L)
        pred_m, tgt_m = pred_emb[mask_b], tgt_emb[mask_b]

        if pred_m.size(0) == 0:
            zero = pred_emb.sum() * 0.0
            return zero, zero.detach(), zero.detach(), torch.zeros(B, device=x.device)

        loss_jepa = F.smooth_l1_loss(
            F.normalize(pred_m, dim=-1),
            F.normalize(tgt_m, dim=-1),
        )
        loss_visreg = self.visreg_loss(pred_m)
        total = loss_jepa + loss_visreg

        with torch.no_grad():
            diff = F.smooth_l1_loss(
                F.normalize(pred_emb, dim=-1),
                F.normalize(tgt_emb, dim=-1),
                reduction="none",
            ).mean(dim=-1).detach()
            per_sample_err = diff[mask_b].view(B, -1).mean(dim=1)

        return total, loss_jepa.detach(), loss_visreg.detach(), per_sample_err

    @torch.no_grad()
    def update_ema(self, epoch):
        t = min(epoch / CONFIG["warmup_epochs"], 1.0)
        m = CONFIG["ema_start"] + (CONFIG["ema_end"] - CONFIG["ema_start"]) * t
        self.current_momentum = m
        for p_t, p_c in zip(self.target_encoder.parameters(),
                            self.context_encoder.parameters()):
            p_t.mul_(m).add_(p_c, alpha=1 - m)

    @torch.no_grad()
    def ema_health(self):
        """返回 target 与 context 首层参数的 L1 距离。EMA 活着 → 很小。"""
        ctx_p = next(self.context_encoder.parameters())
        tgt_p = next(self.target_encoder.parameters())
        return (tgt_p - ctx_p).abs().mean().item()


# =====================================================================
#  5. AMP Compatibility Shim
# =====================================================================
def build_amp_handles(use_amp, device_type):
    """
    构造 autocast context 和 GradScaler。
    兼容:
      - 新版 torch.amp.autocast("cuda", ...) / torch.amp.GradScaler("cuda", ...)
      - 老版 torch.cuda.amp.autocast(...) / torch.cuda.amp.GradScaler(...)
      - CPU 模式用 contextlib.nullcontext()，不做任何事
    """
    if not use_amp:
        return contextlib.nullcontext(), None

    # GradScaler
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    # autocast
    try:
        ctx = torch.amp.autocast("cuda", enabled=True)
    except (TypeError, AttributeError):
        ctx = torch.cuda.amp.autocast(enabled=True)

    return ctx, scaler


# =====================================================================
#  6. Evaluation & Training
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
            loss, _, _, _ = model(x, mask)
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


def train():
    set_seed(CONFIG["seed"])

    print("=" * 68)
    print("  SP-JEPA  |  Model: FengXiao (风小)")
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
    print("=" * 68)
    print()

    dataset = TextDataset(CONFIG["data_path"], CONFIG["seq_len"], CONFIG["val_ratio"])

    val_steps = len(dataset.val) // CONFIG["batch_size"]
    val_masks = []
    for _ in range(val_steps):
        val_masks.append(make_span_mask(CONFIG["seq_len"], CONFIG["mask_ratio"]))

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
    history = {"epoch": [], "train": [], "val": [], "ema_diff": []}
    start_time = time.time()

    print("[TRAIN] Starting...")
    print(f"[TRAIN] Explorer: {CONFIG['explorer_enabled']}")
    print("-" * 68)

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        all_indices = torch.arange(len(dataset.train))
        steps = len(dataset.train) // CONFIG["batch_size"]
        tr_loss, tr_jepa, tr_vis = 0.0, 0.0, 0.0

        for step in range(steps):
            if CONFIG["explorer_enabled"]:
                idx = explorer.choose(all_indices, CONFIG["batch_size"])
            else:
                idx = all_indices[torch.randperm(len(all_indices))[:CONFIG["batch_size"]]]

            x = dataset.train[idx].to(DEVICE)
            B, L = x.shape

            # Scheduled masking: 0% ~ mask_ratio 全谱
            ratio = random.uniform(0.0, CONFIG["mask_ratio"])
            mask = make_span_mask(L, ratio).to(DEVICE)

            optimizer.zero_grad()
            with amp_ctx:
                loss, l_jepa, l_vis, per_sample_err = model(x, mask)

            if amp_scaler:
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, CONFIG["max_grad_norm"])
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, CONFIG["max_grad_norm"])
                optimizer.step()

            # EMA: 每 step 更新，标准 JEPA 语义
            model.update_ema(epoch)

            if CONFIG["explorer_enabled"]:
                explorer.update(idx, per_sample_err.cpu())

            tr_loss += loss.item()
            tr_jepa += l_jepa.item() if torch.is_tensor(l_jepa) else l_jepa
            tr_vis += l_vis.item() if torch.is_tensor(l_vis) else l_vis

        avg_tr = tr_loss / max(steps, 1)
        val_loss = evaluate(model, dataset, val_masks)
        elapsed = time.time() - start_time
        ema_diff = model.ema_health()

        print(f"[Epoch {epoch:02d}] "
              f"Train: {avg_tr:.4f} (JEPA: {tr_jepa/steps:.4f}, VISReg: {tr_vis/steps:.4f}) | "
              f"Val: {val_loss:.4f} | "
              f"EMA: {model.current_momentum:.4f} | "
              f"EMA_diff: {ema_diff:.4e} | "
              f"Time: {elapsed:.1f}s")

        history["epoch"].append(epoch)
        history["train"].append(avg_tr)
        history["val"].append(val_loss)
        history["ema_diff"].append(ema_diff)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": best_val,
                "vocab": dataset.chars,
                "config": CONFIG,
                "hardware": ENV,
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
            }, ckpt_path)
            print(f"  -> Checkpoint saved: {ckpt_path}")

    history_path = os.path.join(CONFIG["save_dir"], "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("-" * 68)
    print(f"[DONE] Best val loss: {best_val:.4f}")
    print(f"[DONE] Checkpoints in: {os.path.abspath(CONFIG['save_dir'])}")
    print(f"[DONE] History: {history_path}")
    print("-" * 68)


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n[STOPPED] User interrupted.")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        try:
            input("\nPress Enter to exit...")
        except EOFError:
            pass
# -*- coding: utf-8 -*-
"""fengxiao-intuition · 阶段2：预训练 + 唤醒两个头 —— 模型定义

基座：FengYu v0.2.1 同架构（RMSNorm + RoPE + SwiGLU + weight tying），
参数爬升 dim 256 / heads 4 / layers 3 → ~2.6M（TA 已批），seq_len 128，vocab 512。

两个新头（本阶段的主角）:
  fok_head   读题面最后一个 token（"ans=" 的 '='）的 hidden state，
             预测"这道题最终会答对吗"——作答前的直觉。
  conf_head  读答案末尾 token 的 hidden state，预测"这个答案对不对"——作答后的自信。
             （对应 sp_jepa 的 confidence_head，λ_conf 从 0 启用到真值）

旧 JEPA/sep 探针：保留为**只读观测仪表**（sep_probe），训练权重为零——
只算读数，不进 loss（"接近零"的机制化实现）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- 基座组件（继承自 fengyu.py v0.2.1，未改机制） ----------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def precompute_rope(seq_len, head_dim, device):
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[:x.size(2)].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.size(2)].unsqueeze(0).unsqueeze(0)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, ffn_dim)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


def compute_ffn_dim(embed_dim):
    """SwiGLU 标准：dim * 8/3，对齐到 64 的倍数（与 fengyu.py 同式）。"""
    return int((embed_dim * 8 / 3 + 63) // 64 * 64)


def _head(dim, hidden):
    return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))


# ---------------- 主模型 ----------------

class IntuitionLM(nn.Module):
    """基座 + FOK/confidence 双头。基座权重与 fengyu.py 同名同义，机制零改动。"""

    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=3,
                 max_len=128, head_hidden=64, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, compute_ffn_dim(embed_dim)) for _ in range(num_layers)])
        self.norm_f = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.fok_head = _head(embed_dim, head_hidden)
        self.conf_head = _head(embed_dim, head_hidden)

        self._init_weights()
        self.lm_head.weight = self.tok_embed.weight   # weight tying AFTER init
        self._rope_cache = {}

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _get_rope(self, T, device):
        key = (T, str(device))
        if key not in self._rope_cache:
            self._rope_cache[key] = precompute_rope(T, self.head_dim, device)
        return self._rope_cache[key]

    def features(self, x):
        """返回最后一层 hidden state (B, T, C)，不接 lm_head。"""
        cos, sin = self._get_rope(x.size(1), x.device)
        h = self.tok_embed(x)
        for block in self.blocks:
            h = block(h, cos, sin)
        return self.norm_f(h)

    def forward(self, x, targets=None):
        h = self.features(x)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss, h

    @staticmethod
    def fok_logit(h, positions):
        """positions: (B,) LongTensor —— 每条序列题面末 token 的列号。"""
        rows = torch.arange(h.size(0), device=h.device)
        return h[rows, positions]  # (B, C)，logit 由调用方接 fok_head

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def head_param_stds(self):
        return {n: float(p.detach().std()) for n, p in self.named_parameters()
                if n.startswith(("fok_head", "conf_head")) and p.ndim > 1}


# ---------------- sep 观测探针（只读，零训练权重） ----------------

@torch.no_grad()
def sep_probe(model, hit_ids, miss_ids, qpos_hit, qpos_miss, device):
    """检索式 sep 读数：HIT = 模97 真题面（q_end hidden），MISS = 域外毒题面。
    sep = mean_intra(HIT cos) - mean_cross(HIT, MISS cos)。
    上一期教训：探针用独立 RNG / 只读，绝不污染训练随机流。"""
    model.eval()

    def _embed(ids_list, qpos):
        T = max(len(s) for s in ids_list)
        x = torch.zeros(len(ids_list), T, dtype=torch.long)
        for i, s in enumerate(ids_list):
            x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        h = model.features(x.to(device))
        rows = torch.arange(len(ids_list))
        return F.normalize(h[rows, torch.tensor(qpos)], dim=-1)

    zh = _embed(hit_ids, qpos_hit)
    zm = _embed(miss_ids, qpos_miss)
    intra = (zh @ zh.T)
    n = zh.size(0)
    mean_intra = (intra.sum() - intra.diagonal().sum()) / max(1, n * (n - 1))
    cross = (zh @ zm.T).mean()
    model.train()
    return {"sep": float(mean_intra - cross),
            "hit_cos": float(mean_intra), "miss_cos": float(cross)}

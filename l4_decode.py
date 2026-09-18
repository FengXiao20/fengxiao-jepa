# -*- coding: utf-8 -*-
"""
L4: Retrieval-Based Decoding
=============================
Predictor 导航 + span 检索 + 循环拼接。

Usage:
    from l4_decode import RetrievalDecoder
    decoder = RetrievalDecoder(checkpoint_path, corpus_path)
    result = decoder.generate("小狐狸走进了", max_steps=10)
    print(result)

Optimizations:
- Batch encoding with masked mean pooling (padding 不污染向量)
- Fixed seed + disk cache with mtime invalidation
"""

import os
import random
import hashlib
import torch
import torch.nn.functional as F

from sp_jepa import SP_JEPA


SPAN_SEED = 42
CACHE_DIR = "./l4_cache"


class RetrievalDecoder:
    def __init__(self, checkpoint_path, corpus_path,
                 span_min=1, span_max=4,
                 error_threshold=0.6,
                 seq_len=None,
                 max_spans=20000,
                 use_cache=True):

        # --- load model ---
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.vocab = ckpt["vocab"]
        self.config = ckpt["config"]
        self.char2idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.mask_token = len(self.vocab)
        self.vocab_size = len(self.vocab) + 1
        self.seq_len = seq_len or self.config["seq_len"]
        self.error_threshold = error_threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SP_JEPA(
            vocab_size=self.vocab_size,
            mask_token=self.mask_token,
            embed_dim=self.config["embed_dim"],
            num_heads=self.config["num_heads"],
            num_layers=self.config["num_layers"],
            seq_len=self.seq_len,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        # --- build or load span library ---
        cache_key = self._make_cache_key(corpus_path, span_min, span_max, max_spans)
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pt")
        os.makedirs(CACHE_DIR, exist_ok=True)

        if use_cache and os.path.exists(cache_path):
            print(f"[L4] Loading cached span library: {cache_path}")
            cached = torch.load(cache_path, map_location=self.device, weights_only=False)
            self.spans = cached["spans"]
            self.span_vectors = cached["vectors"].to(self.device)
            print(f"[L4] Cache hit: {len(self.spans)} spans")
        else:
            self.spans = self._sample_spans(corpus_path, span_min, span_max, max_spans)
            print(f"[L4] Span library: {len(self.spans)} spans (sampled)")
            self.span_vectors = self._build_span_index_batched()
            if use_cache:
                torch.save({
                    "spans": self.spans,
                    "vectors": self.span_vectors.cpu(),
                }, cache_path)
                print(f"[L4] Cached to: {cache_path}")

    @staticmethod
    def _make_cache_key(corpus_path, span_min, span_max, max_spans):
        """文件路径 + 大小 + mtime + 参数做哈希，变化就自动失效"""
        try:
            st = os.stat(corpus_path)
            size, mtime = st.st_size, int(st.st_mtime)
        except OSError:
            size, mtime = 0, 0
        raw = f"{corpus_path}|{size}|{mtime}|{span_min}|{span_max}|{max_spans}|{SPAN_SEED}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _sample_spans(corpus_path, span_min, span_max, max_spans):
        with open(corpus_path, encoding="utf-8") as f:
            text = f.read()

        rng = random.Random(SPAN_SEED)
        spans = []
        step = max(1, len(text) // max_spans)
        for i in range(0, len(text) - span_max, step):
            L = rng.randint(span_min, span_max)
            spans.append(text[i:i + L])

        seen = set()
        spans = [s for s in spans if not (s in seen or seen.add(s))]
        return spans

    @torch.no_grad()
    def _encode_batch(self, texts):
        """
        Batch encode a list of strings.
        Padding 位置由 masked mean pooling 屏蔽，不污染均值。
        OOV 字符统一跳过（与 _encode_single 一致）。
        """
        if not texts:
            return None

        # 统一 OOV: 跳过未知字符
        id_lists = []
        for t in texts:
            ids = [self.char2idx[ch] for ch in t if ch in self.char2idx]
            id_lists.append(ids)

        # 全部为空（极端情况）→ 返回 None
        if not any(id_lists):
            return None

        max_len = max(len(ids) for ids in id_lists)
        ids_batch = []
        pad_mask = []
        for ids in id_lists:
            n = len(ids)
            padded = ids + [0] * (max_len - n)
            ids_batch.append(padded)
            pad_mask.append([1] * n + [0] * (max_len - n))

        x = torch.tensor(ids_batch, dtype=torch.long, device=self.device)
        m = torch.tensor(pad_mask, dtype=torch.float, device=self.device).unsqueeze(-1)
        B, L = x.shape
        pos = self.model.pos_embed[:, :L, :]
        emb = self.model.target_encoder(self.model.token_embed(x) + pos)
        # masked mean pooling
        emb = (emb * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return F.normalize(emb, dim=-1)

    def _build_span_index_batched(self, batch_size=512):
        vecs = []
        total = len(self.spans)
        for i in range(0, total, batch_size):
            chunk = self.spans[i:i + batch_size]
            v = self._encode_batch(chunk)
            if v is not None:
                vecs.append(v)
            if (i // batch_size) % 5 == 0:
                print(f"  [L4] encoded {min(i + batch_size, total)}/{total}")
        return torch.cat(vecs, dim=0)

    @torch.no_grad()
    def _encode_single(self, text):
        ids = [self.char2idx[ch] for ch in text if ch in self.char2idx]
        if not ids:
            return None
        ids = ids[:self.seq_len]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        L = x.size(1)
        pos = self.model.pos_embed[:, :L, :]
        emb = self.model.target_encoder(self.model.token_embed(x) + pos)
        emb = emb.mean(dim=1)
        return F.normalize(emb, dim=-1).squeeze(0)

    @torch.no_grad()
    def _predict_next(self, context):
        ids = [self.char2idx[ch] for ch in context if ch in self.char2idx]
        if not ids:
            return None
        ids = ids[:self.seq_len]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        L = x.size(1)
        pos = self.model.pos_embed[:, :L, :]
        emb = self.model.context_encoder(self.model.token_embed(x) + pos)
        pred = self.model.predictor(emb)
        return F.normalize(pred[:, -1, :], dim=-1).squeeze(0)

    def generate(self, prompt, max_steps=10, verbose=True):
        context = prompt
        if verbose:
            print(f"[L4] Prompt: {prompt}")

        for step in range(max_steps):
            pred_vec = self._predict_next(context)
            if pred_vec is None:
                break

            sims = torch.matmul(self.span_vectors, pred_vec)
            top_score, top_idx = torch.topk(sims, 1)
            score = top_score.item()
            span = self.spans[top_idx.item()]

            error = 1.0 - score
            if verbose:
                print(f"  step {step+1}: '{span}' "
                      f"(sim={score:.4f}, err={error:.4f})")
            if error > self.error_threshold:
                if verbose:
                    print(f"  [STOP] err {error:.4f} > {self.error_threshold}")
                break

            context += span

        return context
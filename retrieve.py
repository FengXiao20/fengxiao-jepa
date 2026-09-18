# -*- coding: utf-8 -*-
"""
FengXiao Retrieval (V1.5)
==========================
Bilingual retrieval with language-aware routing + L2 multi-hit + greedy dedup.

Revision history:
- V1.3b : CORPUS_PATH -> input_augmented.txt
- V1.4  : Language-aware routing (ZH/EN separate indexes)
          + L2 multi-hit listing (all above threshold)
- V1.5  : Greedy near-duplicate dedup (sim > 0.95) now actually implemented.
          Fixed weird indentation in answer().

Design:
- Language detection: char-class ratio on query
- Per-language index built from identity_zh.txt / identity_en.txt
- Fallback: pure text constant, not in index
"""

import os
import re
import sys
import random
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sp_jepa import SP_JEPA


# =====================================================================
#  Config
# =====================================================================
CHECKPOINT_PATH = "./checkpoints_spjepa/fengxiao_best.pth"
IDENTITY_ZH = "./identity_zh.txt"
IDENTITY_EN = "./identity_en.txt"
CORPUS_PATH = "./input_augmented.txt"
THRESHOLD = 0.6
TOP_K = 5
DEDUP_SIM = 0.95
FALLBACK_TEXT = "我还没有学会这个，我的知识范围很有限。"
MAX_LINE_LEN = 64

# =====================================================================
#  规则路由模式（不走向量，直接命中）
# =====================================================================
IDENTITY_PATTERNS_ZH = [
    "你是谁", "你叫什么", "你的名字", "你是什么模型", "你是啥",
    "介绍一下你自己", "介绍下你自己", "介绍你自己",
    "你的中文名", "你的英文名",
    "你能做什么", "你会什么",
]

IDENTITY_PATTERNS_EN = [
    "who are you", "what is your name",
    "introduce yourself", "what are you",
    "who r u", "what r u",
]

MATH_RE = re.compile(r"\d+\s*[\+\-\*\/×÷=]\s*\d+")

# =====================================================================
#  Language detection
# =====================================================================
CJK_RE = re.compile(r"[\u4e00-\u9fa5]")
LATIN_RE = re.compile(r"[a-zA-Z]")


def detect_lang(text):
    cjk = len(CJK_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    total = cjk + latin
    if total == 0:
        return "mix"
    zh_ratio = cjk / total
    if zh_ratio >= 0.7:
        return "zh"
    if zh_ratio <= 0.3:
        return "en"
    return "mix"


# =====================================================================
#  Coverage check
# =====================================================================
def check_identity_coverage(paths, corpus_path=CORPUS_PATH):
    if not os.path.exists(corpus_path):
        print(f"[WARN] Corpus not found: {corpus_path}")
        return False

    with open(corpus_path, encoding="utf-8", errors="ignore") as f:
        corpus = f.read()

    missing = []
    total = 0
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                total += 1
                if line not in corpus:
                    missing.append((path, i, line))

    if missing:
        print(f"[WARN] {len(missing)} / {total} identity lines NOT in corpus:")
        for path, i, line in missing[:5]:
            print(f"    {os.path.basename(path)}:{i}: {line[:50]}")
        return False
    print(f"[OK] All {total} identity lines found in {corpus_path}.")
    return True


# =====================================================================
#  Retriever
# =====================================================================
class FengXiaoRetriever:
    def __init__(self, checkpoint_path, identity_paths, threshold=THRESHOLD):
        self.threshold = threshold
        self.fallback_text = FALLBACK_TEXT

        print(f"[LOAD] Checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.vocab = ckpt["vocab"]
        self.config = ckpt["config"]
        self.char2idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx2char = {i: ch for i, ch in enumerate(self.vocab)}
        self.mask_token = len(self.vocab)
        self.vocab_size = len(self.vocab) + 1
        self.seq_len = self.config["seq_len"]

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

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"[LOAD] Model: {n_params:.2f}M | Device: {self.device}")

        # per-language indexes: lang -> {"texts": [...], "vectors": [N, D]}
        self.indexes = {}
        for lang, path in identity_paths.items():
            if not os.path.exists(path):
                print(f"[WARN] {path} not found, skipping {lang}.")
                continue
            texts = []
            with open(path, encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    if len(line) > MAX_LINE_LEN:
                        print(f"  [WARN] {lang} line {lineno} too long, "
                              f"truncated at encode.")
                    texts.append(line)
            if not texts:
                continue
            vectors = self._build_index(texts, lang)
            self.indexes[lang] = {"texts": texts, "vectors": vectors}
            print(f"[LOAD] {lang} index: {len(texts)} entries")

        if not self.indexes:
            raise RuntimeError("No identity index built. Check file paths.")

    # -----------------------------------------------------------------
    @torch.no_grad()
    def _encode(self, text, warn_prefix=""):
        oov = [ch for ch in text if ch not in self.char2idx]
        if oov:
            print(f"  [WARN] {warn_prefix} OOV: {set(oov)}")
        ids = [self.char2idx[ch] for ch in text if ch in self.char2idx]
        if not ids:
            return None
        ids = ids[:self.seq_len]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        L = x.size(1)
        pos = self.model.pos_embed[:, :L, :]
        emb = self.model.target_encoder(self.model.token_embed(x) + pos)
        emb = emb.mean(dim=1)
        emb = F.normalize(emb, dim=-1)
        return emb.squeeze(0)

    def _build_index(self, texts, lang):
        vecs = []
        for i, t in enumerate(texts):
            v = self._encode(t, warn_prefix=f"[{lang} #{i}]")
            if v is None:
                v = torch.zeros(self.config["embed_dim"], device=self.device)
            vecs.append(v)
        return torch.stack(vecs, dim=0)

    # -----------------------------------------------------------------
    def _query_index(self, lang, query_vec, top_k):
        """Returns list of (text, score, vec) sorted by score desc."""
        idx = self.indexes[lang]
        sims = torch.matmul(idx["vectors"], query_vec)
        k = min(top_k, len(sims))
        scores, indices = torch.topk(sims, k)
        out = []
        for s, i in zip(scores.tolist(), indices.tolist()):
            out.append((idx["texts"][i], s, idx["vectors"][i]))
        return out

    def query(self, text, top_k=TOP_K):
        """Returns list of (text, score, vec) sorted by score desc."""
        q = self._encode(text, warn_prefix="[query]")
        if q is None:
            return []
        lang = detect_lang(text)

        if lang == "mix" or lang not in self.indexes:
            results = []
            for lg in self.indexes:
                results.extend(self._query_index(lg, q, top_k))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        return self._query_index(lang, q, top_k)

    # -----------------------------------------------------------------
    @staticmethod
    def _dedup(hits, sim_threshold=DEDUP_SIM):
        """
        Greedy near-duplicate removal.
        hits: list of (text, score, vec), already sorted by score desc.
        """
        selected = []
        sel_vecs = []
        for t, s, v in hits:
            # 若与任一已选条目 cosine > threshold，视为重复，跳过
            dup = False
            for u in sel_vecs:
                if float(torch.dot(v, u)) > sim_threshold:
                    dup = True
                    break
            if not dup:
                selected.append((t, s))
                sel_vecs.append(v)
        return selected

    def answer(self, text, verbose=False):
        # --- 规则路由层 ---
        lower = text.lower().strip()
        lang = detect_lang(text)

        # 身份类查询 → 从身份语料随机抽一句
        if any(p in lower for p in IDENTITY_PATTERNS_ZH + IDENTITY_PATTERNS_EN):
            target_lang = lang
            if target_lang not in self.indexes:
                target_lang = "zh" if any(
                    p in text for p in IDENTITY_PATTERNS_ZH
                ) else "en"
            if target_lang in self.indexes:
                texts = self.indexes[target_lang]["texts"]
                if texts:
                    if verbose:
                        print(f"  [ROUTE] identity query -> {target_lang} index")
                    return random.choice(texts), 1.0, False
            return self.fallback_text, 0.0, True

        # 数学类查询 → 直接 fallback
        if MATH_RE.search(text):
            if verbose:
                print(f"  [ROUTE] math query -> fallback")
            return self.fallback_text, 1.0, True

        # --- 走向量检索（原逻辑）---
        results = self.query(text, top_k=TOP_K)
        if not results:
            return self.fallback_text, 0.0, True

        if verbose:
            print(f"  [DEBUG] Query: {text!r} | lang={lang}")
            for t, s, _ in results:
                print(f"    {s:.4f}  {t[:60]}")

        hits = [r for r in results if r[1] >= self.threshold]
        if not hits:
            return self.fallback_text, results[0][1], True

        hits = self._dedup(hits)

        if len(hits) == 1:
            return hits[0][0], hits[0][1], False
        combined = "\n".join(f"- {t}" for t, _ in hits)
        return combined, hits[0][1], False


# =====================================================================
#  Calibration
# =====================================================================
def calibrate(retriever):
    literal_queries = [
        ("你是谁", "zh"), ("你叫什么名字", "zh"),
        ("风小是谁", "zh"), ("你的名字是什么", "zh"),
        ("Who are you", "en"), ("What is your name", "en"),
    ]
    paraphrase_queries = [
        ("介绍一下你自己", "zh"), ("你叫什么", "zh"),
        ("你的名字是啥", "zh"), ("你是什么模型", "zh"),
        ("who r u", "en"), ("your name", "en"),
    ]
    miss_queries = [
        ("今天天气如何", "zh"), ("1+1等于几", "zh"),
        ("明天会下雨吗", "zh"), ("怎么做红烧肉", "zh"),
        ("北京是哪个国家的首都", "zh"),
    ]

    def run_group(name, queries):
        print("\n" + "=" * 68)
        print(f"  {name}")
        print("=" * 68)
        scores = []
        for q, expected_lang in queries:
            results = retriever.query(q, top_k=3)
            print(f"\n  Query: {q}  (expected lang={expected_lang})")
            for t, s, _ in results:
                print(f"    {s:.4f}  {t[:60]}")
            if results:
                scores.append(results[0][1])
        return scores

    hl = run_group("HIT (literal)", literal_queries)
    hp = run_group("HIT (paraphrase)", paraphrase_queries)
    ms = run_group("MISS", miss_queries)

    def s_min(xs, d=0.0): return min(xs) if xs else d
    def s_max(xs, d=1.0): return max(xs) if xs else d
    def s_mean(xs, d=0.0): return sum(xs) / len(xs) if xs else d

    min_hit = s_min(hl + hp)
    max_miss = s_max(ms)
    sep = min_hit - max_miss
    suggested = (min_hit + max_miss) / 2

    print("\n" + "=" * 68)
    print("  Calibration Summary")
    print("=" * 68)
    print(f"  Literal HIT    : min={s_min(hl):.4f}  max={s_max(hl):.4f}  "
          f"mean={s_mean(hl):.4f}")
    print(f"  Paraphrase HIT : min={s_min(hp):.4f}  max={s_max(hp):.4f}  "
          f"mean={s_mean(hp):.4f}")
    print(f"  MISS           : min={s_min(ms):.4f}  max={s_max(ms):.4f}  "
          f"mean={s_mean(ms):.4f}")
    print("-" * 68)
    print(f"  min(HIT) = {min_hit:.4f}   max(MISS) = {max_miss:.4f}   "
          f"sep = {sep:.4f}")

    if sep > 0:
        print(f"  [SUGGEST] THRESHOLD = {suggested:.3f}")
        if sep >= 0.10:
            print("  [VERDICT] Healthy.")
        elif sep >= 0.02:
            print("  [VERDICT] Marginal.")
        else:
            print("  [VERDICT] Borderline.")
    else:
        print(f"  [FAIL] overlap by {abs(sep):.4f}")
        print("  [SUGGEST] Add paraphrases to BOTH identity and input.txt, "
              "retrain.")
    print("=" * 68)
    return suggested


# =====================================================================
#  Interactive
# =====================================================================
def interactive(retriever):
    print("\n" + "=" * 68)
    print("  FengXiao Retrieval (V1.5) — Interactive")
    print(f"  Threshold: {retriever.threshold}")
    print("  Commands: 'exit' | 'debug'")
    print("=" * 68)

    verbose = False
    while True:
        try:
            user_input = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[EXIT]")
            break
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "debug":
            verbose = not verbose
            print(f"[MODE] verbose={verbose}")
            continue
        ans, score, fb = retriever.answer(user_input, verbose=verbose)
        tag = "[FALLBACK]" if fb else "[ANSWER]"
        print(f"{tag} (score={score:.4f})")
        print(f"  {ans}")


# =====================================================================
#  Main
# =====================================================================
def main():
    print("=" * 68)
    print("  FengXiao Retrieval (V1.5)")
    print("=" * 68)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Checkpoint missing: {CHECKPOINT_PATH}")
        return

    print("\n[CHECK] Identity coverage...")
    check_identity_coverage([IDENTITY_ZH, IDENTITY_EN], CORPUS_PATH)

    retriever = FengXiaoRetriever(
        CHECKPOINT_PATH,
        {"zh": IDENTITY_ZH, "en": IDENTITY_EN},
    )
    suggested = calibrate(retriever)

    print(f"\n[MODE] THRESHOLD={retriever.threshold}  SUGGESTED={suggested:.3f}")
    interactive(retriever)


if __name__ == "__main__":
    main()
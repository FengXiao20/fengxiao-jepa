# -*- coding: utf-8 -*-
"""fengxiao-intuition · 阶段2 —— 语料访问层

职责（全部只读，数据由阶段1 产物提供）:
  - packed 窗口批（LM loss 用），按 curriculum 段采样，val 窗口固定隔离
  - 题目级访问（rollout 用）：stream 里的训练题 / eval 目录里的评测题与毒题
  - 题面 prompt 切分：q_end 之后的 "=ans" 部分留给模型作答
"""
import json
import os

import numpy as np
import torch

import gen


def _prompt_cut(line):
    """返回题面（含 'ans='）的字符长度。trace 行切在 's=' 之前（模型产出轨迹）。"""
    if ",s=" in line:
        return line.rfind(",s=") + 1
    cut = line.rfind("ans=")
    if cut < 0:
        raise ValueError(f"题面无 ans=: {line!r}")
    return cut + 4


class Corpus:
    def __init__(self, data_dir, seed, val_pct=0.02):
        here = os.path.dirname(os.path.abspath(__file__))
        cdir = os.path.join(data_dir, "corpus")
        self.man = json.load(open(os.path.join(cdir, "manifest.json"), encoding="utf-8"))
        self.stream = gen.reconstruct_stream(cdir, self.man)          # uint16 1D
        b = np.load(os.path.join(cdir, "boundaries.npz"))
        self.starts, self.q_ends, self.a_ends = b["start"], b["q_end"], b["a_end"]
        self.tier, self.mods = b["tier"], b["mod"]
        self.packed = np.load(os.path.join(cdir, "packed_train.npy"))

        # 词表（单一来源：data/vocab.json，与阶段1 编码一致）
        self.itos = json.load(open(os.path.join(here, "data", "vocab.json")))["itos"]
        self.stoi = {c: i for i, c in enumerate(self.itos)}

        # 题目段索引
        self.prob_idx = {t: np.nonzero(b["tier"] == gen.TIER_CODE[t])[0]
                         for t in gen.TIER_ORDER}
        # 窗口段范围（按累计 token 边界；跨段窗口误差 ≤1 个，可忽略）
        # 硬边界：窗口 i 读 stream[i*128 .. i*128+128]（含 target），越界窗口一律不存在
        self.max_window = (len(self.stream) - 129) // 128
        prev = 0
        self.seg_windows = {}
        for t in gen.TIER_ORDER:
            cum = prev + self.man["totals"][t]["tokens"]
            lo, hi = prev // 128, (cum + 127) // 128
            self.seg_windows[t] = np.arange(lo, min(hi, self.max_window + 1))
            prev = cum

        # val 窗口固定隔离（seed 派生，resume 后不漂移）
        rng = np.random.default_rng([int(seed), 0x7A11])
        perm = rng.permutation(self.max_window + 1)
        n_val = max(1, int((self.max_window + 1) * val_pct))
        self.val_windows = set(perm[:n_val].tolist())

        # eval / 毒题集（token ids + prompt 切分）
        self.eval_sets = {}
        edir = os.path.join(data_dir, "eval")
        for fn in sorted(os.listdir(edir)):
            if not fn.endswith(".txt"):
                continue
            name = fn[:-4]
            items = []
            for line in open(os.path.join(edir, fn), encoding="ascii"):
                line = line.rstrip("\n")
                if not line:
                    continue
                cut = _prompt_cut(line)
                items.append({"text": line, "prompt_text": line[:cut],
                              "ids": self.encode(line), "prompt_ids": self.encode(line[:cut])})
            self.eval_sets[name] = items

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        special = {"<pad>", "<bos>", "<eos>"}
        return "".join(self.itos[int(i)] for i in ids
                       if int(i) < len(self.itos) and self.itos[int(i)] not in special
                       and not self.itos[int(i)].startswith("<unused"))

    # ---------------- 窗口批（LM loss） ----------------

    def _gather(self, ids):
        pos = np.asarray(ids, dtype=np.int64) * 128
        ar = np.arange(128)
        x = self.stream[pos[:, None] + ar[None, :]]
        y = self.stream[pos[:, None] + ar[None, :] + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def sample_windows(self, seg, batch_size, rng):
        cand = self.seg_windows[seg]
        ids = rng.choice(cand, size=batch_size * 3)
        ids = np.array([i for i in ids if int(i) not in self.val_windows][:batch_size])
        while len(ids) < batch_size:
            more = rng.choice(cand, size=batch_size)
            ids = np.concatenate([ids, np.array([i for i in more if int(i) not in self.val_windows])])
            ids = np.unique(ids)
        return self._gather(ids[:batch_size])

    def sample_val_windows(self, batch_size, rng):
        ids = rng.choice(sorted(self.val_windows), size=batch_size)
        return self._gather(ids)

    # ---------------- 题目级访问（rollout） ----------------

    def problem_prompt_ids(self, i):
        return self.stream[self.starts[i]: self.q_ends[i] + 1].tolist()

    def problem_prompt_text(self, i):
        return self.decode(self.problem_prompt_ids(i))

    def sample_train_problems(self, n, rng, mix):
        """按 mix 抽训练题索引。"""
        out = []
        for t, p in mix.items():
            k = int(round(n * p))
            out += rng.choice(self.prob_idx[t], size=k).tolist()
        order = rng.permutation(len(out))
        return [out[i] for i in order[:n]]

    def eval_prompt_items(self, name, n, rng):
        """从评测集抽 n 题的完整 item（ids / prompt_ids / text / prompt_text）。"""
        items = self.eval_sets[name]
        sel = rng.choice(len(items), size=min(n, len(items)), replace=False)
        return [items[i] for i in sel]

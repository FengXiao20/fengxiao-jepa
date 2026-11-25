# -*- coding: utf-8 -*-
"""scratchpad 语料生成器 —— t1 trace 格式（阶段1 spec 的实验分支，独立目录，不动 data/）。

trace 格式:  x=6,y=6,op=+,mod=7,s=12,ans=5\n
  - s = x+y 的 raw 和（0..12，不取模）—— 加法梯度落在 s 位
  - ans = s % 7 —— 取模梯度落在 ans 位
  - prompt 切在 "s=" 之前：模型必须自己产出两步轨迹

确定性 / 断点续跑 / manifest 指纹与 gen.py 同机制；打包直接调 gen.pack_corpus。
用法: python3 gen_trace.py            # 按 STAGE2_TRACE_CFG 生成 data_trace/
"""
import hashlib
import json
import os

import numpy as np

import gen

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data_trace")

TRACE_CFG = {
    "seed": 20261121,
    "seq_len": 128,
    "vocab_target": 512,
    "corpus": {
        "shard_problems": 4096,
        "target_tokens": 8388608,          # 与直答探针同预算：8M tokens
        "tier_token_quota": {"t1": 1.0, "t2": 0.0, "t3": 0.0},
        "tier_mods": {"t1": 7},
    },
}

FMT_TRACE = "x={x},y={y},op=+,mod={m},s={s},ans={a}\n"
PREFIX_LEN = len("x=0,y=0,op=+,mod=7,")   # t1 定长题面前缀 = 19（q_end = start+18）


def trace_texts(rng, mod, n):
    """生成 n 道 trace 题（t1: 单加法）。返回 (texts, x, y, s, a)。"""
    x = rng.integers(0, mod, size=n).astype(np.int64)
    y = rng.integers(0, mod, size=n).astype(np.int64)
    s = x + y
    a = s % mod
    texts = [FMT_TRACE.format(x=x[i], y=y[i], m=mod, s=s[i], a=a[i]) for i in range(n)]
    return texts, x, y, s, a


def gen_trace_corpus(cfg=TRACE_CFG, data_dir=DATA_DIR):
    cdir = os.path.join(data_dir, "corpus")
    os.makedirs(cdir, exist_ok=True)
    man = gen._load_manifest(cdir, cfg)
    enc = gen._encoder(gen.build_vocab(cfg["vocab_target"])[1])
    seed, shard_n = int(cfg["seed"]), int(cfg["corpus"]["shard_problems"])
    quota = int(cfg["corpus"]["target_tokens"])
    mod = int(cfg["corpus"]["tier_mods"]["t1"])

    total = man["totals"]["t1"]["tokens"]
    while total < quota:
        idx = sum(1 for sh in man["shards"] if sh["tier"] == "t1")
        rng = gen.shard_rng(seed, "t1", idx)
        texts, x, y, s, a = trace_texts(rng, mod, shard_n)
        done, k, cum = total, 0, 0
        for t in texts:
            if done + cum + len(t) > quota:
                break
            cum += len(t)
            k += 1
        if k == 0:
            break
        texts = texts[:k]
        x, y, s, a = x[:k], y[:k], s[:k], a[:k]
        text = "".join(texts)
        toks = enc(text)
        assert len(toks) == cum

        lens = np.fromiter((len(t) for t in texts), np.int64, k)
        start = np.concatenate(([0], np.cumsum(lens)[:-1]))
        a_end = start + lens - 1                    # '\n'
        q_end = start + (PREFIX_LEN - 1)            # 题面最后一个字符（"mod=7," 的逗号）
        tag = f"t1_{idx:04d}"
        np.savez_compressed(
            os.path.join(cdir, tag + ".npz"),
            tokens=toks,
            start=start.astype(np.int32), q_end=q_end.astype(np.int32),
            a_end=a_end.astype(np.int32),
            tier=np.full(k, gen.TIER_CODE["t1"], np.uint8), mod=np.full(k, mod, np.uint8),
            x=x.astype(np.int16), y=y.astype(np.int16),
            z=np.zeros(k, np.int16), ans=a.astype(np.int16),
            op_a=np.zeros(k, np.int8), op_b=np.full(k, -1, np.int8))
        open(os.path.join(cdir, tag + ".txt"), "w", encoding="ascii").write(text)
        sha = hashlib.sha256(text.encode("ascii")).hexdigest()
        man["shards"].append({"tier": "t1", "idx": idx, "problems": k,
                              "tokens": int(cum), "sha256": sha})
        man["totals"]["t1"]["problems"] += k
        man["totals"]["t1"]["tokens"] += int(cum)
        total = man["totals"]["t1"]["tokens"]
        gen._dump_manifest(os.path.join(cdir, "manifest.json"), man)
        print(f"[shard {tag}] +{k} problems, total {total}/{quota}")

    out = gen.pack_corpus(cfg, data_dir)
    print("[pack]", out)
    return out


def gen_trace_eval(data_dir=DATA_DIR, seed=TRACE_CFG["seed"]):
    """t1 trace 全量评测集（49 题穷举）+ 复制毒题/域外面（MISS 侧沿用阶段1 产物）。"""
    edir = os.path.join(data_dir, "eval")
    os.makedirs(edir, exist_ok=True)
    # HIT 面：t1 trace 全空间穷举（x-major 定序，确定性）
    path = os.path.join(edir, "t1_trace_mod7.txt")
    if not os.path.exists(path):
        lines = [FMT_TRACE.format(x=x, y=y, m=7, s=x + y, a=(x + y) % 7)
                 for x in range(7) for y in range(7)]
        open(path, "w", encoding="ascii").write("".join(lines))
    # MISS 面：毒题 + mod97 域外（旧格式原样复制，判卷侧由 judge_trace 判 unparseable→MISS）
    for name in ("poison_mod13", "poison_mod61", "id_mod97"):
        src = os.path.join(HERE, "data", "eval", name + ".txt")
        dst = os.path.join(edir, name + ".txt")
        if os.path.exists(src) and not os.path.exists(dst):
            open(dst, "w", encoding="ascii").write(open(src, encoding="ascii").read())
    n = {f: sum(1 for _ in open(os.path.join(edir, f), encoding="ascii"))
         for f in sorted(os.listdir(edir))}
    print("[eval]", n)


if __name__ == "__main__":
    gen_trace_corpus()
    gen_trace_eval()

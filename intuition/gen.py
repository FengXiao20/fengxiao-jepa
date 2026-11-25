# -*- coding: utf-8 -*-
"""fengxiao-intuition · 阶段1：换考卷 —— 题目生成器 / 字符级编码 / 语料打包

题面格式（字符级，机器可判卷，无歧义）:
  t1 (curriculum 档1):  x=3,y=5,op=+,mod=7,ans=1              # 模7 单加法
  t2 (curriculum 档2):  x=17,y=23,op=*,mod=97,ans=42          # 模97 混合四则
  t3 (curriculum 档3):  x=17,y=23,z=8,op1=+,op2=*,mod=97,ans=42
                        # 组合题: (x op1 y) op2 z，逐层取模

除法语义: a / b ≡ a · b^{-1} (mod m)（模逆元），要求 b ≢ 0 (mod m)。
违例样本在生成阶段自动重采样，语料中不存在非法除法题。

确定性:
  - 每个 shard 独立种子 [seed, tier_code, shard_idx]，中断后重跑自动续，bit 级一致
  - manifest.json 记录 config 指纹 + 已完成 shard（含 sha256），config 变了会拒绝静默混跑
"""
import hashlib
import json
import os
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- 字符表（单一来源常量；编码/判卷/验收共用，不从数据派生） ----------------
SPECIALS = ("<pad>", "<bos>", "<eos>")
CHARS = "0123456789xyzopmdans+-*/=,\n"   # 覆盖全部题面字符（含换行），实测 27 个

TIER_ORDER = ("t1", "t2", "t3")
TIER_CODE = {"t1": 1, "t2": 2, "t3": 3}
TIER_OPS = {"t1": ("+",), "t2": ("+", "-", "*", "/"), "t3": ("+", "-", "*", "/")}
FMT_T12 = "x={x},y={y},op={op},mod={m},ans={a}\n"
FMT_T3 = "x={x},y={y},z={z},op1={o1},op2={o2},mod={m},ans={a}\n"


def config_fingerprint(cfg):
    keys = ("seed", "seq_len", "vocab_target", "corpus")
    payload = json.dumps({k: cfg[k] for k in keys}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_vocab(vocab_target):
    """ specials + 字符表，pad 到 vocab_target。返回 (itos, stoi)。"""
    itos = list(SPECIALS) + list(CHARS)
    assert len(itos) <= vocab_target, f"字符表 {len(itos)} 超出 vocab_target {vocab_target}"
    itos += [f"<unused_{i}>" for i in range(vocab_target - len(itos))]
    return itos, {t: i for i, t in enumerate(itos)}


def ensure_vocab(cfg, data_dir):
    """落盘 vocab.json（带指纹，变了才重写）。返回指纹。"""
    itos, _ = build_vocab(cfg["vocab_target"])
    fp = hashlib.sha256("|".join(itos).encode("ascii")).hexdigest()[:16]
    path = os.path.join(data_dir, "vocab.json")
    need = True
    if os.path.exists(path):
        try:
            need = json.load(open(path, encoding="utf-8"))["fingerprint"] != fp
        except Exception:
            need = True
    if need:
        json.dump({"fingerprint": fp, "vocab_target": cfg["vocab_target"],
                   "n_real_tokens": len(SPECIALS) + len(CHARS), "itos": itos},
                  open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return fp


def _encoder(stoi):
    """查表编码：ascii bytes -> uint16 token 流（非法字符直接断言炸掉）。"""
    lut = np.full(256, 65535, dtype=np.uint16)
    for c, i in stoi.items():
        if len(c) == 1:
            lut[ord(c)] = i

    def enc(text):
        arr = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
        toks = lut[arr]
        assert not (toks == 65535).any(), "出现字符表之外的字符"
        return toks

    return enc


def shard_rng(seed, tier, idx):
    return np.random.default_rng([int(seed), TIER_CODE[tier] * 1000003, int(idx), 0xFACADE])


# ---------------- 题目生成 ----------------

def _apply(v, w, op, mod):
    """逐元素模运算（v/w 为 int64 数组）。'/' 用模逆元，调用方保证 w≢0。"""
    if op == "+":
        return (v + w) % mod
    if op == "-":
        return (v - w) % mod
    if op == "*":
        return (v * w) % mod
    if op == "/":
        inv = np.array([pow(int(wi) % mod, -1, mod) for wi in w], dtype=np.int64)
        return (v * inv) % mod
    raise ValueError(op)


def gen_batch(rng, tier, mod, n):
    """生成 n 道题。返回 (texts, meta)；meta 键与 npz schema 对齐。"""
    ops = TIER_OPS[tier]
    div = ops.index("/") if "/" in ops else -1
    x = rng.integers(0, mod, size=n).astype(np.int64)
    y = rng.integers(0, mod, size=n).astype(np.int64)
    oa = rng.integers(0, len(ops), size=n).astype(np.int64)
    if tier == "t3":
        z = rng.integers(0, mod, size=n).astype(np.int64)
        ob = rng.integers(0, len(ops), size=n).astype(np.int64)
    else:
        z = np.zeros(n, dtype=np.int64)
        ob = np.full(n, -1, dtype=np.int64)

    if div >= 0 and tier != "t3":
        bad = (oa == div) & (y % mod == 0)
        if bad.any():
            y[bad] = rng.integers(1, mod, size=int(bad.sum()))

    if tier == "t3":
        if div >= 0:
            bad = (oa == div) & (y % mod == 0)
            if bad.any():
                y[bad] = rng.integers(1, mod, size=int(bad.sum()))
        v1 = np.empty(n, dtype=np.int64)
        for gi, op in enumerate(ops):
            m_ = oa == gi
            if m_.any():
                v1[m_] = _apply(x[m_], y[m_], op, mod)
        if div >= 0:
            # 除数(第二层) = z ≢ 0
            badz = (ob == div) & (z % mod == 0)
            if badz.any():
                z[badz] = rng.integers(1, mod, size=int(badz.sum()))
            # 除数(第二层) = 中间结果 v1 ≢ 0：重抽 x,y（取 [1,mod) 保进度），循环至收敛
            badv = (ob == div) & (v1 % mod == 0)
            while badv.any():
                sel = np.nonzero(badv)[0]
                x[sel] = rng.integers(1, mod, size=len(sel))
                y[sel] = rng.integers(1, mod, size=len(sel))
                for i in sel:
                    v1[i] = _apply(x[i:i + 1], y[i:i + 1], ops[oa[i]], mod)[0]
                badv = (ob == div) & (v1 % mod == 0)
        a_in1, a_in2, a_op = v1, z, ob
    else:
        a_in1, a_in2, a_op = x, y, oa

    ans = np.empty(n, dtype=np.int64)
    for gi, op in enumerate(ops):
        m_ = a_op == gi
        if m_.any():
            ans[m_] = _apply(a_in1[m_], a_in2[m_], op, mod)

    if tier == "t3":
        texts = [FMT_T3.format(x=x[i], y=y[i], z=z[i], o1=ops[oa[i]], o2=ops[ob[i]],
                               m=mod, a=ans[i]) for i in range(n)]
    else:
        texts = [FMT_T12.format(x=x[i], y=y[i], op=ops[oa[i]], m=mod, a=ans[i])
                 for i in range(n)]
    meta = {"x": x, "y": y, "z": z, "op_a": oa, "op_b": ob, "ans": ans}
    return texts, meta


def space_size(tier, mod):
    return (mod ** 2 if tier != "t3" else mod ** 3) * (len(TIER_OPS[tier]) ** (1 if tier != "t3" else 2))


# ---------------- 训练语料（shard 化 + manifest 断点续跑） ----------------

def _dump_manifest(path, man):
    tmp = path + ".tmp"
    json.dump(man, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _load_manifest(cdir, cfg):
    path = os.path.join(cdir, "manifest.json")
    fp = config_fingerprint(cfg)
    if os.path.exists(path):
        man = json.load(open(path, encoding="utf-8"))
        if man.get("config_fp") != fp:
            raise RuntimeError(
                f"manifest 与当前 config 不一致（{man.get('config_fp')} != {fp}）。"
                f"若确要换参数，先清空 {cdir} 再跑，禁止新旧语料混跑。")
        return man
    return {"config_fp": fp, "seed": cfg["seed"], "shards": [],
            "totals": {t: {"problems": 0, "tokens": 0} for t in TIER_ORDER}}


def gen_corpus(cfg, data_dir):
    cdir = os.path.join(data_dir, "corpus")
    os.makedirs(cdir, exist_ok=True)
    man = _load_manifest(cdir, cfg)
    enc = _encoder(build_vocab(cfg["vocab_target"])[1])
    corp = cfg["corpus"]
    seed, shard_n = int(cfg["seed"]), int(corp["shard_problems"])
    quota = {t: int(corp["target_tokens"] * corp["tier_token_quota"][t]) for t in TIER_ORDER}

    for tier in TIER_ORDER:
        mod = int(corp["tier_mods"][tier])
        while man["totals"][tier]["tokens"] < quota[tier]:
            idx = sum(1 for s in man["shards"] if s["tier"] == tier)
            rng = shard_rng(seed, tier, idx)
            texts, meta = gen_batch(rng, tier, mod, shard_n)
            # 按 token 配额整题截断（最后一 shard 收尾）
            done, k, cum = man["totals"][tier]["tokens"], 0, 0
            for t in texts:
                if done + cum + len(t) > quota[tier]:
                    break
                cum += len(t)
                k += 1
            if k == 0:
                break
            texts = texts[:k]
            meta = {kk: vv[:k] for kk, vv in meta.items()}
            text = "".join(texts)
            toks = enc(text)
            assert len(toks) == cum

            lens = np.fromiter((len(t) for t in texts), np.int64, k)
            start = np.concatenate(([0], np.cumsum(lens)[:-1]))
            adig = np.fromiter((len(str(int(a))) for a in meta["ans"]), np.int64, k)
            a_end = start + lens - 1          # '\n' 位置
            q_end = a_end - adig - 1          # 'ans=' 的 '=' —— 题面最后一个 token

            tag = f"{tier}_{idx:04d}"
            np.savez_compressed(
                os.path.join(cdir, tag + ".npz"),
                tokens=toks,
                start=start.astype(np.int32), q_end=q_end.astype(np.int32),
                a_end=a_end.astype(np.int32),
                tier=np.full(k, TIER_CODE[tier], np.uint8), mod=np.full(k, mod, np.uint8),
                x=meta["x"].astype(np.int16), y=meta["y"].astype(np.int16),
                z=meta["z"].astype(np.int16), ans=meta["ans"].astype(np.int16),
                op_a=meta["op_a"].astype(np.int8), op_b=meta["op_b"].astype(np.int8))
            open(os.path.join(cdir, tag + ".txt"), "w", encoding="ascii").write(text)
            sha = hashlib.sha256(text.encode("ascii")).hexdigest()
            man["shards"].append({"tier": tier, "idx": idx, "problems": k,
                                  "tokens": int(cum), "sha256": sha})
            man["totals"][tier]["problems"] += k
            man["totals"][tier]["tokens"] += int(cum)
            _dump_manifest(os.path.join(cdir, "manifest.json"), man)

    out = {"total_problems": sum(v["problems"] for v in man["totals"].values()),
           "total_tokens": sum(v["tokens"] for v in man["totals"].values()),
           "per_tier": man["totals"]}
    return out


# ---------------- 评测集 / 毒题集 ----------------

def gen_eval_sets(cfg, data_dir):
    """mod13/61 毒题（OOD，永不入训）+ mod97 in-dist 评测 + t1 全量评测。内部去重。"""
    out = os.path.join(data_dir, "eval")
    os.makedirs(out, exist_ok=True)
    seed = int(cfg["seed"])
    info = {}
    for name, spec in cfg["eval_sets"].items():
        path = os.path.join(out, name + ".txt")
        if os.path.exists(path):
            info[name] = {"n": sum(1 for _ in open(path, encoding="ascii")), "cached": True}
            continue
        mod, n = int(spec["mod"]), int(spec["n"])
        t3r = float(spec.get("t3_ratio", 0.0))
        rng = np.random.default_rng([seed, 0xE7A1, zlib.crc32(name.encode())])
        seen, texts, warns = set(), [], []

        if spec.get("tier") == "t1":
            plan = [("t1", n)]
        else:
            n3 = int(round(n * t3r))
            plan = [("t3", n3), ("t2", n - n3)]

        for tier, want in plan:
            got, space = 0, space_size(tier, mod)
            seen_tier = set()   # per-tier 独立计数，防止跨层污染空间上限判断
            while got < want and len(seen_tier) < space:
                tb, _ = gen_batch(rng, tier, mod, 4096)
                for t in tb:
                    seen_tier.add(t)
                    if t not in seen:
                        seen.add(t)
                        texts.append(t)
                        got += 1
                        if got >= want:
                            break
            if got < want:
                warns.append(f"{tier}: 空间仅 {space}，只凑到 {got}")

        open(path, "w", encoding="ascii").write("".join(texts))
        info[name] = {"n": len(texts), "requested": n, "warnings": warns}
    return info


# ---------------- 打包（LM 预训练窗口 + 全局边界元数据） ----------------

def reconstruct_stream(cdir, man):
    """按 t1->t2->t3 顺序把全部 shard 的 token 流拼回一维 stream。"""
    parts = []
    for tier in TIER_ORDER:
        for s in sorted((x for x in man["shards"] if x["tier"] == tier),
                        key=lambda x: x["idx"]):
            parts.append(np.load(os.path.join(cdir, f"{tier}_{s['idx']:04d}.npz"))["tokens"])
    return np.concatenate(parts)


def pack_corpus(cfg, data_dir):
    """把 token 流切成 (N, seq_len) 窗口（标准 LM packing，题目可跨窗口），
    并落盘全局边界 boundaries.npz —— 阶段2 的 FOK/confidence 头按
    全局 offset // seq_len 换算窗口、offset % seq_len 换算列。"""
    cdir = os.path.join(data_dir, "corpus")
    man = _load_manifest(cdir, cfg)
    seq_len = int(cfg["seq_len"])

    stream = reconstruct_stream(cdir, man)
    assert len(stream) == sum(v["tokens"] for v in man["totals"].values())

    keys = ("start", "q_end", "a_end", "tier", "mod", "x", "y", "z", "ans", "op_a", "op_b")
    toks_parts, bounds, base = [], {k: [] for k in keys}, 0
    for tier in TIER_ORDER:
        for s in sorted((x for x in man["shards"] if x["tier"] == tier),
                        key=lambda x: x["idx"]):
            d = np.load(os.path.join(cdir, f"{tier}_{s['idx']:04d}.npz"))
            for k in keys:
                arr = d[k].astype(np.int64) if k in ("start", "q_end", "a_end") else d[k]
                bounds[k].append(arr + base if k in ("start", "q_end", "a_end") else arr)
            toks_parts.append(d["tokens"])
            base += len(d["tokens"])
    bounds = {k: np.concatenate(v) for k, v in bounds.items()}

    n_win = len(stream) // seq_len
    body = stream[:n_win * seq_len].reshape(n_win, seq_len)
    tail = stream[n_win * seq_len:]
    if len(tail):
        pad_row = np.full(seq_len, 0, dtype=np.uint16)   # <pad>=0
        pad_row[:len(tail)] = tail
        packed = np.concatenate([body, pad_row[None]], axis=0)
    else:
        packed = body
    np.save(os.path.join(cdir, "packed_train.npy"), packed)
    np.savez_compressed(os.path.join(cdir, "boundaries.npz"), **bounds)
    json.dump({"seq_len": seq_len, "n_windows": int(len(packed)),
               "stream_len": int(len(stream)), "tail_padded": int(len(tail)), "pad_id": 0},
              open(os.path.join(cdir, "pack_meta.json"), "w"), indent=1)
    return {"windows": int(len(packed)), "tokens": int(len(stream)),
            "tail_padded": int(len(tail)), "problems": int(len(bounds["start"]))}

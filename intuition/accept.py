# -*- coding: utf-8 -*-
"""阶段1 验收：
A. shard 完整性（sha256 全量核对）
B. 训练语料全量判卷（~62 万题，死因枚举 + 去重统计 + 字符表覆盖检查，单遍完成）
C. 评测集/毒题集全量判卷
D. packed 窗口 <-> token 流 往返一致性
E. 100 题分层抽样判卷（验收口径）
产物: artifacts/stage1_acceptance_report.json + stage1_acceptance_samples.txt
"""
import hashlib
import json
import os
from collections import Counter

import numpy as np

import gen
import judge


def _txt_path(cdir, s):
    return os.path.join(cdir, f"{s['tier']}_{s['idx']:04d}.txt")


def run_accept(cfg, here, data_dir):
    cdir = os.path.join(data_dir, "corpus")
    edir = os.path.join(data_dir, "eval")
    adir = os.path.join(here, "artifacts")
    os.makedirs(adir, exist_ok=True)
    man = json.load(open(os.path.join(cdir, "manifest.json"), encoding="utf-8"))
    rng = np.random.default_rng([int(cfg["seed"]), 0xACCE97])

    rep = {"seed": int(cfg["seed"]), "config_fp": gen.config_fingerprint(cfg)}

    # A. shard 完整性
    bad = []
    for s in man["shards"]:
        h = hashlib.sha256(open(_txt_path(cdir, s), "rb").read()).hexdigest()
        if h != s["sha256"]:
            bad.append(s["tier"] + str(s["idx"]))
    rep["checks"] = {"shard_integrity": {"total": len(man["shards"]), "failed": bad}}

    # B. 全量判卷（单遍：死因枚举 + distinct + charset）
    reasons = Counter()
    distinct = {t: set() for t in gen.TIER_ORDER}
    bad_chars = Counter()
    allowed = set(gen.CHARS)
    n_lines = 0
    for s in man["shards"]:
        tier = s["tier"]
        with open(_txt_path(cdir, s), encoding="ascii") as f:
            for line in f:
                n_lines += 1
                reasons[judge.grade(line)["reason"]] += 1
                distinct[tier].add(line.rstrip("\n"))
                bad_chars.update(set(line) - allowed)
    rep["judge_all"] = {"lines": n_lines, "reasons": dict(reasons)}
    rep["distinct"] = {t: {"count": len(distinct[t]), "space": gen.space_size(
        t, int(cfg["corpus"]["tier_mods"][t]))} for t in gen.TIER_ORDER}
    del distinct
    rep["checks"]["charset"] = {"ok": not bad_chars, "bad_chars": dict(bad_chars)}

    # C. 评测集判卷
    ev = {}
    for name in cfg["eval_sets"]:
        r, n = Counter(), 0
        with open(os.path.join(edir, name + ".txt"), encoding="ascii") as f:
            for line in f:
                n += 1
                r[judge.grade(line)["reason"]] += 1
        ev[name] = {"n": n, "reasons": dict(r)}
    rep["eval_sets"] = ev

    # D. packed 往返一致性
    packed = np.load(os.path.join(cdir, "packed_train.npy"))
    stream = gen.reconstruct_stream(cdir, man)
    sl = int(cfg["seq_len"])
    n_win = len(stream) // sl
    rt_ok = bool(np.array_equal(packed[:n_win].reshape(-1),
                                stream[:n_win * sl]))
    tail_ok = True
    if len(stream) % sl:
        tail_ok = bool(np.array_equal(packed[n_win, :len(stream) % sl],
                                      stream[n_win * sl:]) and
                       (packed[n_win, len(stream) % sl:] == 0).all())
    rep["checks"]["pack_roundtrip"] = {"ok": rt_ok and tail_ok,
                                       "windows": int(len(packed))}

    # E. 100 题分层抽样
    srcs = cfg["acceptance"]["sources"]
    samples, sreasons = [], Counter()

    def _record(src, line):
        g = judge.grade(line)
        sreasons[g["reason"]] += 1
        samples.append((src, g))

    for src, k in srcs.items():
        if src.startswith("train_"):
            tier = src.split("_", 1)[1]
            shards = [s for s in man["shards"] if s["tier"] == tier]
            pool, pid = None, None
            for _ in range(k):
                s = shards[int(rng.integers(len(shards)))]
                sid = s["tier"] + str(s["idx"])
                if pool is None or sid != pid:
                    with open(_txt_path(cdir, s), encoding="ascii") as f:
                        pool, pid = f.readlines(), sid
                _record(src, pool[int(rng.integers(len(pool)))])
        else:
            with open(os.path.join(edir, src + ".txt"), encoding="ascii") as f:
                fl = f.readlines()
            for _ in range(k):
                _record(src, fl[int(rng.integers(len(fl)))])
    lines_out = ["# fengxiao-intuition stage1 验收抽样  seed={}  n={}".format(
        cfg["seed"], len(samples))]
    for src, g in samples:
        lines_out.append("[src={src}] {verdict:<14} {line:<45} expected={exp}".format(
            src=src, verdict=g["reason"].upper(), line=g["line"],
            exp=g.get("expected", "-")))
    open(os.path.join(adir, "stage1_acceptance_samples.txt"), "w",
         encoding="utf-8").write("\n".join(lines_out) + "\n")
    rep["sample"] = {"n": len(samples), "reasons": dict(sreasons)}

    # 汇总判定
    total_tokens = sum(v["tokens"] for v in man["totals"].values())
    eval_pass = all(set(e["reasons"]) <= {"pass"} for e in ev.values())
    rep["summary"] = {
        "total_problems": sum(v["problems"] for v in man["totals"].values()),
        "total_tokens": total_tokens,
        "corpus_judge": "{} pass / {} total".format(reasons.get("pass", 0), n_lines),
        "sample": "{}/{} pass".format(sreasons.get("pass", 0), len(samples)),
        "all_checks_ok": bool(
            not bad and not bad_chars and rt_ok and tail_ok
            and reasons.get("pass", 0) == n_lines and eval_pass
            and sreasons.get("pass", 0) == len(samples)),
    }

    path = os.path.join(adir, "stage1_acceptance_report.json")
    json.dump(rep, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return rep

# -*- coding: utf-8 -*-
"""fengxiao-intuition · 阶段2 —— 验收报告（可独立重跑）

验收口径（任务规格）: confidence 头输出与实际正确率相关性 > 0.3。
同时对 FOK 头出同口径读数（作答前直觉 vs 作答后自信）。

做法: 加载最新 epoch 快照，在四份考卷各抽 per_set 题做 rollout，
逐题判卷回填标签，算点二列相关（Pearson）+ 校准图 + ledger。
产物: artifacts/stage2_acceptance_report.json + runs/stage2/ledger_eval.csv

用法: python stage2_report.py --config stage2.json [--ckpt runs/stage2/epoch_10.pt]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import model as model_mod
import train_stage2 as ts2
from model import IntuitionLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "stage2.json"))
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    run_dir = os.path.join(HERE, cfg["paths"]["run_dir"])
    ckpt_path = a.ckpt
    if ckpt_path is None:
        cands = sorted(f for f in os.listdir(run_dir)
                       if f.startswith("epoch_") and f.endswith(".pt"))
        if os.path.exists(os.path.join(run_dir, "resume.pth")):
            ckpt_path = os.path.join(run_dir, "resume.pth")
        elif cands:
            ckpt_path = os.path.join(run_dir, cands[-1])
        else:
            raise SystemExit("没有可用的 checkpoint")
    print(f"[report] ckpt = {ckpt_path}")

    set_seed_seed = int(cfg["seed"])
    ts2.set_seed(set_seed_seed)
    ts2.select_judge(cfg)   # trace 语料 → judge_trace（与训练侧同一旋钮）
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = IntuitionLM(vocab_size=cfg["vocab_target"],
                        embed_dim=cfg["model"]["embed_dim"],
                        num_heads=cfg["model"]["num_heads"],
                        num_layers=cfg["model"]["num_layers"],
                        dropout=cfg["model"]["dropout"],
                        head_hidden=cfg["model"]["head_hidden"])
    model.load_state_dict(ck["model_state"])
    corpus = ts2.corpus_mod.Corpus(os.path.join(HERE, cfg["paths"].get("data_dir", "data")),
                                   set_seed_seed, cfg["train"]["val_windows_pct"])

    rng = np.random.default_rng([set_seed_seed, 0x2EAC])
    n_per_set = max(1, cfg["eval_rollout"]["per_set"] // len(cfg["eval_rollout"]["sets"]))
    rollouts = []
    for name in cfg["eval_rollout"]["sets"]:
        rollouts += ts2.do_rollout(model, corpus, cfg, rng,
                                   device="cuda" if torch.cuda.is_available() else "cpu",
                                   eval_sets=[name], per_override=n_per_set)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fok_l, conf_l, fok_p, conf_p = ts2.head_losses(model, rollouts, dev)
    for r, fp, cp in zip(rollouts, fok_p, conf_p):
        r["fok"], r["conf"] = fp, cp
    ts2.append_ledger(run_dir, rollouts, split="eval_final")

    # 逐来源读数
    acc = {}
    for r in rollouts:
        acc.setdefault(r["item"]["source"], []).append(r)
    per_src = {}
    for src, rs in acc.items():
        c = ts2.correlations(rs)
        per_src[src] = {"n": len(rs), "accuracy": c["accuracy"],
                        "conf_corr": c["conf_corr"], "fok_corr": c["fok_corr"]}
    all_corr = ts2.correlations(rollouts)

    rep = {"ckpt": os.path.basename(ckpt_path), "n_eval": len(rollouts),
           "overall": all_corr, "per_source": per_src,
           "threshold": 0.3,
           "conf_accept": bool(all_corr["conf_corr"] > 0.3 and all_corr["n"] >= 80),
           "fok_corr": all_corr["fok_corr"]}

    ts2.plot_calibration(run_dir, rollouts, epoch=ck.get("epoch", 0), tag="final")
    out = os.path.join(HERE, "artifacts", "stage2_acceptance_report.json")
    json.dump(rep, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

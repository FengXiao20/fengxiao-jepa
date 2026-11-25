# -*- coding: utf-8 -*-
"""fengxiao-intuition · 阶段2 —— 全程监控（挂机脚本，触发才报警）

消费 train_stage2.py 产出的 metrics.jsonl（每 N 步逐层 weight std/mean 摘要），
三条红旗（对应任务规格）:
  R1  embed 与主干 scale 差 > 10 倍
  R2  任何头 weights std 趋零（< 1e-4 且仍在下降）
  R3  激活范数突变（偏离滚动中位数 5 倍以上）

无事时静默退出（exit 0）；触发则写 alerts/alerts.jsonl 并 exit 1。
用法:
  python monitor.py --run runs/stage2            # 单次检查（适合 cron / 心跳）
  python monitor.py --run runs/stage2 --watch    # 常驻挂机，60s 一查
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_metrics(run_dir):
    path = os.path.join(HERE, run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def check(rows, lookback=50):
    """返回告警列表（空 = 静默）。只看最近 lookback 条，历史仅供参考。"""
    alerts = []
    if not rows:
        return alerts
    recent = rows[-lookback:]

    # R1: embed 与主干 scale 差 > 10×
    r = recent[-1]
    if r.get("blk_std_mean") and r.get("emb_std"):
        ratio = r["emb_std"] / max(r["blk_std_mean"], 1e-12)
        if ratio > 10 or ratio < 0.1:
            alerts.append({"flag": "R1_embed_scale", "step": r["step"],
                           "ratio": round(ratio, 3)})

    # R2: 头 weights std 趋零
    for key, name in (("fok_std", "fok_head"), ("conf_std", "conf_head")):
        vals = [x[key] for x in recent if x.get(key) is not None]
        if len(vals) >= 2:
            slope = vals[-1] - vals[0]
            if vals[-1] < 1e-4 and slope <= 0:
                alerts.append({"flag": "R2_head_collapse", "step": r["step"],
                               "head": name, "std": vals[-1]})

    # R3: 激活范数突变（滚动中位数 5 倍）
    acts = [x["act_norm"] for x in recent if x.get("act_norm")]
    if len(acts) >= 5:
        med = float(np.median(acts[:-1]))
        if med > 0 and abs(acts[-1] - med) > 5 * med:
            alerts.append({"flag": "R3_act_norm_jump", "step": r["step"],
                           "last": acts[-1], "median": med})
    return alerts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/stage2")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    a = ap.parse_args()

    alert_path = os.path.join(HERE, a.run, "alerts", "alerts.jsonl")
    while True:
        alerts = check(load_metrics(a.run))
        if alerts:
            os.makedirs(os.path.dirname(alert_path), exist_ok=True)
            with open(alert_path, "a", encoding="utf-8") as f:
                for al in alerts:
                    f.write(json.dumps({"ts": time.strftime("%F %T"), **al},
                                       ensure_ascii=False) + "\n")
            print("[ALERT]", json.dumps(alerts, ensure_ascii=False), flush=True)
            rc = 1
        else:
            rc = 0  # 静默：无事不扰
        if not a.watch:
            return rc
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""fengxiao-intuition 统一入口。

用法（在 intuition/ 目录下）:
  python run.py --config config.json --stage all     # 全流程，断点续跑
  python run.py --stage gen                          # 只补训练语料
  python run.py --stage eval                         # 只补评测/毒题集
  python run.py --stage pack                         # 只重打包
  python run.py --stage accept                       # 只跑验收
阶段顺序: vocab -> gen -> eval -> pack -> accept，均幂等可重入。
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import accept  # noqa: E402
import gen     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--stage", default="all",
                    choices=["vocab", "gen", "eval", "pack", "accept", "all"])
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    data_dir = os.path.join(HERE, cfg.get("data_dir", "data"))
    os.makedirs(data_dir, exist_ok=True)

    def log(event, **kw):
        print(json.dumps({"event": event, **kw}, ensure_ascii=False), flush=True)

    stages = ["vocab", "gen", "eval", "pack", "accept"] if a.stage == "all" else [a.stage]
    for st in stages:
        if st == "vocab":
            fp = gen.ensure_vocab(cfg, data_dir)
            log("vocab_ok", fingerprint=fp, n_chars=len(gen.CHARS))
        elif st == "gen":
            log("gen_done", **gen.gen_corpus(cfg, data_dir))
        elif st == "eval":
            log("eval_sets_done", **gen.gen_eval_sets(cfg, data_dir))
        elif st == "pack":
            log("pack_done", **gen.pack_corpus(cfg, data_dir))
        elif st == "accept":
            rep = accept.run_accept(cfg, HERE, data_dir)
            log("accept_done", summary=rep["summary"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

# -*- coding: utf-8 -*-
"""scratchpad 探针：trace 两步轨迹的 teacher-forced 命中率（s 位与 ans 位分开读数）。

用法: python3 probe_tf_trace.py <ckpt.pth> [n_sets...]   # 默认测 t1_trace_mod7
指标:
  s_digit_acc   raw 和 s 的每个数字位 argmax 命中率（对照随机 1/10+位权）
  s_exact_acc   整个 s 数值全对率（随机基线 ≈ 1/13 加权）
  ans_acc       取模答案位命中率（对照直答探针随机线 1/7=0.143）
  ansCE         ans 位交叉熵（对照直答探针钉死的 ln7=1.946）
"""
import sys
import os
import json

import numpy as np
import torch

import corpus as corpus_mod
from model import IntuitionLM

HERE = os.path.dirname(os.path.abspath(__file__))


def tf_trace(model, corpus, items):
    """逐集合批量 teacher-forcing。返回 (s_digit_acc, s_exact, ans_acc, ansCE)。"""
    s_hits, s_exact, a_hits, a_ces = [], [], [], []
    by_len = {}
    for it in items:
        by_len.setdefault(len(it["ids"]), []).append(it["ids"])
    with torch.no_grad():
        for L, group in sorted(by_len.items()):
            for i in range(0, len(group), 256):
                chunk = group[i:i + 256]
                x = torch.tensor([ids[:-1] for ids in chunk], dtype=torch.long)
                h = model.features(x)
                logits = model.lm_head(h)                       # (B, T-1, V)
                logit_at = lambda b, k: logits[b, k - 1]        # 预测 ids[k] 的位置
                for b, ids in enumerate(chunk):
                    text = corpus.decode(ids)
                    s0 = text.find("s=") + 2
                    s1 = text.rfind(",ans=")
                    a0 = text.rfind("ans=") + 4
                    sd = [int(logits[b, k - 1].argmax().item()) == ids[k]
                          for k in range(s0, s1)]
                    s_hits += sd
                    s_exact.append(all(sd))
                    a_hits.append(int(logits[b, a0 - 1].argmax().item()) == ids[a0])
                    a_ces.append(float(torch.nn.functional.cross_entropy(
                        logits[b, a0 - 1][None], torch.tensor([ids[a0]]))))
    return (float(np.mean(s_hits)), float(np.mean(s_exact)),
            float(np.mean(a_hits)), float(np.mean(a_ces)))


def main():
    ckpt_path = sys.argv[1]
    sets = sys.argv[2:] or ["t1_trace_mod7"]
    cfg = json.load(open(os.path.join(HERE, "stage2_scratch.json"), encoding="utf-8"))
    corpus = corpus_mod.Corpus(os.path.join(HERE, "data_trace"), cfg["seed"])
    model = IntuitionLM(vocab_size=cfg["vocab_target"], **cfg["model"])
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print(f"# ckpt={os.path.basename(ckpt_path)} epoch={ck.get('epoch')} step={ck.get('global_step')}")
    for name in sets:
        items = corpus.eval_sets[name]
        sd, se, aa, ace = tf_trace(model, corpus, items)
        print(f"{name:16s} n={len(items):4d} s_digit={sd:.3f} s_exact={se:.3f} "
              f"ans_acc={aa:.3f} ansCE={ace:.3f}")


if __name__ == "__main__":
    main()

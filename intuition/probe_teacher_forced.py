# -*- coding: utf-8 -*-
"""探针收割：答案位 teacher-forced 命中率（对照 α 收割的 16%/t1, 4%/mod97）。

用法: python3 probe_teacher_forced.py <ckpt.pth> [set1 set2 ...]
不给 set 时默认测 t1_mod7 + id_mod97。
"""
import sys
import os
import json

import numpy as np
import torch

import corpus as corpus_mod
from model import IntuitionLM

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ckpt_path = sys.argv[1]
    sets = sys.argv[2:] or ["t1_mod7", "id_mod97"]

    cfg = json.load(open(os.path.join(HERE, "stage2_probe.json"), encoding="utf-8"))
    corpus = corpus_mod.Corpus(os.path.join(HERE, "data"), cfg["seed"])
    model = IntuitionLM(vocab_size=cfg["vocab_target"], **cfg["model"])
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print(f"# ckpt={os.path.basename(ckpt_path)} epoch={ck.get('epoch')} step={ck.get('global_step')}")

    for name in sets:
        items = corpus.eval_sets[name]
        hits, ces, argmaxes = [], [], []
        by_len = {}
        for it in items:
            by_len.setdefault(len(it["ids"]), []).append(it["ids"])
        with torch.no_grad():
            for L, group in sorted(by_len.items()):
                for i in range(0, len(group), 256):
                    chunk = group[i:i + 256]
                    x = torch.tensor([ids[:-1] for ids in chunk], dtype=torch.long)
                    tgt = torch.tensor([ids[-1] for ids in chunk])
                    h = model.features(x)                     # (B, T, C)
                    logits = model.lm_head(h[:, -1])          # 末位预测答案位
                    hits += (logits.argmax(-1) == tgt).tolist()
                    ces += torch.nn.functional.cross_entropy(
                        logits, tgt, reduction="none").tolist()
                    argmaxes += logits.argmax(-1).tolist()
        digit = lambda t: corpus.itos[t]
        dist = {}
        for a in argmaxes:
            dist[digit(a)] = dist.get(digit(a), 0) + 1
        print(f"{name:12s} n={len(items):4d} tf_hit={np.mean(hits):.3f} "
              f"ansCE={np.mean(ces):.3f} argmax分布={dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()

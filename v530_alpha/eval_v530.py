# -*- coding: utf-8 -*-
"""
FengXiao eval_v530.py — v5.3.0 训练后评估
==========================================
对 best_sep checkpoint 做全量评估, 与 v5.1.1 基线对照:
  1. calibrate 全量(sep / HIT / MISS / suggested threshold)
  2. identity 索引 EffRank(各向异性火警指标)
  3. 权重体检摘要(weights_report 逻辑内联, 只报嵌入矩阵)

用法: python eval_v530.py [checkpoint]   默认 checkpoints_v530/fengxiao_best_sep.pth
"""
import io
import sys
import contextlib

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import retrieve as R

R.USE_MEAN_SUBTRACTION = False   # 纯 v5.1.1 检索形态, A/B 公平

DEFAULT_CKPT = "./checkpoints_v530/fengxiao_best_sep.pth"


def eff_rank(vectors):
    s = torch.linalg.svdvals(vectors - vectors.mean(0, keepdim=True))
    s = s.clamp(min=0)
    return float((s.sum() ** 2) / ((s * s).sum() + 1e-12))


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
    print("=" * 68)
    print(f"  FengXiao v5.3.0 Evaluation — {ckpt_path}")
    print("=" * 68)

    ret = R.FengXiaoRetriever(ckpt_path, {"zh": R.IDENTITY_ZH, "en": R.IDENTITY_EN})

    # ---- 1. EffRank ----
    for lang in ("zh", "en"):
        v = ret.indexes[lang]["vectors"]
        print(f"[EffRank] {lang} index ({v.shape[0]} vecs): {eff_rank(v):.2f}/{v.shape[1]}")
    all_v = torch.cat([ret.indexes[l]["vectors"] for l in ret.indexes])
    print(f"[EffRank] combined: {eff_rank(all_v):.2f}/{all_v.shape[1]}"
          f"   (基线 v5.1.1: 7.53/128, 白化后目标 ~128)")

    # ---- 2. calibrate 全量 ----
    print("\n[calibrate] full run (输出重定向至 eval_calibrate_output.txt)")
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        suggested = R.calibrate(ret)
    with open("eval_calibrate_output.txt", "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    tail = [l for l in buf.getvalue().splitlines()
            if "sep" in l or "VERDICT" in l or "SUGGEST" in l]
    for l in tail:
        print("   " + l.strip())
    print(f"\n[suggested threshold] {suggested:.3f}")

    # ---- 3. 嵌入矩阵体检摘要 ----
    print("\n[weights check] token_embed / target_encoder 末层 EffRank:")
    sd = ret.model.state_dict()
    for k in sd:
        if k == "token_embed.weight" or (
                "target_encoder" in k and k.endswith("linear2.weight")):
            t = sd[k]
            s = torch.linalg.svdvals(t)
            er = float((s.sum() ** 2) / ((s * s).sum() + 1e-12))
            print(f"   {k}: EffRank {er:.1f}/{t.shape[1]}")
    print("=" * 68)


if __name__ == "__main__":
    main()

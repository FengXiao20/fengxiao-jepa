# -*- coding: utf-8 -*-
"""scratchpad（计算轨迹）判卷器 —— 与 gen_trace.py 完全独立：独立正则、独立参考实现。

trace 题面格式（t1，模7 加法）:
  完整行:  x=6,y=6,op=+,mod=7,s=12,ans=5
  题面(prompt): x=6,y=6,op=+,mod=7,     ← 切在 "s=" 之前
  模型作答:     s=12,ans=5              ← 两步轨迹: 先算 raw 和 s，再取模得 ans

死因枚举（与 judge.py 同风格，不用自然语言）:
  pass             s 与 ans 全对且格式规范
  s_mismatch       中间和 s 错
  ans_mismatch     ans 错（含 s 对但取模错）
  ans_out_of_range ans 越界（不在 [0, mod)）
  noncanonical     数字字段有前导零等非规范写法
  unparseable      结构坏 / 缺轨迹字段
"""
import re

_TRACE_T1 = re.compile(r"x=(\d),y=(\d),op=\+,mod=(\d),s=(\d{1,2}),ans=(\d)")


def parse(line):
    """严格单解析。返回 fields dict；结构坏抛 ValueError。"""
    s = line.rstrip("\n")
    m = _TRACE_T1.fullmatch(s)
    if m is None:
        raise ValueError(f"无法解析 trace 行: {s!r}")
    f = {"tier": "t1_trace", "x": int(m[1]), "y": int(m[2]), "op": "+",
         "mod": int(m[3]), "s": int(m[4]), "ans": int(m[5])}
    raw = {"x": m[1], "y": m[2], "mod": m[3], "s": m[4], "ans": m[5]}
    f["canonical"] = all(str(f[k]) == r for k, r in raw.items())
    f["_raw"] = s
    return f


def ref_answer(f):
    """参考实现：raw 和 s = x+y（不取模），ans = s % mod。独立于 gen_trace.py。"""
    s = f["x"] + f["y"]
    return s, s % f["mod"]


def grade(line):
    """判一行 trace 题。返回 {ok, reason, expected, given, ...}。"""
    base = {"line": line.rstrip("\n")}
    try:
        f = parse(line)
    except ValueError as e:
        return dict(base, ok=False, reason="unparseable", detail=str(e))
    out = dict(base, tier=f["tier"], mod=f["mod"], given=f["ans"])
    if not f["canonical"]:
        return dict(out, ok=False, reason="noncanonical")
    exp_s, exp_a = ref_answer(f)
    out["expected_s"], out["expected"] = exp_s, exp_a
    if f["s"] != exp_s:
        return dict(out, ok=False, reason="s_mismatch")
    if not (0 <= f["ans"] < f["mod"]):
        return dict(out, ok=False, reason="ans_out_of_range")
    if f["ans"] != exp_a:
        return dict(out, ok=False, reason="ans_mismatch")
    return dict(out, ok=True, reason="pass")


def grade_s_only(line):
    """只判 s 步（诊断用：把'加法'与'取模'两段学习分开看）。"""
    try:
        f = parse(line)
        return f["s"] == f["x"] + f["y"]
    except ValueError:
        return False

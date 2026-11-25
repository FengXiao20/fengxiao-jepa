# -*- coding: utf-8 -*-
"""机器判卷器 —— 与生成器（gen.py）完全独立：独立正则、独立参考答案实现。
"考卷-判卷"两侧不共享算术代码，验收结论才站得住。阶段2/3 的 rollout 判卷也走这里。

死因枚举（不用自然语言）:
  pass            答案正确且格式规范
  ans_mismatch    答案错
  ans_out_of_range 答案越界（不在 [0, mod)）
  noncanonical    数字字段有前导零等非规范写法
  ambiguous       无法唯一解析（既像 t12 又像 t3 / 都不像）
  unparseable     结构坏
  ref_error       参考答案计算异常（如模零除法——语料里不该有）
"""
import re

_T12 = re.compile(r"x=(\d+),y=(\d+),op=([+\-*/]),mod=(\d+),ans=(\d+)")
_T3 = re.compile(r"x=(\d+),y=(\d+),z=(\d+),op1=([+\-*/]),op2=([+\-*/]),mod=(\d+),ans=(\d+)")


class Ambiguous(Exception):
    pass


def parse(line):
    """严格单解析：返回 fields dict；无法唯一解析抛 Ambiguous，结构坏抛 ValueError。"""
    s = line.rstrip("\n")
    m12, m3 = _T12.fullmatch(s), _T3.fullmatch(s)
    if (m12 is None) == (m3 is None):
        raise Ambiguous(f"无法唯一解析: {s!r}")
    if m12 is not None:
        f = {"tier": "t12", "x": int(m12[1]), "y": int(m12[2]), "op": m12[3],
             "mod": int(m12[4]), "ans": int(m12[5])}
        raw = {"x": m12[1], "y": m12[2], "mod": m12[4], "ans": m12[5]}
    else:
        f = {"tier": "t3", "x": int(m3[1]), "y": int(m3[2]), "z": int(m3[3]),
             "op1": m3[4], "op2": m3[5], "mod": int(m3[6]), "ans": int(m3[7])}
        raw = {"x": m3[1], "y": m3[2], "z": m3[3], "mod": m3[6], "ans": m3[7]}
    f["canonical"] = all(str(f[k]) == r for k, r in raw.items())
    f["_raw"] = s
    return f


def ref_answer(f):
    """参考答案：独立于 gen.py 的算术实现。"""
    m = f["mod"]

    def ap(v, w, op):
        if not (0 <= v < m and 0 <= w < m):
            raise ValueError(f"操作数越界 v={v} w={w} mod={m}")
        if op == "+":
            r = v + w
        elif op == "-":
            r = v - w
        elif op == "*":
            r = v * w
        elif op == "/":
            if w % m == 0:
                raise ValueError("模零除法")
            r = v * pow(w, -1, m)
        else:
            raise ValueError(f"未知算符 {op!r}")
        return r % m

    if f["tier"] == "t12":
        return ap(f["x"], f["y"], f["op"])
    v1 = ap(f["x"], f["y"], f["op1"])
    return ap(v1, f["z"], f["op2"])


def grade(line):
    """判一行题。返回 {ok, reason, tier, mod, expected, given}。"""
    base = {"line": line.rstrip("\n")}
    try:
        f = parse(line)
    except Ambiguous as e:
        return dict(base, ok=False, reason="ambiguous", detail=str(e))
    except ValueError as e:
        return dict(base, ok=False, reason="unparseable", detail=str(e))
    out = dict(base, tier=f["tier"], mod=f["mod"], given=f["ans"])
    if not f["canonical"]:
        return dict(out, ok=False, reason="noncanonical")
    try:
        exp = ref_answer(f)
    except ValueError as e:
        return dict(out, ok=False, reason="ref_error", detail=str(e))
    out["expected"] = exp
    if not (0 <= f["ans"] < f["mod"]):
        return dict(out, ok=False, reason="ans_out_of_range")
    if f["ans"] != exp:
        return dict(out, ok=False, reason="ans_mismatch")
    return dict(out, ok=True, reason="pass")

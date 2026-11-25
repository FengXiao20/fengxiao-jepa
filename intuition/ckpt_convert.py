# -*- coding: utf-8 -*-
"""ckpt_convert.py —— pth ↔ json 双向转换（权重三件套的冷归档机制）

分工铁律（TA 2026-11-21 定）：pth 是唯一热格式，json 绝不进运行路径。
本脚本只在归档/救援时离线使用：
  to_json: pth → json 全量（可读、可 diff、不绑 torch 生态）
  to_pth:  json → pth，并 assert 与源 bit-exact（转回必须无损）

用法:
  python ckpt_convert.py to_json runs/stage2/resume.pth archive_resume.json
  python ckpt_convert.py to_pth  archive_resume.json rebuilt.pth --verify runs/stage2/resume.pth
"""
import argparse
import json
import sys

import torch


def to_json(pth_path, json_path):
    obj = torch.load(pth_path, map_location="cpu", weights_only=False)

    def enc(v):
        if torch.is_tensor(v):
            return {"__tensor__": True, "dtype": str(v.dtype), "shape": list(v.shape),
                    "data": v.detach().cpu().reshape(-1).tolist()}
        if isinstance(v, dict):
            return {str(k): enc(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return {"__seq__": type(v).__name__, "items": [enc(x) for x in v]}
        if isinstance(v, (int, float, str, bool)) or v is None:
            return v
        return {"__raw__": repr(v)}   # 不可序列化对象（如 rng 状态）降级为文本，可读不还原

    json.dump(enc(obj), open(json_path, "w"), separators=(",", ":"))
    print(f"[to_json] {pth_path} -> {json_path}")


def dec(v):
    if isinstance(v, dict):
        if v.get("__tensor__"):
            t = torch.tensor(v["data"], dtype=getattr(torch, v["dtype"].split(".")[-1]))
            return t.reshape(v["shape"])
        if "__seq__" in v:
            items = [dec(x) for x in v["items"]]
            return tuple(items) if v["__seq__"] == "tuple" else items
        return {k: dec(x) for k, x in v.items()}
    return v


def to_pth(json_path, pth_path, verify=None):
    obj = dec(json.load(open(json_path)))
    torch.save(obj, pth_path)
    if verify:
        src = torch.load(verify, map_location="cpu", weights_only=False)

        def cmp(a, b, path=""):
            if torch.is_tensor(a):
                assert torch.equal(a, b), f"bit-exact 失败: {path}"
            elif isinstance(a, dict):
                assert a.keys() == b.keys(), f"键不一致: {path}"
                for k in a:
                    cmp(a[k], b[k], f"{path}.{k}")
        cmp(obj, src)
        print(f"[to_pth] {json_path} -> {pth_path}  bit-exact ✓ ({verify})")
    else:
        print(f"[to_pth] {json_path} -> {pth_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["to_json", "to_pth"])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--verify", default=None, help="to_pth 时与源 pth 做 bit-exact 校验")
    a = ap.parse_args()
    try:
        if a.cmd == "to_json":
            to_json(a.src, a.dst)
        else:
            to_pth(a.src, a.dst, a.verify)
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

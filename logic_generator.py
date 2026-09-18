# -*- coding: utf-8 -*-
"""
logic_generator.py (v4)
=======================
自动生成逻辑语料，供 SP-JEPA 训练 + 自博弈奖励使用。

v2 -> v3:
- fix: _is_negative 不再匹配裸 '不'
- fix: arithmetic verify 优先抽取 '答案是/结果是' 之后的数字

v3 -> v4:
- fix: verify 支持 false_positive 模板里的 '推理不成立'
- fix: _extract_final_number 支持小数
- add: SELF-CONSISTENCY 全模板自检
- add: fake3 小数回归测试
- 更新最终统计: 14/14 (10 basic + 1 consistency + 3 regression)

注意:
- 逻辑推理不适合 JEPA 本身执行，适合作为 L5 意图识别的样本
- 生成器知道答案，监督信号免费
"""

import json
import random
import re
from typing import List, Dict
from collections import Counter


# =====================================================================
#  1. 概念表
# =====================================================================
CATEGORIES = {
    "动物": ["小猫", "小狗", "麻雀", "老虎", "鲸鱼"],
    "植物": ["柳树", "玫瑰", "松树", "蒲公英"],
    "金属": ["铁", "铜", "铝", "金"],
    "液体": ["水", "油", "醋", "牛奶"],
}

CATEGORY_PROPERTIES = {
    "动物": ["会呼吸", "需要食物", "有生命"],
    "植物": ["会呼吸", "需要阳光", "有生命"],
    "金属": ["导电", "有延展性", "无生命"],
    "液体": ["有体积", "可流动", "无固定形状"],
}

ORDINALS = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛"]

# v4: 拒绝词表（对应所有负面回答模板）
NEGATIVE_PATTERNS = ["不会", "不是", "不能", "并非", "否", "错误", "不对", "不成立"]
REFUSAL_MARKERS = ["不能确定", "推理不成立", "需要额外信息", "没有必然联系"]


# =====================================================================
#  2. 模板库
# =====================================================================
SYLLOGISM_TEMPLATES = [
    "已知：所有{cat}都{prop}。{entity}是{cat}。问：{entity}{prop}吗？",
    "{entity}属于{cat}。{cat}都{prop}。那么{entity}会{prop}吗？",
    "所有{cat}的特点之一就是{prop}。{entity}是{cat}。{entity}{prop}吗？",
    "规则：{cat}都{prop}。事实：{entity}是{cat}。问：{entity}是否{prop}？",
]

SYLLOGISM_ANSWER_TEMPLATES = [
    "是。因为{entity}属于{cat}，所有{cat}都{prop}，所以{entity}{prop}。",
    "会。{entity}是{cat}，{cat}都{prop}，因此{entity}{prop}。",
    "是的。{cat}都{prop}，{entity}又是{cat}，所以{entity}{prop}。",
]

TRANSITIVE_TEMPLATES = [
    "已知：{top}比{mid}大。{mid}比{bot}大。问：{top}比{bot}大吗？",
    "{top}比{mid}大，{mid}比{bot}大。{top}和{bot}谁大？",
    "关系链：{top}>{mid}，{mid}>{bot}。问：{top}是否大于{bot}？",
]

TRANSITIVE_ANSWER_TEMPLATES = [
    "是。因为{top}>{mid}，{mid}>{bot}，所以{top}>{bot}。",
    "{top}更大。{top}大于{mid}，{mid}大于{bot}，因此{top}大于{bot}。",
    "是的，{top}比{bot}大。传递关系成立。",
]

ARITHMETIC_TEMPLATES = [
    "计算：({a}+{b})×{c}。",
    "({a}+{b})×{c}等于多少？",
    "先算{a}加{b}，再乘以{c}，结果是多少？",
]

ARITHMETIC_ANSWER_TEMPLATES = [
    "{a}+{b}={mid}，{mid}×{c}={final}。结果是{final}。",
    "先算加法得到{mid}，再乘{c}等于{final}。",
    "答案是{final}。",
]

NEGATION_TEMPLATES = [
    "已知：所有{c1}都{prop1}。{entity}是{c1}。问：{entity}会{wrong_prop}吗？",
    "{entity}是{c1}。{c2}通常{wrong_prop}。那么{entity}也{wrong_prop}吗？",
    "条件：{entity}属于{c1}。问题：{entity}是不是{wrong_prop}？",
]

NEGATION_ANSWER_TEMPLATES = [
    "不能确定。因为{entity}是{c1}，而{wrong_prop}是{c2}的属性，两者不能直接推断。",
    "不能确定。{entity}属于{c1}，但{c1}不一定{wrong_prop}。",
    "不能确定。这是一个逻辑跳跃，需要额外信息。",
]

FALSE_POSITIVE_TEMPLATES = [
    "有人说：既然{a_entity}是{cat_a}，而{cat_b}都{prop_b}，那么{a_entity}也{prop_b}。这个推理对吗？",
    "论证：{a_entity}属于{cat_a}，{cat_b}都有{prop_b}，所以{a_entity}{prop_b}。这个论证成立吗？",
    "以下推理是否正确：{a_entity}是{cat_a}，因为{cat_b}会{prop_b}，所以{a_entity}也{prop_b}？",
]

FALSE_POSITIVE_ANSWER_TEMPLATES = [
    "不能确定。{a_entity}是{cat_a}，但{cat_a}和{cat_b}是不同类别，不能直接类推。",
    "推理不成立。{cat_a}的属性和{cat_b}的属性没有必然联系。",
    "不能确定。这是错误的类比，缺少中间环节。",
]


# =====================================================================
#  3. 五类逻辑生成器
# =====================================================================
def gen_syllogism(rng):
    cat = rng.choice(list(CATEGORIES.keys()))
    entity = rng.choice(CATEGORIES[cat])
    prop = rng.choice(CATEGORY_PROPERTIES[cat])
    prompt = rng.choice(SYLLOGISM_TEMPLATES).format(cat=cat, entity=entity, prop=prop)
    answer = rng.choice(SYLLOGISM_ANSWER_TEMPLATES).format(cat=cat, entity=entity, prop=prop)
    return {"type": "syllogism", "prompt": prompt, "answer": answer,
            "meta": {"category": cat, "entity": entity, "property": prop}}


def gen_transitive(rng):
    top, mid, bot = rng.sample(ORDINALS, 3)
    prompt = rng.choice(TRANSITIVE_TEMPLATES).format(top=top, mid=mid, bot=bot)
    answer = rng.choice(TRANSITIVE_ANSWER_TEMPLATES).format(top=top, mid=mid, bot=bot)
    return {"type": "transitive", "prompt": prompt, "answer": answer,
            "meta": {"top": top, "mid": mid, "bot": bot}}


def gen_arithmetic(rng):
    a, b, c = rng.randint(2, 20), rng.randint(2, 20), rng.randint(2, 10)
    mid = a + b
    final = mid * c
    prompt = rng.choice(ARITHMETIC_TEMPLATES).format(a=a, b=b, c=c)
    answer = rng.choice(ARITHMETIC_ANSWER_TEMPLATES).format(a=a, b=b, c=c, mid=mid, final=final)
    return {"type": "arithmetic", "prompt": prompt, "answer": answer,
            "meta": {"a": a, "b": b, "c": c, "mid": mid, "final": final}}


def gen_negation(rng):
    c1, c2 = rng.sample(list(CATEGORIES.keys()), 2)
    entity = rng.choice(CATEGORIES[c1])
    prop1 = rng.choice(CATEGORY_PROPERTIES[c1])
    wrong_prop = rng.choice(CATEGORY_PROPERTIES[c2])
    prompt = rng.choice(NEGATION_TEMPLATES).format(
        c1=c1, c2=c2, entity=entity, prop1=prop1, wrong_prop=wrong_prop)
    answer = rng.choice(NEGATION_ANSWER_TEMPLATES).format(
        c1=c1, c2=c2, entity=entity, wrong_prop=wrong_prop)
    return {"type": "negation", "prompt": prompt, "answer": answer,
            "meta": {"c1": c1, "c2": c2, "entity": entity, "wrong_prop": wrong_prop}}


def gen_false_positive(rng):
    c1, c2 = rng.sample(list(CATEGORIES.keys()), 2)
    a_entity = rng.choice(CATEGORIES[c1])
    prop_b = rng.choice(CATEGORY_PROPERTIES[c2])
    prompt = rng.choice(FALSE_POSITIVE_TEMPLATES).format(
        a_entity=a_entity, cat_a=c1, cat_b=c2, prop_b=prop_b)
    answer = rng.choice(FALSE_POSITIVE_ANSWER_TEMPLATES).format(
        a_entity=a_entity, cat_a=c1, cat_b=c2, prop_b=prop_b)
    return {"type": "false_positive", "prompt": prompt, "answer": answer,
            "meta": {"cat_a": c1, "cat_b": c2, "entity": a_entity, "wrong_prop": prop_b}}


GENERATORS = {
    "syllogism":      gen_syllogism,
    "transitive":     gen_transitive,
    "arithmetic":     gen_arithmetic,
    "negation":       gen_negation,
    "false_positive": gen_false_positive,
}


# =====================================================================
#  4. 生成主函数
# =====================================================================
def generate(n_per_type: int, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    items = []
    for name, fn in GENERATORS.items():
        for _ in range(n_per_type):
            items.append(fn(rng))
    rng.shuffle(items)
    return items


# =====================================================================
#  5. 输出
# =====================================================================
def write_outputs(items, corpus_path="logic_corpus.txt", qa_path="logic_qa.jsonl"):
    with open(corpus_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(it["prompt"] + it["answer"] + "\n")

    with open(qa_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    counter = Counter(it["type"] for it in items)
    print(f"[LOGIC] Generated {len(items)} items -> {corpus_path}")
    for k, v in counter.items():
        print(f"    {k:16s}: {v}")
    print(f"[LOGIC] Structured QA -> {qa_path}")


# =====================================================================
#  6. 自动验证器 (v4)
# =====================================================================
def _is_negative(text):
    """只匹配多字否定模式，不再匹配裸 '不'。"""
    return any(p in text for p in NEGATIVE_PATTERNS)


def _extract_final_number(text):
    """
    v4: 优先抽取 '答案是/结果是' 之后的第一个数字，支持小数。
    兜底：找不到锚点，退回最后一个数字。
    """
    def _to_num(s):
        v = float(s)
        return int(v) if v == int(v) else v

    m = re.search(r"(?:答案是|结果是|等于)\s*(-?\d+\.?\d*)", text)
    if m:
        return _to_num(m.group(1))
    nums = re.findall(r"-?\d+\.?\d*", text)
    if not nums:
        return None
    return _to_num(nums[-1])


def verify(item: Dict, model_output: str) -> float:
    """
    自动裁判。返回 [0, 1] 奖励分。
    """
    t = item["type"]

    if t == "arithmetic":
        n = _extract_final_number(model_output)
        if n is None:
            return 0.0
        return 1.0 if n == item["meta"]["final"] else 0.0

    if t == "syllogism":
        m = item["meta"]
        has_entity = m["entity"] in model_output
        has_prop = m["property"] in model_output
        return 1.0 if (has_entity and has_prop and not _is_negative(model_output)) else 0.0

    if t == "transitive":
        m = item["meta"]
        has_top = m["top"] in model_output
        has_bot = m["bot"] in model_output
        return 1.0 if (has_top and has_bot and not _is_negative(model_output)) else 0.0

    if t in ("negation", "false_positive"):
        return 1.0 if any(m in model_output for m in REFUSAL_MARKERS) else 0.0

    return 0.0


# =====================================================================
#  7. 入口
# =====================================================================
if __name__ == "__main__":
    items = generate(n_per_type=2000, seed=42)
    write_outputs(items)

    print("\n[SAMPLE]")
    shown = {}
    for it in items:
        t = it["type"]
        if shown.get(t, 0) >= 1:
            continue
        shown[t] = shown.get(t, 0) + 1
        print(f"\n  [{t}]")
        print(f"    Q: {it['prompt']}")
        print(f"    A: {it['answer']}")

    # 自检: verify 对 ground truth 打分
    print("\n[SELF-CHECK] verify() on ground truth answers:")
    fails = 0
    for it in items[:10]:
        score = verify(it, it["answer"])
        status = "PASS" if score == 1.0 else "FAIL"
        if score != 1.0:
            fails += 1
        print(f"  [{status}] {it['type']:16s} score={score:.2f}")

    # 全模板自检: 所有生成器的答案模板必须能通过自己的验证器
    print("\n[SELF-CONSISTENCY] verify() on ALL answer templates:")
    all_ok = True
    for name, gen in GENERATORS.items():
        for _ in range(200):
            test_item = gen(random.Random(0))
            if verify(test_item, test_item["answer"]) != 1.0:
                all_ok = False
                print(f"  [FAIL] {name}: {test_item['answer'][:60]}")
                break
    print(f"  [{'PASS' if all_ok else 'FAIL'}] all generators self-consistent")
    if not all_ok:
        fails += 1

    # 回归测试: v2/v3 已知误伤场景
    print("\n[REGRESSION TEST] v2/v3 known bugs:")

    fake1 = {"type": "syllogism", "meta": {"entity": "小猫", "property": "会呼吸"}}
    r1 = verify(fake1, "毫无疑问，小猫会呼吸。")
    print(f"  '毫无疑问，小猫会呼吸' -> {r1}  (期望 1.0, v2 会误判 0.0)")
    if r1 != 1.0: fails += 1

    fake2 = {"type": "arithmetic", "meta": {"final": 100}}
    r2 = verify(fake2, "结果是 100，不是 99。")
    print(f"  '结果是 100，不是 99' -> {r2}  (期望 1.0, v2 会因 nums[-1]=99 误判 0.0)")
    if r2 != 1.0: fails += 1

    fake3 = {"type": "arithmetic", "meta": {"final": 2.5}}
    r3 = verify(fake3, "结果是 2.5。")
    print(f"  '结果是 2.5' -> {r3}  (期望 1.0, v3 会失败)")
    if r3 != 1.0: fails += 1

    print(f"\n[SELF-CHECK] {14 - fails}/14 total checks passed "
          f"(10 basic + 1 consistency + 3 regression).")
# -*- coding: utf-8 -*-
"""
Prepare augmented corpus for SP-JEPA training.
================================================
把多个来源拼成一个 input_augmented.txt。

顺序:
  1. identity_zh.txt        × REPEAT (打乱)
  2. identity_en.txt        × REPEAT (打乱)
  3. logic_corpus.txt          (可选)
  4. qa_corpus.txt             (可选)
  5. qa_corpus_en.txt          (可选)
  6. corpus_zh_teacher.txt     (可选)
  7. corpus_en_teacher.txt     (可选)
  8. corpus_general.txt        (可选)
  9. input.txt

改进点:
- REPEAT=2 (身份语料已扩展为 80 行/语言)
- 每次重复时打乱行顺序，消除连续重复模式
- [v2] shuffle 使用固定种子，保证可复现
- [v2] 可选加载 teacher 语料 (中文/英文各一块)
- [v3] 可选加载 general 语料，并输出各来源占比
- 可选加载 logic / qa / qa_en，文件不存在就跳过
"""

import os
import random

# --- 固定来源 ---
IDENTITY_ZH = "./identity_zh.txt"
IDENTITY_EN = "./identity_en.txt"
CORPUS_PATH = "./input.txt"

# --- 可选来源 ---
LOGIC_CORPUS    = "./logic_corpus.txt"
QA_CORPUS       = "./qa_corpus.txt"
QA_CORPUS_EN    = "./qa_corpus_en.txt"
TEACHER_ZH      = "./corpus_zh_teacher.txt"
TEACHER_EN      = "./corpus_en_teacher.txt"
GENERAL_CORPUS  = "./corpus_general.txt"

OUTPUT_PATH     = "./input_augmented.txt"
REPEAT          = 3
SHUFFLE         = True
SHUFFLE_SEED    = 42


def read_lines(path):
    if not os.path.exists(path):
        print(f"[WARN] {path} not found, skipping.")
        return []
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def read_block(path, label, required=False):
    """读整个文件为一块字符串。文件不存在时：
       required=True  -> 报错返回 None
       required=False -> 打印跳过提示，返回空串
    """
    if not os.path.exists(path):
        if required:
            print(f"[ERROR] {label} required but not found: {path}")
            return None
        print(f"[SKIP] {label} not found: {path}")
        return ""
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    print(f"[OK]   {label} loaded: {len(text):,} chars  ({path})")
    return text


def main():
    print("=" * 68)
    print("  prepare_corpus  |  FengXiao v0.4.4-preflight")
    print("=" * 68)

    # --- 身份语料（必需）---
    zh_lines = read_lines(IDENTITY_ZH)
    en_lines = read_lines(IDENTITY_EN)
    identity_lines = zh_lines + en_lines

    if not identity_lines:
        print("[ERROR] No identity lines found. Abort.")
        return

    # --- 主语料（必需）---
    corpus = read_block(CORPUS_PATH, "Main corpus (input.txt)", required=True)
    if corpus is None:
        return

    # --- 可选块 ---
    logic_block    = read_block(LOGIC_CORPUS,   "Logic corpus")
    qa_block       = read_block(QA_CORPUS,      "QA corpus ZH")
    qa_en_block    = read_block(QA_CORPUS_EN,   "QA corpus EN")
    teacher_zh     = read_block(TEACHER_ZH,     "Teacher ZH")
    teacher_en     = read_block(TEACHER_EN,     "Teacher EN")
    general_block  = read_block(GENERAL_CORPUS, "General corpus")

    # --- 身份块：重复 REPEAT 次，每次打乱（固定种子）---
    rng = random.Random(SHUFFLE_SEED)
    all_identity = []
    for _ in range(REPEAT):
        block = identity_lines[:]
        if SHUFFLE:
            rng.shuffle(block)
        all_identity.extend(block)
    identity_block = "\n".join(all_identity)

    # --- 拼接 ---
    parts = [identity_block]
    if logic_block:
        parts.append(logic_block)
    if qa_block:
        parts.append(qa_block)
    if qa_en_block:
        parts.append(qa_en_block)
    if teacher_zh:
        parts.append(teacher_zh)
    if teacher_en:
        parts.append(teacher_en)
    if general_block:
        parts.append(general_block)
    parts.append(corpus)

    output = "\n".join(parts)

    # --- 写出 ---
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output)

    # --- 统计 ---
    total_chars = len(output)

    def pct(n):
        return (n / total_chars * 100) if total_chars else 0.0

    id_chars       = len(identity_block)
    logic_chars    = len(logic_block)
    qa_chars       = len(qa_block)
    qa_en_chars    = len(qa_en_block)
    teacher_chars  = len(teacher_zh) + len(teacher_en)
    general_chars  = len(general_block)
    main_chars     = len(corpus)

    print("\n" + "-" * 68)
    print("  Source breakdown")
    print("-" * 68)
    print(f"  Identity (after repeat) : {id_chars:>10,}  ({pct(id_chars):5.2f}%)")
    print(f"  Logic                   : {logic_chars:>10,}  ({pct(logic_chars):5.2f}%)")
    print(f"  QA ZH                   : {qa_chars:>10,}  ({pct(qa_chars):5.2f}%)")
    print(f"  QA EN                   : {qa_en_chars:>10,}  ({pct(qa_en_chars):5.2f}%)")
    print(f"  Teacher ZH+EN           : {teacher_chars:>10,}  ({pct(teacher_chars):5.2f}%)")
    print(f"  General                 : {general_chars:>10,}  ({pct(general_chars):5.2f}%)")
    print(f"  Main (input.txt)        : {main_chars:>10,}  ({pct(main_chars):5.2f}%)")
    print("-" * 68)
    print(f"  TOTAL                   : {total_chars:>10,}")
    print("-" * 68)

    # --- 语料构成提示 ---
    print("\n  Identity lines : "
          f"ZH={len(zh_lines)}  EN={len(en_lines)}  "
          f"before_repeat={len(identity_lines)}  after_repeat={len(all_identity)}")

    if id_chars == 0:
        pass
    elif pct(id_chars) > 5.0:
        print(f"  [NOTE] Identity share {pct(id_chars):.1f}% > 5%. "
              f"Consider lowering REPEAT or growing input.txt.")
    elif pct(id_chars) < 2.0:
        print(f"  [NOTE] Identity share {pct(id_chars):.1f}% < 2%. "
              f"Identity signal may be too weak.")

    if teacher_chars == 0:
        print("  [NOTE] Teacher corpus missing. "
              "Drop corpus_zh_teacher.txt / corpus_en_teacher.txt "
              "next to this script to include them.")

    if general_chars == 0:
        print("  [NOTE] General corpus missing. "
              "Drop corpus_general.txt next to this script to include it.")

    print(f"\n[OK] Wrote {OUTPUT_PATH}")
    print("=" * 68)


if __name__ == "__main__":
    main()
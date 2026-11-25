# -*- coding: utf-8 -*-
"""fengxiao-intuition · 阶段2 —— 训练主循环

设计要点（对应任务规格）:
  - LM 预训练为主任务；FOK/confidence 双头由 rollout 判卷结果免费回填标签
  - 旧 JEPA/sep 探针只读观测，训练权重为零
  - 固定 seed、断点续跑（resume.pth 全状态）、每 epoch 固定产物:
      checkpoint 快照 / log.jsonl / ledger.csv / calibration.png / sep 读数
  - CPU 友好：无 AMP，批 16×128；长跑用 setsid 脱离 + monitor.py 挂机

用法:
  python train_stage2.py --config stage2.json            # 断点续跑
  python train_stage2.py --config stage2.json fresh      # 强制重训
"""
import argparse
import csv
import importlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import corpus as corpus_mod
import judge
import model as model_mod
from model import IntuitionLM


# ---------------- 基础设施 ----------------

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_lr(step, total, warmup, base, mn):
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return mn + (base - mn) * 0.5 * (1 + math.cos(math.pi * prog))


def build_optimizer(model, base_lr, wd):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.ndim <= 1 or "tok_embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return AdamW([{"params": decay, "weight_decay": wd},
                  {"params": no_decay, "weight_decay": 0.0}],
                 lr=base_lr, betas=(0.9, 0.95))


def seg_for_epoch(epoch, curriculum):
    for rule in curriculum:
        if epoch <= rule["until_epoch"]:
            return rule["seg"]
    return curriculum[-1]["seg"]


# ---------------- rollout：作答 + 判卷 + 双头标签 ----------------

@torch.no_grad()
def rollout_batch(model, corpus, prompt_items, max_new, device):
    """贪心作答一批题。返回逐题 dict: prompt/gen/ids/位置/熵。"""
    model.eval()
    B = len(prompt_items)
    plens = [len(it["prompt_ids"]) for it in prompt_items]
    T0 = max(plens)
    x = torch.zeros(B, T0, dtype=torch.long)
    for i, it in enumerate(prompt_items):
        ids = it["prompt_ids"]
        x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    seqs = [list(it["prompt_ids"]) for it in prompt_items]
    entropies = [[] for _ in range(B)]
    gen_chars = [[] for _ in range(B)]
    finished = [False] * B

    for _ in range(max_new):
        T = max(len(s) for s in seqs)
        xb = torch.zeros(B, T, dtype=torch.long)
        for i, s in enumerate(seqs):
            xb[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        logits, _, _ = model(xb)
        for i in range(B):
            if finished[i]:
                continue
            li = logits[i, len(seqs[i]) - 1]
            p = F.softmax(li, dim=-1)
            ent = float(-(p * (p + 1e-12).log()).sum())
            nxt = int(torch.argmax(li))
            entropies[i].append(ent)
            seqs[i].append(nxt)
            ch = corpus.itos[nxt]
            gen_chars[i].append(ch)
            if ch == "\n" or len(gen_chars[i]) >= max_new - 1:
                finished[i] = True
        if all(finished):
            break
    model.train()

    outs = []
    for i, it in enumerate(prompt_items):
        gen_text = "".join(gen_chars[i])
        # conf 读取位置：答案末字符（若以 \n 收尾，取其前一个）
        last_ans_pos = len(seqs[i]) - 2 if (gen_text.endswith("\n")) else len(seqs[i]) - 1
        outs.append({
            "item": it, "gen_text": gen_text,
            "full_ids": seqs[i], "prompt_len": plens[i],
            "last_ans_pos": max(last_ans_pos, plens[i] - 1),
            "entropy": float(np.mean(entropies[i])) if entropies[i] else 0.0,
        })
    return outs


def head_losses(model, rollouts, device):
    """FOK 读题面末 token，CONF 读答案末 token；标签 = 判卷对错（免费回填）。"""
    B = len(rollouts)
    T = max(len(r["full_ids"]) for r in rollouts)
    x = torch.zeros(B, T, dtype=torch.long)
    for i, r in enumerate(rollouts):
        x[i, :len(r["full_ids"])] = torch.tensor(r["full_ids"], dtype=torch.long)
    h = model.features(x.to(device))

    rows = torch.arange(B, device=device)
    qpos = torch.tensor([r["prompt_len"] - 1 for r in rollouts], device=device)
    apos = torch.tensor([min(r["last_ans_pos"], T - 1) for r in rollouts], device=device)
    labels = torch.tensor([1.0 if r["correct"] else 0.0 for r in rollouts], device=device)

    fok_logit = model.fok_head(h[rows, qpos]).squeeze(-1)
    conf_logit = model.conf_head(h[rows, apos]).squeeze(-1)
    fok_loss = F.binary_cross_entropy_with_logits(fok_logit, labels)
    conf_loss = F.binary_cross_entropy_with_logits(conf_logit, labels)
    with torch.no_grad():
        fok_p = torch.sigmoid(fok_logit).cpu().tolist()
        conf_p = torch.sigmoid(conf_logit).cpu().tolist()
    return fok_loss, conf_loss, fok_p, conf_p


def do_rollout(model, corpus, cfg, rng, device, seg=None, eval_sets=None, per_override=None):
    """seg 给定 → 训练题 rollout（回填标签）；eval_sets 给定 → 评测 rollout（落账本）。"""
    if eval_sets is None:
        idxs = corpus.sample_train_problems(cfg["rollout"]["batch"], rng, cfg["rollout"]["mix"])
        items = [{"source": "train",
                  "prompt_ids": corpus.problem_prompt_ids(i),
                  "prompt_text": corpus.problem_prompt_text(i),
                  "text": None, "ids": None} for i in idxs]
    else:
        items = []
        per = per_override or max(1, cfg["eval_rollout"]["per_set"] // len(eval_sets))
        for name in eval_sets:
            for it in corpus.eval_prompt_items(name, per, rng):
                it = dict(it, source=name)
                items.append(it)

    ro = rollout_batch(model, corpus, items, cfg["rollout"]["max_new"], device)
    for r in ro:
        line = r["item"]["prompt_text"] + r["gen_text"].rstrip("\n")
        g = judge.grade(line if line.endswith("\n") else line + "\n")
        if r["gen_text"].rstrip("\n") == "":
            g = {"ok": False, "reason": "incomplete", "expected": g.get("expected")}
        r["judge"] = g
        r["correct"] = bool(g["ok"])
    return ro


def append_ledger(run_dir, rollouts, split):
    """账本：题面特征 / 熵 / conf / fok / 死因枚举，全部机器字段，不用自然语言。"""
    path = os.path.join(run_dir, "ledger.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["split", "prompt", "gen", "correct", "reason",
                        "entropy", "conf", "fok", "expected"])
        for r in rollouts:
            w.writerow([split, r["item"]["prompt_text"], r["gen_text"].rstrip("\n"),
                        int(r["correct"]), r["judge"]["reason"],
                        f"{r['entropy']:.4f}", f"{r.get('conf', -1):.4f}",
                        f"{r.get('fok', -1):.4f}", r["judge"].get("expected", "")])


# ---------------- probe 数据准备 ----------------

def probe_inputs(corpus, cfg, rng):
    hit = corpus.eval_prompt_items(cfg["probe"].get("hit_set", "id_mod97"),
                                   cfg["probe"]["n_hit"], rng)
    miss = []
    for name in ("poison_mod13", "poison_mod61"):
        miss += corpus.eval_prompt_items(name, cfg["probe"]["n_miss_per_set"], rng)
    hit_ids = [it["prompt_ids"] for it in hit]
    miss_ids = [it["prompt_ids"] for it in miss]
    qh = [len(s) - 1 for s in hit_ids]
    qm = [len(s) - 1 for s in miss_ids]
    return hit_ids, miss_ids, qh, qm


# ---------------- 训练 ----------------

def select_judge(cfg):
    """judge 旋钮：trace 语料用 judge_trace（独立判卷器，接口同 judge.grade）。"""
    global judge
    judge = importlib.import_module(cfg.get("judge_mod", "judge"))
    return judge


def train(cfg_path, fresh=False):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    seed = int(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    run_dir = os.path.join(HERE, cfg["paths"]["run_dir"])
    os.makedirs(run_dir, exist_ok=True)
    data_dir = os.path.join(HERE, cfg["paths"].get("data_dir", "data"))

    select_judge(cfg)

    corpus = corpus_mod.Corpus(data_dir, seed, cfg["train"]["val_windows_pct"])
    model = IntuitionLM(vocab_size=cfg["vocab_target"], **cfg["model"]).to(device)
    n_params = model.num_params()

    bs, sl = cfg["train"]["batch_size"], cfg["seq_len"]
    steps_per_epoch = cfg["train"]["tokens_per_epoch"] // (bs * sl)
    total_steps = steps_per_epoch * cfg["train"]["epochs"]
    opt = build_optimizer(model, cfg["train"]["lr"], cfg["train"]["weight_decay"])

    # resume（全状态；cosine 按 global_step 连续）
    history, start_epoch, global_step = {"epochs": []}, 1, 0
    resume_path = os.path.join(run_dir, "resume.pth")
    if not fresh and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer_state"])
        start_epoch, global_step = ck["epoch"] + 1, ck["global_step"]
        history = ck.get("history", history)
        print(f"[RESUME] epoch {start_epoch} / step {global_step}")
    if start_epoch > cfg["train"]["epochs"]:
        print(f"[TRAIN] 已完成 {start_epoch - 1} epochs，无事可做（fresh 强制重训）")
        return

    log_path = os.path.join(run_dir, "log.jsonl")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    def logj(event, **kw):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": event, **kw}, ensure_ascii=False) + "\n")
        print(json.dumps({"event": event, **kw}, ensure_ascii=False), flush=True)

    logj("train_start", params=f"{n_params/1e6:.2f}M", steps_per_epoch=steps_per_epoch,
         total_steps=total_steps, device=str(device))

    t0 = time.time()
    lam_fok, lam_conf = cfg["heads"]["lam_fok"], cfg["heads"]["lam_conf"]

    # 哨兵批：固定窗口，专供激活范数读数（R3 红旗），与训练采样无关
    sentinel_x, _ = corpus.sample_val_windows(bs, np.random.default_rng([seed, 0x53EED]))

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        seg = seg_for_epoch(epoch, cfg["train"]["curriculum"])
        epoch_rng = np.random.default_rng([seed, 0xE90C, epoch])  # per-epoch 确定性，resume 不漂移
        model.train()
        ep_losses, ep_lm, ep_fok, ep_conf = [], [], [], []

        for step in range(steps_per_epoch):
            lr = get_lr(global_step, total_steps, cfg["train"]["warmup_steps"],
                        cfg["train"]["lr"], cfg["train"]["min_lr"])
            for g in opt.param_groups:
                g["lr"] = lr

            # curriculum + replay：25% 从低段回放防遗忘
            if seg == "t1" or epoch_rng.random() >= cfg["train"]["replay_ratio"]:
                use_seg = seg
            else:
                use_seg = "t1" if epoch_rng.random() < 0.5 else \
                    ("t2" if seg == "t3" else "t1")
            x, y = corpus.sample_windows(use_seg, bs, epoch_rng)

            _, lm_loss, _ = model(x, y)
            loss = lm_loss
            fok_l = conf_l = torch.tensor(0.0)

            # rollout 回填双头标签（免费：判卷结果就是标签）
            if global_step % cfg["rollout"]["every"] == 0:
                ros = do_rollout(model, corpus, cfg, epoch_rng, device)
                fok_l, conf_l, fok_p, conf_p = head_losses(model, ros, device)
                loss = lm_loss + lam_fok * fok_l + lam_conf * conf_l
                for r, fp, cp in zip(ros, fok_p, conf_p):
                    r["fok"], r["conf"] = fp, cp
                append_ledger(run_dir, ros, split=f"train_e{epoch}_s{global_step}")
                ep_fok.append(float(fok_l.detach()))
                ep_conf.append(float(conf_l.detach()))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["max_grad_norm"])
            opt.step()

            ep_losses.append(float(loss.detach()))
            ep_lm.append(float(lm_loss.detach()))
            global_step += 1

            if global_step % cfg["train"]["log_interval"] == 0:
                print(f"  [E{epoch} S{global_step}/{total_steps}] loss={float(loss):.4f} "
                      f"lm={float(lm_loss):.4f} lr={lr:.2e} | {time.time()-t0:.0f}s", flush=True)
            if global_step % cfg["train"]["metrics_interval"] == 0:
                with torch.no_grad():
                    act_norm = float(model.features(sentinel_x).norm(dim=-1).mean())
                write_metrics(metrics_path, model, global_step, epoch,
                              float(loss), lr, act_norm)

        # ---------------- epoch 产物 ----------------
        val_loss = evaluate_val(model, corpus, cfg, epoch_rng)
        hit_ids, miss_ids, qh, qm = probe_inputs(corpus, cfg, epoch_rng)
        sep = model_mod.sep_probe(model, hit_ids, miss_ids, qh, qm, device)

        # 评测 rollout → 账本 + 头读数
        ros = do_rollout(model, corpus, cfg, epoch_rng, device,
                         eval_sets=cfg["eval_rollout"]["sets"])
        fok_l, conf_l, fok_p, conf_p = head_losses(model, ros, device)
        for r, fp, cp in zip(ros, fok_p, conf_p):
            r["fok"], r["conf"] = fp, cp
        append_ledger(run_dir, ros, split=f"eval_e{epoch}")
        corr = correlations(ros)

        rec = {"epoch": epoch, "seg": seg, "train_loss": float(np.mean(ep_losses)),
               "lm_loss": float(np.mean(ep_lm)),
               "fok_loss": float(np.mean(ep_fok)) if ep_fok else None,
               "conf_loss": float(np.mean(ep_conf)) if ep_conf else None,
               "val_loss": val_loss, "sep": sep,
               "eval_corr": corr, "lr": lr, "global_step": global_step,
               "elapsed_s": round(time.time() - t0, 1)}
        history["epochs"].append(rec)
        logj("epoch_done", **rec)

        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(), "global_step": global_step,
                    "history": history, "config": cfg},
                   os.path.join(run_dir, f"epoch_{epoch:02d}.pth"))
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(), "global_step": global_step,
                    "history": history}, resume_path)
        try:
            plot_calibration(run_dir, ros, epoch)
        except Exception as e:
            logj("calibration_skipped", error=str(e))

    logj("train_done", best_epochs=len(history["epochs"]))


def evaluate_val(model, corpus, cfg, rng):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(cfg["train"]["val_batches"]):
            x, y = corpus.sample_val_windows(cfg["train"]["batch_size"], rng)
            _, l, _ = model(x, y)
            losses.append(float(l))
    model.train()
    return float(np.mean(losses))


def write_metrics(path, model, step, epoch, loss, lr, act_norm):
    """每 N 步逐层 weight std/mean 摘要（不存全量权重），monitor.py 消费。
    act_norm = 哨兵批上 hidden 范数均值（固定批，跨步可比）。"""
    emb_std = float(model.tok_embed.weight.detach().std())
    blk_stds = [float(p.detach().std()) for n, p in model.named_parameters()
                if n.startswith("blocks") and p.ndim >= 2]
    heads = model.head_param_stds()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"step": step, "epoch": epoch, "loss": loss, "lr": lr,
                            "emb_std": emb_std,
                            "blk_std_mean": float(np.mean(blk_stds)),
                            "fok_std": heads.get("fok_head.0.weight"),
                            "conf_std": heads.get("conf_head.0.weight"),
                            "act_norm": act_norm}) + "\n")


def correlations(rollouts):
    """验收口径：conf / fok 与实际对错的相关（点二列 = Pearson）。"""

    def pearson(xs, ys):
        x, y = np.asarray(xs, float), np.asarray(ys, float)
        if x.std() < 1e-9 or y.std() < 1e-9:
            return 0.0
        return float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std()))

    cor = [1.0 if r["correct"] else 0.0 for r in rollouts]
    return {"n": len(rollouts),
            "accuracy": float(np.mean(cor)),
            "conf_corr": pearson([r["conf"] for r in rollouts], cor),
            "fok_corr": pearson([r["fok"] for r in rollouts], cor)}


def plot_calibration(run_dir, rollouts, epoch, tag=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, title in ((axes[0], "conf", "confidence head"),
                           (axes[1], "fok", "FOK head")):
        ps = np.array([r[key] for r in rollouts])
        cs = np.array([1 if r["correct"] else 0 for r in rollouts])
        bins = np.linspace(0, 1, 6)
        idx = np.digitize(ps, bins) - 1
        xs, ys = [], []
        for b in range(5):
            m = idx == b
            if m.sum() >= 3:
                xs.append(float(ps[m].mean()))
                ys.append(float(cs[m].mean()))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.plot(xs, ys, "o-")
        ax.set_title(title)
        ax.set_xlabel("predicted P(correct)")
        ax.set_ylabel("empirical accuracy")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle(f"calibration · epoch {epoch}" + (f" · {tag}" if tag else ""))
    fig.tight_layout()
    name = f"calibration_{tag}" if tag else f"calibration_e{epoch:02d}"
    fig.savefig(os.path.join(run_dir, f"{name}.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "stage2.json"))
    ap.add_argument("mode", nargs="?", default="", choices=["", "fresh"])
    a = ap.parse_args()
    try:
        train(a.config, fresh=(a.mode == "fresh"))
    except KeyboardInterrupt:
        print("[STOPPED] 进度已存 resume.pth，重跑即续")

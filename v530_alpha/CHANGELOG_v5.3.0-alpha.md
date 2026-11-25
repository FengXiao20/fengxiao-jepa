# FengXiao Changelog — v5.3.0-alpha（对比学习版）

**日期**: 2026-11-18
**性质**: 探索实验——白化排除实验（`CHANGELOG_rescue_exp_2026-11-18.md`）判定"地图真空白"后，
第一次把**对比学习**加进训练目标（诊断文档的"特殊监督算法"）
**产物**: `sp_jepa_v530.py`（基于 sp_jepa.py v5.2.0 cp+改，原文件零改动）

---

## 改了什么

| # | 改动 | 位置 |
|---|---|---|
| 1 | `__version__` → 5.3.0 | L95 |
| 2 | CONFIG: data_path→input_v530.txt, save_dir→checkpoints_v530, +8 个对比配置键, epochs 30→12(alpha 先看 sep 趋势) | CONFIG 块 |
| 3 | `make_enc(dropout)`: context_encoder 0.1(SimCSE 正例来源), target 0.0(检索侧无噪声) | SP_JEPA.__init__ |
| 4 | 新增模块级 `load_para_pool` / `contrast_forward` / `sep_probe`(独立 RNG 防训练流污染) | train() 前 |
| 5 | train loop: 对比项独立 backward 梯度累加(amp/非 amp 两分支), tr_contrast 统计+打印 | step 循环 |
| 6 | 每 5 epochs sep probe(target_encoder, calibrate 同款查询组), 按 sep 存 `fengxiao_best_sep.pth`; resume 含 best_sep | epoch 循环 |
| 7 | history 加 contrast/sep 两键 | history dict |

## 对比项设计

- **正例×2**: paraphrase 对(TSV 145 对监督) + SimCSE 式同句两次前向(dropout 0.1)
- **负例**: batch 内非配对句(8 对=16 句 → 14 负例/anchor)
- **InfoNCE**: τ=0.05, λ=0.5, `-log(e(pos_sim)+e(pos_para)) / Σall`
- **检索形态**: 无 mask 前向 → 有效位置 mean pool → normalize（与 `_encode` 精确同构）

## 语料

| 文件 | 内容 |
|---|---|
| `identity_zh_paraphrase_v530.tsv` | 145 对 paraphrase(教师=苏木, 零 OOV, check_corpus 质检通过) |
| `input_v530.txt` | 159K 原语料 + 节气教师语料(首次并入, 46 行) + 英文教师语料 + 145 对×2 行; 173,199 字符, vocab 2214→2264 |

## 训练环境备注

- 沙箱 torch 丢失→重装 2.15.0+cpu; 单步 89ms 微基准
- 网关 ~120s 硬超时 + 进程回收 → **setsid 脱离会话**后台训练 + 短轮询
- CPU 节流: epoch 耗时 61.6s → 375s 递增

## 结果（12 epochs 完成，CPU 累计 ~2h）

| 指标 | v5.1.1 基线 | v5.3.0-alpha (best_sep=E5) | 趋势 |
|---|---|---|---|
| sep | -0.0532 | **-0.2873** | E5→E10→E12: -0.2873→-0.3705→**-0.4250 单调恶化** |
| min(HIT) | 0.9011 | 0.6457 | 持续下降(0.65→0.56) |
| max(MISS) | 0.9542 | 0.9329 | 不降反升(0.93→0.98) |
| identity EffRank | 7.53/128 | **39.05/128** (zh 47.6 / en 12.4) | 几何撑开 5 倍 |
| span 库 EffRank(raw) | 31.91 | ~31(变化不大) | — |

**判定：sep 未转正，alpha 失败——但拿到干净的机制归因。**

## 失败机制（三点，按证据强度排序）

1. **负例设计缺口（设计错误，我实现时漏了半边）**：诊断文档原设计
   "MISS = easy negatives"，实现时负例只用了 batch 内随机（全是 identity
   域内句），域外句从未被教"该远"→ max(MISS) 纹丝不动(0.93→0.98)。
2. **HIT 掉的机制**：batch 内负例 = 互相推开的 identity/paraphrase 句——
   正是 HIT 查询需要命中的索引句族。推开负例 = 稀释查询-索引匹配
   → min(HIT) 0.65→0.56 单调掉。
3. **监督有效性确认（正结果）**：paraphrase 对撑开几何 5 倍
   （EffRank 7.5→39，且 zh 47.6 >> en 12.4，正例只给了中文）——
   监督信号方向对，剂量与搭配错。

## 下一步候选（视 epoch 10/12 结果）

→ 已按结果定案 **v5.3.1 方案**（见下节）

## v5.3.1 方案（修复设计缺口）

1. **MISS easy negatives 进池**：域外句（"今天天气如何"类，可从
   corpus 教师语料采样+构造）作为显式负例，与全部 identity 句互斥
   ——直接教拒答维度，攻 max(MISS)=0.93 这个不动点
2. **同句族多正例**：原句+改写+同族 query 归为正例集合（多正例
   InfoNCE），防止"推开负例伤 HIT"
3. **降剂量**：λ 0.5→0.15，τ 0.05→0.1（软温度），contrast_pairs 8→16
4. hard negatives：语义近邻 span（检索 top-k 邻居动态挖）作后续升级

## 中间观察（Epoch 1-11）

- Ctr loss: 0.0023 → 0.0006(快速下降后平台)
- **EffRank 卡在 2.1/128 不涨**: 对比项疑似在低维锥内重排而非撑开几何
- **Epoch 5 sep probe: -0.2873**(比基线更负): HIT 0.6457 / MISS 0.9329
- val 4.47 → 缓升(主任务受干扰)
- 初步假设: 145 对×14 负例信号量不足以破各向异性; InfoNCE 可在 2 维锥内局部满足

## 下一步候选（视 epoch 10/12 结果）

1. sep 转正 → 扩 pairs + 长训练 + 完整 calibrate
2. sep 不转正 → v5.3.1 调参实验: τ 0.05→0.1, λ 0.5→0.2, contrast_pairs 8→24
3. 对比学习与 whitening 不冲突(SimCSE+whitening 是文献标准组合) → 检索侧白化可叠加
4. 负例升级: hard negatives(语义近邻) 替代 batch 内随机负例

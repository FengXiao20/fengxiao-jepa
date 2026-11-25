# 阶段3 设计稿：毒题考核 + 账本（θ 扫描）

> 2026-11-21 预写，等 α/β 容量对结果出来后按赢家配置实施。
> GPU 到位后本设计零改动可跑（device 已自适应），rollout 推理是 GPU 提速最大的环节。

## 一、拒答的两个时机（直觉层的两种用法）

| 模式 | 信号 | 时机 | 语义 |
|---|---|---|---|
| **pre** | FOK 头（题面末 token） | 作答**前** | "这题我直觉不会" → 直接弃答，省算力 |
| **post** | confidence 头（答案末 token） | 作答**后** | "这答案我不自信" → 答了但标记存疑/回收 |

阶段3 两种都扫。pre 模式是外挂 LLM 场景的主打（不浪费 LLM 调用），
post 模式是选择性预测的文献标准做法（risk-coverage 曲线）。

## 二、扫描矩阵

- θ ∈ {0.1, 0.2, ..., 0.9}（9 档）
- 模式 ∈ {pre, post}（2）
- 考卷 ∈ {t1_mod7, id_mod97, poison_mod13, poison_mod61}（4，各 100 题，固定 seed 抽样）
- 模型 ∈ {α 2.574M, β 7.33M}（容量对直接接上）

实现：`stage3_sweep.py --config stage3.json --ckpt runs/stage2_beta/resume.pth`
- 每档 rollout 一次，判定逻辑纯后处理（一次 rollout 全档复用，θ 扫描不重复推理——
  pre 模式例外：需要按 fok 先筛再生成，但可用同一批 fok 读数离线模拟）
- 输出：`sweep_curves.png`（三条曲线 × 2 模式 × 2 模型）、`sweep_report.json`、
  `ledger_sweep.csv`（θ, mode, source, rejected, correct, reason, fok, conf, entropy）

## 三、三条曲线（验收口径）

1. **拒答率** = rejected / n，对 θ 单调升
2. **答错率** = 错题数 / 未拒题数（答错率显著下降 = 直觉头有分辨力）
3. **计算开销** = 实际作答题数 / 总题数（pre 模式下 = 1 - 拒答率，post 模式恒 1）

**验收**：存在 θ 使答错率显著下降（≥ 2× 相对下降）且正确题误拒率 < 30%。

## 四、附加读数（免费，rollout 一次全有）

- **AUC / risk-coverage**：按 conf 降序逐步覆盖，画 error vs coverage——
  selective prediction 标准图，比单点 θ 更能说明头的质量
- **分考卷死因构成**：rejected 里 fok 低的题，其 judge 死因分布
  （ans_mismatch vs incomplete vs out_of_range）——直觉头"怕的错"和实际错的错型是否对齐
- **熵-对错相关性**：作答熵与 correct 的相关，作 fok/conf 之外的第三信号源

## 五、与阶段4 的接口

阶段4 消融（有/无 FOK 头）只需：同 seed 重训一版 λ_fok=0 的模型（其余全同），
复用本 sweep 脚本跑对比——阶段3 的脚本期4 直接复用，不新写。

## 六、待卡到位后的增量

- rollout 批量推理换更大 batch（GPU 上 256 起步），扫描时间缩 10×+
- 可扩毒题集（重造 mod29/47 等新毒题，几分钟的事，gen.py 直接出）
- α/β checkpoint 迁移上卡 = 验证 pth 跨设备性（zip 容器 + weights_only 加载）

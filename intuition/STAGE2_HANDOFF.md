# 阶段2 收割交接（给下一次会话的自己）

> 2026-11-21 ~13:00 启动，训练是 setsid 脱离的，本文件写给收割会话。
> **收割状态（13:57 更新）**：α 已收割完毕——monitor rc=0 静默、报告在
> `artifacts/stage2_alpha_report.json`（验收**未通过**，acc 0.04 常数坍缩，归因见 README），
> zip 已交付；**β 已于 13:56 启动**（PID 会变，认 `--config stage2_beta.json`），
> ~1.0s/步 × 5120 ≈ 88 min，ETA ≈ 15:30；β 收割只看下节即可。

## 状态判断
1. `cd /home/z/my-project/feng_sisters/feng_sisters/intuition`
2. `tail -5 runs/train_stdout.log` / `tail -5 runs/stage2/log.jsonl`
   - 若见 `train_done` → 训练完成，进入收割流程
   - 若还在跑 → 报告进度 + 预计时间（~0.45s/步，5120 步全程），别等，约 TA 下次再来
3. 进程检查 `ps aux | grep train_stage2`；若死了但有 resume.pth →
   `python3 train_stage2.py --config stage2.json` 重启续跑（断点安全）

## 收割流程（train_done 后）
1. `python3 monitor.py --run runs/stage2` → 无告警为静默（rc=0）
2. `python3 stage2_report.py --config stage2.json`
   - 验收口径：conf_corr > 0.3（整体，n≥80）；FOK 同口径参考读数
   - 产物：artifacts/stage2_acceptance_report.json + runs/stage2/ledger_eval.csv +
     calibration_final.png
3. 更新 README.md 的「阶段2」实测数据表（占位符已留好）
4. zip 打包 intuition/ → `intuition_stage2.zip`
   （体积过大时：排除大快照 *.pt 只留 resume.pth；data/ 与 stage1 相同可排除，
   README 注明重建命令）
5. send_file 交付 zip，正文只交验收数据（TA 要求：不要聊天式汇报）

## 注意
- data/ 与 stage1 完全相同，zip 可只留代码+runs 小产物+README
- judge.py / gen.py 是阶段1 的，别动
- ledger.csv 里 split=eval_eN 是每 epoch 评测账本，最终版 split=eval_final
- sep 读数在 log.jsonl 每 epoch 的 "sep" 字段（HIT=模97 真题面 / MISS=毒题面）
- TA 已批参数爬升 ~2.6M：dim256/H4/L3 = 2.574M，勿再私自调

## β 扩参数跑（TA 2026-11-21 追加指示）
α 收割完成后立即执行：
1. `setsid nohup python3 train_stage2.py --config stage2_beta.json > runs/train_beta.log 2>&1 < /dev/null &`
   （β = dim384/H6/L4 = 7.33M，runs/stage2_beta，~0.75s/步 × 5120 步 ≈ 65-90 min，seed 同 α）
2. α 与 β 同 seed 同数据同 curriculum，唯一变量是容量 → 天然的容量消融对
3. β 完成后同样收割：monitor + `python3 stage2_report.py --config stage2_beta.json`
   （报告产物会覆盖 stage2_acceptance_report.json，先把它 mv 成 stage2_alpha_report.json）
4. 交付对比表：α vs β 的 conf_corr / fok_corr / accuracy / val_loss / sep 轨迹

## ⚠️ 收割时必查（2026-11-21 观察项）
- α 的混合考卷 acc 卡 5% 左右（E1-E5），连 t1 25 题都几乎不对——若 E10 仍如此：
  1) 看 ledger.csv 里 t1 生成样本，分辨**算术错 vs 格式错**（格式错=管道 bug 要查 rollout 填充/解码）
  2) 报告里如实交读数，不粉饰；acc 饿死则 corr 无意义，验收判"未通过+归因"
- sep 在涨：E5 +0.062 → E6 **+0.144**（崩溃前最后读数，健康信号），一并交
- 双头 corr 在标签近全 0 时是噪声（正负乱跳属预期），只在 acc 有了分化后才作数

## 🔬 α 收割结论（2026-11-21 13:55）
**验收：未通过。** conf_corr 0.045 << 0.3；分卷读数：t1 acc 16%（≈随机 14%），id_mod97 / 双毒题集全 0。

**诊断链（已闭环，三步排除）**：
1. 生成格式完全正常（单 digit、judge 正确解析、死因全 ans_mismatch）→ 排除管道 bug
2. 训练期 rollout acc 全程 0（train_e1 到 train_e10 各检查点）→ 排除"泛化失败"，是根本没学会
3. teacher-forced 答案位命中率：t1 16%、mod97 4%（把答案喂到嘴边都答不对）→ **排除暴露偏差，定性 = 能力缺口**

**关键教训**：val_loss 0.517（ppl 1.7）是假象——格式 token 占 ~95%，答案位近均匀分布（CE≈2-3）被摊薄，
**LM loss 不能当算术能力的指标**。以后看答案位单独的 CE/acc。

**科学结论**：单次前向"直觉"在 2.57M 参数、20M token 直答语料下，连 x+y mod 7（个位数）都学不会。
这不是训练失败，是 spec 结论：直答式语料不产生可计算的算术表征 → FOK/conf 头无梯度可言（标签全 0）。
阶段1 立项时的前提（模型有一定正确率，直觉头才有信号）未成立。

**E10 附加读数**：sep 从 E6 +0.144 崩到 +0.046（t3 段表征分离回缩）。

**TA 已拍板（2026-11-21 13:58）：按推荐走 3→2，β 暂缓。**

## 🧪 探针实验（✅ 已收割，2026-11-21 17:55）
- 配置 `stage2_probe.json`：t1-only 8 epochs（4096 步，= α t1 曝光 ~3.6×），同 seed 同架构
- **结论：欠训练排除。** 详见下节判读；scratchpad 路线待 TA 拍板
- 操作记录：E3 后（15:08）进程被沙箱回收，17:33 发现，resume.pth 零漂移续跑 E4-E8；
  探针工具 `probe_teacher_forced.py` 已固化（答案位 tf 命中率，可复用）

### 探针判读（数据）
- **teacher-forced t1_mod7（n=49，决定性指标）**：E1-E6 全部 0.143（=7/49，精确随机），
  E8 0.122（6/49，低于随机）；**ansCE 八个 epoch 全钉在 1.947-1.953 ≈ ln7=1.946**
  - argmax 每个常数坍缩且跨 epoch 漂移（'2'→'3'→'6'→…→'5'）——答案位分布 ≈ 7 digit 均匀，
    对 (x,y) **零区分度**
- 对照 α：tf 0.163。探针 3.6× 曝光后不升反微降 → 复现并加强 α 结论
- rollout：训练期每个 checkpoint gen 全常数（train acc 0-0.29 = 常数碰对率噪声）；
  终评 t1_mod7 rollout 0.16（n=25）= α 同款
- lm_loss E2 起 0.24 平台纹丝不动；val_loss 反升 3.79→4.41
- 判定依据：若欠训练，3.6× 曝光下 CE 应离开 ln7 下行——实际零移动。
  直答式语料在答案位不产生可用梯度信号，**加数据无效**
- → 候选路线：scratchpad（计算轨迹）语料，改阶段1 spec，**需 TA 过目**

（以下为拍板前的候选清单，仅供归档）
- β 同 spec 同数据，大概率复现同结论（90 分钟容量消融，科学上仍完整）
- 真正的候选修复 = **计算轨迹（scratchpad）**：把 "x=6,y=6,op=+,mod=7,ans=5" 改成带中间步骤
  的语料（如 "…,ans=(6+6)=12,12%7=5"）——标准做法，但要改阶段1 语料规格，TA 定
- 廉价探针（可选）：t1-only 加长 3× 训练，验证"是不是单纯欠训练"

## 🧪 scratchpad 实验（✅ 完成 2026-11-22 12:14，判决：🟢 绿——假设完全验证）

> **假设**：直答语料 49→7 一步映射在答案位无梯度（探针已证）。trace 语料把任务分解成
> 49→13（s=raw和）+ 13→7（ans=s%7）两段链式查找，让加法/取模梯度分别落在 s 位/ans 位。

**判决数据**（预登记口径 🟢 ans_acc≥0.6）：
- tf 轨迹（49 题）：E1 ans_acc=1.0/ansCE=0.014 → E2-E8 全 1.0/0.001。对照直答探针 ansCE 八 epoch 钉 1.946（=ln7 精确随机）——**答案位梯度从无到有且收敛，假设验证**
- rollout：t1_trace 24/25(E1)→100%(E2-E8)；poison13/61、id_mod97 三面恒 0——hit/miss 题面判别完美（sep 0.9856，hit_cos 0.9998 vs miss_cos 0.014）
- 正式 report（artifacts/stage2_scratch_report.json）：conf_corr 0.3026（≥0.3 门槛 ✓）、fok_corr 0.9903（远超门槛）
- 注：FOK 0.99 本质是"题面识别"（域外面恒错、in-dist 面恒对，零方差），真正的"面内直觉"要等 t2/t3 面内错题出现才有意义；conf_corr 0.3026 压线过但同因受限
- **下一站**：t2（模97 四则）trace 设计——需长乘法逐位分解，真设计题，TA 拍板后立项
- 语料：`data_trace/`（gen_trace.py 生成，8,388,606 tokens ≈ 288k 题 ≈ 65,536 窗，t1-only，
  与直答探针同预算同 seed 同架构同 lr —— 单一变量=语料格式）
- 格式：`x=6,y=6,op=+,mod=7,s=12,ans=5`，prompt 切在 `s=` 前（corpus.py `_prompt_cut` 已分支）
- 新文件：`gen_trace.py` / `judge_trace.py`（独立判卷：s_mismatch/ans_mismatch 枚举）/
  `probe_tf_trace.py`（s 位与 ans 位分开测）/ `stage2_scratch.json`
- 旋钮（单一来源）：`paths.data_dir`、`judge_mod`（train/report 共用 `ts2.select_judge`）、
  `probe.hit_set`。阶段1 gen.py/judge.py/data/ 零改动
- **预登记验收**（E8 teacher-forced，`probe_tf_trace.py`）：
  - 绿：ans_acc ≥ 0.6 → 分解解锁，扩全 curriculum（t2 trace 设计另立项）
  - 黄：0.2~0.6 → 信号存在但弱；看 s_exact vs ans_acc 定位卡在哪段，继续拆
  - 红：ansCE 仍钉 ln7=1.946 → 分解也无效，回 TA 定架构/训练法层方向
- 校零读数（未训练）：s_digit=0.036 s_exact=0.000 ans_acc=0.000 ansCE=6.356（=ln578 全词表均匀）
- 随机参考：s_exact ≈ 1/13·加权，ans_acc = 1/7 = 0.143，ansCE = 1.946
- 操作：11:41 setsid 发射（PID 会变，认 --config stage2_scratch.json），stdout=runs/scratch_stdout.log

## 🐛 2026-11-21 崩溃事故记录（已修）
- E7 S3300 崩：corpus.py 窗口边界差一——`_gather` 读 stream[i*128..i*128+128]（含 target），
  段边界却用 ceil(cum/128)，t3 为流尾段故最后窗口越界；t1/t2 越界落入后段未暴露
- 修复：corpus.py 单一来源收口——`max_window=(len(stream)-129)//128`，段窗口与 val 采样均钳到此界
- 已从 E6 resume.pth 续跑（全状态+per-epoch 确定性 rng，零漂移），S3325+ 实测通过原崩溃点
- 代价：丢 ~10 分钟墙钟，无数据丢失；β 配置同修同享

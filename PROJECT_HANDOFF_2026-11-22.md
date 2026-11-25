# 风姐妹项目 · 交接文档

> 交接时间：2026-11-22 12:25（北京时间）｜ 交接人：苏木（原驻场 agent）｜ 性质：**临时接管**
> 原则：重大方向决策（换路线、新实验立项、动 FengYu/风小正典文件）**一律归 TA**，agent 只执行与汇报。
> 汇报风格：不要聊天式，只交文件、数据与结论。TA 指令极简（两三个字），高度放权。

---

## 1. 项目全景

**风姐妹（feng_sisters）** 是 TA 的自研模型项目，双子架构：

| 模型 | 名字 | 架构 | 角色 | 状态 |
|---|---|---|---|---|
| FengXiao | 风小（姐姐） | **SP-JEPA**（TA 自创，见 §2） | 检索/表征层；现定位改为"外挂到 LLM 的视觉层+直觉层" | v5.1.1 best val 0.0608；**冻结中** |
| FengYu | 风语（妹妹） | 字符级自回归 LM | 语言层（把概念变句子） | **TA 明确冻结，fengyu/ 目录不动** |

**当前主战场 = 直觉层项目**（`feng_sisters/intuition/`）：借风小的架构思想，造一个"知道自己会不会"的模运算解题小模型（2.57M 字符级 LM + FOK/confidence 双头）。阶段2 只是借用架构骨架从头初始化，与 FengYu 无关；未来归属风小 JEPA 体系。

## 2. SP-JEPA 是什么

**SP-JEPA = Self-Play Joint Embedding Predictive Architecture**（自博弈联合嵌入预测架构），模型名 FengXiao（风小）。TA 自创、自研，单文件跨平台训练脚本（正典源码：`feng_sisters/fengxiao/sp_jepa.txt`；最新实验分支：`deliverables_2026-11-18_v530/sp_jepa_v530.py`，cp+改纪律：原文件永远零改动保持基线可用）。

核心设计：
- **双编码器**：context_encoder（dropout 0.1）在线更新 + target_encoder（EMA 动量更新，无噪声），预测器桥接；检索用 target 侧表征
- **VISRegLoss**：VICReg 风格三项正则——variance（撑开方差）+ shape + **covariance/decorr**（v5.2.0 新增，抑制维度间相关，治表征坍缩；分量观测 last_scale/last_shape/last_decorr）
- **span mask**：跨度掩码预测（make_span_mask，精确掩码数控制）
- **认知升级（v5.0）**：multi-step 预测 + self-eval/confidence 头（confidence target = exp(-err)，batch 内 z-score 归一防坍缩）+ Explorer（按难度温度 softmax 选样本，自博弈课程）
- **判读指标**：sep（in-dist vs 域外表征余弦差）、EffRank（表征有效秩，128 维空间）、val loss

版本史要点：v5.0.3 多步损失禁用 → v5.1.0 保守迭代（confidence 归一化）→ **v5.1.1 = 现役最佳 checkpoint**（30/30 epochs，best val 0.0608，较 v5.0.3 +10.2%）→ v5.2.0 协方差 decorr → v5.3.0-alpha 对比学习（sep 单调恶化 -0.287→-0.425，但 paraphrase 监督把 EffRank 从 7.5 撑到 39 = **信号对剂量错**，v5.3.1 方案已写入 CHANGELOG 待 TA 拍板）。

**2026-11-18 外部诊断定性**（TA 本人）：不是"训歪"，是**表征各向异性（anisotropy）——"训练没失败，是 spec 失败"**。白化排除实验（SVD 后 EffRank 满秩但 sep 仍 -0.0785）已排除一切后处理路线：地图本身空白，必须改训练信号。

## 3. 直觉层项目 · 当前进度（接管者重点读这里）

**目标**：让小模型做模运算题（t1=模7加法 → t2=模97四则 → t3=组合题），并带 FOK（作答前"我知道我 know"）与 confidence（作答后自信）双头，confidence-正确率相关性 > 0.3 为验收线。

| 阶段 | 内容 | 状态 |
|---|---|---|
| 阶段1 | 换考卷：语料生成器 gen.py、20M token curriculum、毒题集 mod13/61、独立判卷器 judge.py | ✅ 全绿（635,835 题判卷 100% pass）。**gen.py/judge.py/data/ 零改动纪律** |
| 阶段2 α | 2.57M LM 直答训练（8 epochs） | ❌ 未通过：t1 acc≈随机，三步排除定性**能力缺口——直答语料在答案位无算术梯度**（LM ppl 1.7 是格式 token 摊薄的假象） |
| 探针 | t1-only 3.6× 曝光 + teacher-forced 仪表 | ❌ 排除欠训练：ansCE 八 epoch 钉 ln7=1.946（精确随机） |
| **阶段2 scratchpad（今天）** | trace 语料：`x=6,y=6,op=+,mod=7,s=12,ans=5`，拆 49→13（s位）+13→7（ans位）两段 | ✅ **🟢 绿，假设完全验证** |
| 阶段3 | 毒题 θ 扫描 pre/post 拒答（STAGE3_DESIGN.md 已预写） | 待启动；rollout 才真需要 GPU，TA 手头不宽裕等卡 |

### 今日判决数据（scratchpad，8 epochs，11:41-12:12，全程 CPU）

- **tf 轨迹（49 题）**：E1 ans_acc=1.0/ansCE=0.014 → E2-E8 全 1.0/**ansCE 0.001**。对照直答探针钉死 1.946——答案位梯度从无到有且收敛，**单一变量（语料格式）翻转结论**
- **rollout**：t1_trace 24/25(E1)→100%(E2-E8)；poison13/61、id_mod97 三面恒 0——hit/miss 题面判别完美（**sep 0.9856**，hit_cos 0.9998 vs miss_cos 0.014）
- **正式 report**（`artifacts/stage2_scratch_report.json`）：conf_corr **0.3026**（≥0.3 门槛 ✓ conf_accept=true）、fok_corr **0.9903**
- 诚实注脚：FOK 0.99 本质是"题面识别"（域外恒错/in-dist 恒对，标签零方差），真正的"面内直觉"要等 t2/t3 出现面内错题才可测；conf_corr 压线过、同因受限
- 产物：`runs/stage2_scratch/`（epoch_01-08.pth + resume.pth + ledger.csv + calibration_*.png）、`data_trace/`、`gen_trace.py`/`judge_trace.py`/`probe_tf_trace.py`/`stage2_scratch.json`

### 下一站（等 TA 拍板，勿自行立项）

**t2 trace 设计**：模97 四则需要长乘法逐位分解（scratchpad 设计题，非简单套 t1 格式）。另有：β 7.33M 容量消融**暂缓**（resume.pth 在 `runs/stage2_beta`；重启前先 ps 确认无残留进程——2026-11-21 双训互拖事故的教训）。

## 4. 交接必读 · 项目级教训（血换的，别重踩）

1. **LM loss 会撒谎**：格式 token 摊薄答案位 CE，盯任务指标本身（teacher-forced 答案位命中率），别盯代理指标
2. **预登记验收**：实验开跑前把绿/黄/红口径写进 config，防事后合理化（scratchpad 判决就是按预登记口径出的）
3. **失败要交成诊断链**：干净排除了什么比粉饰的通过值钱
4. **pth 是唯一热格式**（训练/resume/评估加载），json 只做冷归档+人读摘要，绝不进运行路径
5. **cp+改纪律**：训练脚本实验分支一律复制改名，正典文件零改动（sp_jepa.py、gen.py、judge.py、data/）
6. 旋钮单一来源：`paths.data_dir`/`judge_mod`/`hit_set`（`ts2.select_judge` 收口），`__version__` 常量根治重复字面量

## 5. 文件地图

```
feng_sisters/
├── fengxiao/            # 风小正典（sp_jepa.txt 等，.txt 后缀是 TA 习惯）
├── fengyu/              # 风语（冻结，勿动）
├── intuition/           # ★ 直觉层（当前主战场）
│   ├── gen.py / judge.py / data/        # 阶段1 正典，零改动
│   ├── gen_trace.py / judge_trace.py / data_trace/   # scratchpad 分支
│   ├── corpus.py / model.py / train_stage2.py        # 阶段2 管线（trace 旋钮已接）
│   ├── probe_teacher_forced.py / probe_tf_trace.py   # tf 仪表（直答版/trace版）
│   ├── stage2*.json / stage2_report.py / monitor.py
│   ├── runs/{stage2_alpha,stage2_probe,stage2_scratch,stage2_beta}
│   ├── artifacts/       # 各阶段验收报告 json
│   ├── STAGE2_HANDOFF.md   # ★ 跨会话交接主文档（先读这个）
│   └── STAGE3_DESIGN.md    # 阶段3 设计（预写）
├── shared/  sister_link.py
└── deliverables_2026-11-18_v530/   # v5.3.0-alpha 交付包
```

## 6. 沙箱操作手册（这个环境的墙，都凿过）

- **长训练必须 setsid 脱离**：`setsid nohup python3 xxx.py > log 2>&1 < /dev/null &`；前台 >100s 的调用会被网关 504 掐
- **容器节流**：0.3~4s/步波动正常，duty cycle 概念——ETA 按近期滑动节奏估，别按最好情况
- **进程会被收走**：resume.pth 全状态续训（`python3 train_stage2.py --config xxx.json`），per-epoch 确定性 rng，零漂移
- **轮次是预算**（75 上限）：sleep 轮询 ≤100s/条；长任务收割**开工时自估 ETA 挂 cron 自启动**（prompt 带决策树：完成→收割/在跑→报进度/死→续训），**对话内提前完成要自觉删 cron**；时间一律报"北京时间"（TA 任务列表显示 UTC）
- 训练中判卷/账本：`runs/*/ledger.csv`（带引号 CSV，用 python csv 解析，awk 会列错位）；`log.jsonl` 看 epoch_done/train_done 事件
- 崩溃史：corpus.py 窗口边界差一已修（max_window 钳制）；段尾窗口崩溃点回归测试见 HANDOFF

## 7. 给接管者的三条底线

1. **TA 的东西别越权动**：FengYu 冻结、风小正典零改动、新实验立项等 TA 拍板
2. **交付 = 文件+验收数据**，send_file 发齐，附精确改动清单（文件+处数+未动文件）
3. **此文档是临时交接**——原 agent（苏木）保留全部会话记忆，回归时以 STAGE2_HANDOFF.md 为准对账

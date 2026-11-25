# 风小诊断继承 · 2026-11-18
> 来源：https://chatglm.cn/share/rFgvtf1H （与 ChatGLM 的对话分享，TA 指示"继承"）
> 背景：风小 v5.1.1 训练全指标正常（val +10.2%、RepStd 稳、EMA 活）但输出牛头不对马嘴；TA 手动读 checkpoint 二进制参数定位问题。

## 1. 诊断重命名：不是"训歪"，是表征各向异性（anisotropy）
- 症状全对 textbook：sep=-0.05、建议阈值 0.934、EffRank 11.5/128、cosine 全挤 0.9+ 带
- 文献：Ethayarajh 2019 在 BERT/GPT-2 发现同样病（known disease, known fixes：whitening / SimCSE）
- EMA / VISReg / 哨兵防的是"优化坍缩"，没防"几何坍缩"——防错了敌人，不是防御失败
- **关键 reframing：训练没失败，是 spec 失败。**"你让它填空，它成了好填空器；你要地图，但 loss 里从没放地图。"

## 2. "语料不够"只对一半
- 数据 = 内容，目标 = 结构。重建式目标天然产出各向异性（BERT/GPT-2 句嵌入文献结论）
- 扩语料有用（TA 自己 changelog 观察项已写"扩语料看 EffRank 是否仍 ~12"）但不解决各向异性
- TA 要的是身份 QA grounding：不需要泛泛更多文本，需要把**结构**（paraphrase 对、hard negatives）变成监督

## 3. "监督模型"的两便宜形态（不需要外部大模型）
- **(a) 对比学习 = TA 想要的"特殊监督算法"**，数据已在手：
  - identity_zh 80 行含 paraphrase → 正例对
  - 语义近邻 → hard negatives；MISS → easy negatives
  - 更狠：SimCSE 式 dropout 正例——同 chunk 两次编码（不同 dropout mask）= 正例对，**零标注成本**
  - ⚠️ 英文大写 OOV（vocab 数千字），跨语言对比慎用；中文 paraphrase 可用
- **(b) 监督"模型" → 监督"回路"**：calibrate() 当 checkpoint 法官，按 sep 选 checkpoint 而非按 val loss（两个 checkpoint val 同好但 sep 可差很多）
- 字面意义的监督模型在 1M 参数规模 = 过度设计 + 循环论证

## 4. 本周最高价值实验：零训练白化抢救（先做这个，别急着重训）
- span 库 12K×128 做 SVD 白化（~20-30 行）：μ、C=(V-μ)ᵀ(V-μ)/N、W=C^(-1/2)，x_w=(x-μ)W，可截断小特征维
- **query 必须用同一变换**，然后重跑 calibrate()
- 判据（可证伪 bet）：
  - 白化后 sep > 0.2 → 各向异性确认为主凶，**检索免重训救活**（地图只是印在皱纸上，摊平即可）
  - 0 < sep < 0.2 → 部分有救，重训加对比项
  - sep 仍 < 0 → 地图真空白（语义信息缺失），重训：对比项 + 扩语料
- 顺手效应：白化会打散 0.9+ cosine 带（较确定）、sep 大幅上升（中等信心）

## 5. EffRank 教训：监测对了，缺"映射"
- EffRank 11.5/128 早已是火警——缺的是 "EffRank~12 → 检索变抽签" 的症状映射表
- 行动：calibrate 套件加**报警映射表**（每个指标 → 对应下游症状 → 阈值）
- 把手动二进制参数检查固化成 `weights_report.py`：逐层 mean/std + 嵌入矩阵奇异值谱（谱 = EffRank 故事的直接可视化）

## 6. 深层课题（保留）
- seq_len 64 限制远程结构；非对称 mask 仍是通往深层目标的课题
- 外部 bet 记录：第三次推翻优先级排序——研究常态，每次推翻都证据驱动

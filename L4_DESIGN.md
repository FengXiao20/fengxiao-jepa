# L4: Retrieval-Based Decoding (设计文档)

**Project: FengXiao v0.4.4-preflight**
**Status: 起飞前检查完成，等待硬件。**

## 目标
把 JEPA 的 predictor 用作"导航仪"，在语料库的 span 流形上逐步走出一条路径，
输出训练数据中不存在的新文本。

## 核心机制
1. 输入：完整 context（无 mask）
2. predictor 输出：下一个位置该是什么的"表征"
3. 检索：在 span 库里找最近邻
4. 拼接：把命中的 span 接到 context 后面
5. 循环：直到终止条件满足

## 关键决策
- Span 粒度：1-4 字符（和训练时的 mask 策略对齐）
- Span 库：等距采样 + 随机长度，上限 MAX_SPANS（默认 20000）
- 向量来源：用 target_encoder 编码每个 span
- Pooling：masked mean pooling（padding 位置不参与均值）
- 缓存：span 库落盘缓存，缓存键含文件 mtime，语料变了自动失效
- 终止条件：预测误差 > 阈值，或达到最大步数（默认 10）

## 已解决：分布偏移
训练时 predictor 输入是 masked context（0%~30% 被遮），
解码时输入是完整 context（0% 被遮）。

修法（v4.3 落地）：训练时 mask_ratio 从 [0, 0.3] 随机采样，
让 predictor 见过 0% 到 30% 的全谱可见度。

## 已知限制
- 输出受限于语料库，语料没有的内容无法生成
- 每步都要跑一次前向传播 + 一次全库检索，速度慢
- 拼接的 span 可能语义断裂（predictor 误差就是裁判）
- Span 库 2 万条是妥协：太少了生成单调，太多了编码慢
- **生成质量受限于模型规模**：0.74M 参数 + 几 MB 语料，
  连贯性有限，存在重复 span 问题。优化优先级低于检索层。
- **OOV 字符**：训练 vocab 之外的字符（大写英文、生僻字）
  会被跳过。当前策略是"静默跳过"，未来可加 OOV 比例检查，
  超过阈值直接走 fallback。

## 版本化身份的闭环
等 L4 上线，`identity_zh.txt` 里那行"探索者和内部裁判模块还在理论设计"
改成"已经运行"，重新编码入库。风小的自我陈述自动更新。

**进度（截至 v4.4）**：
- 身份语料已扩展为双语 80 行/语言（含 paraphrase）
- 探索者（Explorer）：已实现，难例挖掘 + 预留 reward 接口
- 内部裁判（Internal Judge）：`logic_generator.verify()` 已实现，
  作为规则裁判用于逻辑语料自检
- L4 上线验证：待硬件落地后跑通 `retrieve.py` 的 `sep > 0`，
  即可更新身份语料中的这句陈述
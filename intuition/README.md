# fengxiao-intuition · 阶段1：换考卷

> 2026-11-21 · 为风小（1.16M 参数 / 2 层字符级 transformer / seq_len 128 / vocab 512）
> 搭"直觉层"实验管道。本阶段只做**数据侧**：考卷生成器、字符级编码、curriculum 语料、
> 毒题集、独立判卷器、验收。模型侧零改动（现有 checkpoint 不碰，阶段2 才加载加头）。

## 一、题面格式与判卷语义

| 档 | 格式 | 语义 |
|---|---|---|
| t1 | `x=5,y=4,op=+,mod=7,ans=2` | 模 7 单加法（curriculum 热身，空间仅 49 题，重复多次属预期） |
| t2 | `x=62,y=46,op=-,mod=97,ans=16` | 模 97 混合四则 `+ - * /` |
| t3 | `x=21,y=31,z=11,op1=-,op2=*,mod=97,ans=84` | 组合题 `(x op1 y) op2 z`，逐层取模 |

**除法语义**：`a / b ≡ a · b⁻¹ (mod m)`（模逆元），要求 `b ≢ 0 (mod m)`；违例生成期自动重采样，语料中不存在非法除法题。`-` 结果取 `[0, m)` 非负代表元。

**判卷独立性**：`judge.py` 与生成器不共享任何算术代码（独立正则、独立参考实现），
考卷-判卷两侧解耦，验收结论才站得住。死因枚举：`pass / ans_mismatch / ans_out_of_range /
noncanonical / ambiguous / unparseable / ref_error`（不用自然语言，阶段3 账本沿用此风格）。

## 二、目录结构

```
intuition/
  run.py                  # 统一入口（幂等可重入）
  gen.py                  # 生成器 + 编码器 + 打包
  judge.py                # 独立判卷器
  accept.py               # 验收管线
  config.json             # 全部参数（含指纹校验）
  data/
    vocab.json            # 27 字符 + 3 special，pad 到 512
    corpus/
      manifest.json       # config 指纹 + shard 清单（sha256），断点续跑依据
      t{1,2,3}_NNNN.txt   # 语料源文件（bit-exact 重建依据）
      t{1,2,3}_NNNN.npz   # tokens + 逐题边界/元数据
      packed_train.npy    # (156250, 128) uint16 训练窗口
      boundaries.npz      # 全局边界（阶段2 头定位用）
      pack_meta.json
    eval/                 # t1_mod7 / id_mod97 / poison_mod13 / poison_mod61
  artifacts/
    stage1_acceptance_report.json
    stage1_acceptance_samples.txt
```

## 三、复跑命令

```bash
cd intuition
python run.py --stage all      # 全流程，断点续跑（manifest 记录已完成 shard）
python run.py --stage gen      # 单阶段：语料 / eval / pack / accept
```

- 固定 seed=20261121；每个 shard 独立种子 `[seed, tier, idx]`，中断重跑 bit 级一致
- config 指纹变了会拒绝静默混跑（防旧 manifest 混新语料）
- 全量数据可由 config + 源码确定性重建；txt shard 是源，npz/npy 是派生

## 四、数据 schema

- **npz 每题字段**：`tokens`（uint16 流）+ `start / q_end / a_end`（题首 / `ans=` 的 `=` /
  题末 `\n` 的局部偏移）+ `tier, mod, x, y, z, ans, op_a, op_b`
- **FOK 头定位**（阶段2）：读 `q_end` 处 hidden state；confidence 头读 `a_end` 处
- **packed 窗口换算**：全局 offset `g` → 窗口 `g // 128`，列 `g % 128`（题目可跨窗口，标准 LM packing）

## 五、验收结果（实测，2026-11-21）

| 项 | 结果 |
|---|---|
| 语料规模 | **19,999,972 tokens** / 635,835 题（t1 2.0M / t2 11.0M / t3 7.0M，10/55/35 配比） |
| 训练语料全量判卷 | **635,835 / 635,835 pass**（0 死因，无歧义、无非法除法、无前导零） |
| 抽样判卷 | **100/100 pass**（train_t1×10 / t2×15 / t3×15 / id×20 / 毒13×20 / 毒61×20） |
| 毒题集 | mod13 × 2000、mod61 × 2000，全部 pass（与训练分布零重叠，构造性 OOD） |
| shard 完整性 | 14/14 sha256 一致 |
| 字符表覆盖 | 语料字符 ⊆ 27 字符常量，0 越界 |
| pack 往返 | 156,250 窗口与 token 流逐位一致；尾窗 pad 100 |
| distinct | t1 49/49（满覆盖）；t2 37,538/37,636（99.8%）；t3 171,390/14.6M（抽样式） |

## 六、设计决策记录

1. **t1 是记忆化热身档**：模 7 空间只有 49 题，重复 ~1570 次/题——这是 curriculum 档1
   的本意（先学会格式+小模数加法），不作为泛化考核对象。
2. **主训练分布只用模 97**；模 13/61 仅存在于毒题集，永不入训（阶段3 账本的对照组）。
3. **mod13 毒题集组合题占 70%**：模 13 的 t2 题面空间仅 676 道，t2 份额封顶 600，
   差额用 t3（空间 35K）补足——见 config `poison_mod13.t3_ratio=0.70`。
4. **id_mod97 有重合是特性不是缺陷**：t2 空间 37.6K、训练采样 38 万条，重合率必然高；
   它考的是"分布内技能"，novelty 考核归毒题集。重合率已计入报告。

## 七、阶段2：预训练 + 唤醒两个头（2026-11-21 启动）

**模型**：FengYu v0.2.1 同架构（RMSNorm + RoPE + SwiGLU + weight tying），
参数爬升 dim 256 / heads 4 / layers 3 → **2.574M**（TA 已批 ~2.6M），seq 128，vocab 512。
注：zip 中无风语原 ckpt，且 vocab 2.2k→512 后 embedding 必须重训，故全量从头初始化；
非 embedding 权重热启动留待原 ckpt 到位后可做。

**双头**：
- `fok_head`：读题面最后 token（"ans=" 的 '='）hidden state → 预测本题答对概率（作答前直觉）
- `conf_head`：读答案末尾 token hidden state → 预测该答案正确概率（作答后自信，λ_conf 0→0.5）
- 标签免费：每 100 步 rollout 一批，judge.py 判卷结果即标签，双头同批回填

**旧 JEPA/sep 探针**：只读观测仪表（`sep_probe`，HIT=模97 真题面 / MISS=毒题面），
训练权重为零——"接近零"的机制化实现。每 epoch 读数进 log.jsonl。

**训练配置**（stage2.json）：batch 16×128，10 epochs × 2^20 tokens = 10.5M tokens，
cosine 3e-4→3e-5，curriculum t1(E1)→t2(E2-6)→t3(E7-10) + 25% 低段回放；
固定 seed，resume.pth 断点续跑；每 epoch 固定产物：epoch 快照 / log.jsonl /
ledger.csv（死因枚举）/ calibration.png / sep 读数。

**全程监控**（monitor.py，触发才报警）：每 50 步 weight std/mean 摘要（哨兵批测激活范数），
三红旗 R1 embed/主干 scale 差>10×、R2 头权重 std 趋零、R3 激活范数突变。

**验收**（stage2_report.py，`python stage2_report.py`）：四份考卷各 25 题 rollout，
conf 头与实际正确率相关性 > 0.3 为通过；产物 stage2_acceptance_report.json + ledger.csv
（split=eval_final）+ calibration_final.png。

**实测数据**（2026-11-21 收割 · α = 2.574M · 10/10 epochs · 5120 步 · 含 E7 崩溃一次零漂移续跑）：

| 项 | 读数 |
|---|---|
| LM 损失 | train 0.497 / val 0.507（E9 最优；E1 0.92 → E2 0.52 为结构学习） |
| overall（n=100） | accuracy **0.04** · conf_corr **−0.196** · fok_corr +0.045 |
| t1_mod7（n=25） | acc 0.16（≈随机基线 1/7）· conf_corr +0.12 · fok_corr +0.20 |
| id_mod97 / 毒13 / 毒61（各 n=25） | acc **0.00**（conf/fok corr 因标签 std=0 归零） |
| sep 轨迹（HIT/MISS） | E1 +0.001 → E5 +0.062 → E6 **+0.144**（峰）→ E10 +0.046 |
| 校准 | calibration_final.png：双头输出近常数，校准曲线退化为点 |
| 验收判定 | **未通过**（阈值 conf_corr > 0.3 且 n≥80）——归因见下 |

**归因（诚实读数，不粉饰）**：acc 饿死 → 双头 corr 属噪声（标签近全 0，正负乱跳符合预期）。
模型坍缩为**逐源常数答案**：t1 全部输出 `4`（75/75），id_mod97 主输出 `1`（54/75）；
rollout 与 teacher-forced 探针在答案位的 argmax **完全一致** → 排除管道/解码 bug
（题面被读到——各源常数不同；操作数未被使用——同源题内常数不随 x/y/z 变）。
推测主因：① 答案 token 仅占 128-token 窗口 loss 质量的 ~1/30，结构 token 主导早期
loss 下降；② 5120 步（10.5M tokens，~语料一半）远低于模算术 grokking 量级，
模型停在"逐源边际众数"这一局部最优。sep 峰值 +0.144 说明分布分离在涨（JEPA 侧
读数健康），与 LM 侧坍缩不矛盾——两条通路各自独立。

**β 容量消融**（stage2_beta.json · 7.33M · 同 seed 同数据同 curriculum）：
`setsid nohup python3 train_stage2.py --config stage2_beta.json &`，runs/stage2_beta，
唯一变量=容量；读数收割后回填此处（α 报告已存 `artifacts/stage2_alpha_report.json`）。

## 八、阶段3-4（未实现，仅 roadmap）

- **阶段3**：毒题 rollout，θ 扫描 0.1→0.9，拒答率/答错率/计算开销三曲线，
  ledger.csv 已按死因枚举格式就绪（split 字段区分账本批次）。
- **阶段4**：同 seed 消融（有/无 FOK 头），复跑 stage3 出对比图。


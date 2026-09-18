# FengXiao Changelog

## v0.4.4-preflight (2026-09-16)

- 数据层：logic_generator v4（14/14 自检通过）
- 语料层：身份双语 80 行/语言 + 逻辑语料 10000 条
- 模型层：sp_jepa v4.4（EMA 活着 / mask 修复 / env 检测）
- 检索层：retrieve V1.5（规则路由 + dedup + 语言路由）
- 解码层：l4_decode v1.1（OOV 统一 + masked pooling）
- 状态：语法全通过，**等待硬件落地**跑通 sep
- 硬件推荐：RTX 3060级别+

## 下一步 (v0.5.0)
- [ ] PC 环境跑通 sp_jepa.py
- [ ] retrieve.py 的 sep 转正
- [ ] 更新 identity_zh.txt：探索者/裁判 "已经运行"
- [ ] 补 L4_DESIGN.md
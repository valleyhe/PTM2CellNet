# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 完整人类蛋白质组序列下载 (UniProt 204,729条)
- 数据整合脚本 v2 (80,508条整合数据 + 112,012条已标记数据)
- Mamba编码器支持 (Selective State Space Models)
- PEFT/LoRA微调集成
- GENKI3可解释性框架集成
- 氨基酸字符标准化 (U→C, X→A, J→L, B→D, Z→E, O→K)
- Docker生产部署配置
- docker-compose编排配置

### Changed
- PTM融合机制扩展为 attention + gated 两种策略
- 池化方式支持 mean / attention / weighted_mean / multi-head
- Python版本要求从3.9+更新为3.10+
- mypy配置从3.9更新为3.10

### Fixed
- dbPTM数据格式解析兼容性修复 (5列 vs 6列格式)
- 预训练编码器测试网络依赖问题
- 日志f-string格式警告 (W1203)
- 混合行尾符问题 (CRLF → LF)
- evaluation/__init__.py 类型注解问题

### Known Issues
- PTM位点-氨基酸匹配率 87.13% (多源数据整合预期范围内)
- 预训练编码器测试需要网络连接
- cell_state标签仅2种 (Quiescent/Activated)，需扩展

## [1.0.0] - 2026-02-24

### Added
- 基础模型架构 (CNN, Transformer, LSTM编码器)
- PTM处理模块 (注意力融合)
- 分类/回归预测器
- 数据处理模块 (加载、预处理、特征提取)
- 训练器模块 (原生 + Lightning)
- 评估模块 (指标、可视化)
- API模块 (FastAPI)
- 单元测试套件

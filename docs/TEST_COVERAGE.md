# 测试覆盖率治理

PTM2CellNet 使用语句覆盖率与分支覆盖率共同度量。覆盖率是回归风险门禁，不能替代
真实模型、GPU、外部服务和受控数据资产的分层验收。

## 可重复的默认门禁

默认 CI 运行无网络依赖的单元、核心集成和根目录单测，并拒绝覆盖率低于当前基线：

```bash
python -m pytest tests/unit tests/integration tests/test_*.py \
  -m "not slow and not gpu" \
  --cov=src --cov-branch \
  --cov-report=term-missing:skip-covered \
  --cov-report=xml \
  --cov-fail-under=74
```

`.coveragerc` 是覆盖范围的单一配置源。最低门槛采用 ratchet 策略：覆盖率提升后同步
上调 `fail_under`，不得为了通过 CI 排除业务模块或添加无断言测试。

## 分层验证

- 默认覆盖率门禁：纯 CPU、无下载、无外部服务，适合每次 push/PR。
- 集成/E2E：`python -m pytest tests/integration tests/e2e -m "not slow and not gpu"`。
- 真实资产：按 `docs/guides/real_assets_acceptance.md` 显式启用；缺少资产时必须 skip，
  不能以 mock 结果宣称真实资产通过。

## 补测优先级

1. 数据契约、安全 I/O、checkpoint/config 和 API 输入边界。
2. 核心模型的形状、mask、fallback/provenance 与错误路径。
3. 训练和评估的状态转换、恢复、提前停止与退化输入。
4. 第三方网络/GPU 路径放入对应分层测试，不进入默认门禁。

新增测试应覆盖正常、空/边界、异常和至少一个回归场景，并在本地先跑定向测试，
再跑默认覆盖率门禁。

## 2026-09-01 状态

2026-09-01 全量离线回归为 **2357 passed、16 skipped、47 warnings、2373 项收集、536.65s**，
详细命令和证据见 [`project_analysis_20260910.md`](../project_analysis_20260910.md) 的验证章节。
本轮未执行带 `--cov` 的独立覆盖率测量，因此不能把历史数值当作当前覆盖率；门禁仍为
74%，下一次覆盖率刷新应在依赖一致的 CI/独立环境完成。

2026-08-09 的旧覆盖率快照（1866 passed、5 skipped、75.27% branch coverage，以及
1920 passed、13 skipped 的完整套件记录）已归档至
[`archive/20260827/reports/TEST_COVERAGE_20260809.md`](../archive/20260827/reports/TEST_COVERAGE_20260809.md)，
以保留原始基线和历史审计记录。

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

## 2026-09-13 状态

2026-09-01 全量离线回归为 **2357 passed、16 skipped、47 warnings、2373 项收集、536.65s**，
详细命令和证据见 [`archive/20260914/project_analysis_20260913.md`](../archive/20260914/project_analysis_20260913.md) 的验证章节（当前权威分析为 [`project_analysis_20260917.md`](../project_analysis_20260917.md)）。
2026-09-13 默认口径回归为 **2578 passed、21 skipped、69 warnings、798.65s**（`-m "not slow and not gpu" --timeout=300`）；本轮仍未执行带 `--cov` 的独立覆盖率测量。
本轮未执行带 `--cov` 的独立覆盖率测量，因此不能把历史数值当作当前覆盖率；门禁仍为
74%，下一次覆盖率刷新应在依赖一致的 CI/独立环境完成。

DAVF × PerturbGen 的 matched-null 生成、候选 empirical-p 聚合、formal 输入隔离、
未扰动质量、donor split 和 dual-path AND 已有代码接口；E2E 自 2026-09-13 第四轮起
经 `--assemble-statistical-evidence` 自动串接并写回 lineage，接口回归通过不等于
真实统计 evidence。正式验收仍需真实 normal/disease raw counts、显式 donor、
至少 3 个可评估 donor（pairing 显式声明）、canonical Ensembl、冻结
scVI/embedding manifest、真实 null/质量和双场景统计。fixture、mock 和 smoke
只验证工程契约或可运行链路；`real_assets` 只表示资源门控的测试层，测试通过数或
coverage 百分比都不是生物学验收结果。Gate-0 现状：scPerturb 0/30 合规；
GSE174367 已过 between_donor 数据契约 preflight 并完成 EX M6 冻结（数据契约
验收，非生物学 PASS）。

PTM activity → AD 交集主线（2026-09-14）新增 5 个单元契约测试文件与 1 个
四阶段 CLI 全链集成测试（`tests/integration/test_ptm_activity_pipeline.py`），
合成数据只证明契约接通。同轮全量回归为 **2760 passed / 1 failed（第三轮起的
已知真实资产基线：旧 checkpoint 非 canonical ENSG）/ 22 skipped**；mypy 171
文件 0 errors。主线命令与数据契约见
[`guides/ptm_activity_pipeline.md`](guides/ptm_activity_pipeline.md)。

研究边界以 [`CURRENT_STATUS`](CURRENT_STATUS.md)、[中央双路径方案](DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md)、[PTM activity 执行方案](PTM_activity_AD_intersection_DAVF_PerturbGen_执行方案.md) 和 [`task_plan.md`](../.planning/task_plan.md) 为准。

2026-08-09 的旧覆盖率快照（1866 passed、5 skipped、75.27% branch coverage，以及
1920 passed、13 skipped 的完整套件记录）已归档至
[`archive/20260827/reports/TEST_COVERAGE_20260809.md`](../archive/20260827/reports/TEST_COVERAGE_20260809.md)，
以保留原始基线和历史审计记录。

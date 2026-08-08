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

## 2026-08-08 基线与剩余缺口

当前可重复核心套件结果为 **1785 passed、5 skipped，综合 branch coverage 75.00%**；
门禁设置为 74%，用于吸收 Python/可选依赖造成的小幅路径差异。独立 E2E/CI/real-assets
套件结果为 **54 passed、8 skipped**，其中真实资产未启用的 skip 属预期行为。

本轮重点模块覆盖率：

| 模块 | 覆盖率 |
|---|---:|
| `src/data/data_manifest.py` | 97.99% |
| `src/data/loaders/file_loaders.py` | 94.97% |
| `src/integration/contracts.py` | 95.83% |
| `src/integration/ptm_gene_mapper.py` | 96.67% |
| `src/utils/checkpoint_utils.py` | 95.54% |

下一轮 ratchet 应优先补核心且低覆盖的 `training/self_supervised.py`（31.62%）、
`api/routes/initialize.py`（47.88%）、`data/validation.py`（53.65%）和
`training/callbacks.py`（53.71%）。`scvi_adapter.py`、`latent_davf.py`、GenKI 与
外部工具路径依赖可选运行时或真实资产，应在对应分层套件中提升，不能通过排除文件
或宽泛 `pragma: no cover` 提高总数。

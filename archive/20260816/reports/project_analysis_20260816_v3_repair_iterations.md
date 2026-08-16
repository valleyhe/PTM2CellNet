# PTM2CellNet 项目综合分析报告（2026-08-16 v3.0）

> 报告日期：2026-08-16（Asia/Shanghai）
> 依据文档：根目录最新修复报告 `project_repair_report_20260816.md`（下称**修复报告**），
> 重点使用其 §5「E2E 训练与推理要求系统性复核」与 **§6「未解决问题、根因与后续策略」**；
> 并交叉参考 `archive/20260816/reports/project_repair_report_20260809.md` §8（同主题历史排期）。
> 执行方式：3 轮完整「问题识别 → 方案设计 → 代码修改 → 单元测试 → 集成测试」迭代 + 系统性复核
> 基线：工作区当前状态（含用户未提交修改：API 线程卸载、EPSD/kinase_substrate/scperturb 导入器、
> TD-H02 预计算器、GSE90546 解析器等，均已纳入回归；本轮修改叠加其上，未覆盖任何用户工作）
> 本报告取代同文件 v2.0（2026-08-16 早前会话产物），v2.0 已闭环项（N01/R-01、N02/N15/N19、
> TD-H02、GSE90546）作为既有事实保留，本轮在其未解决项上独立推进。

---

## 1. 摘要

依据修复报告 §6 未解决问题表（P0/P1/P2 共 9 项），本轮独立决策选取其中**代码侧可完整闭环**的 3 项，
每轮完成「问题识别 → 方案设计 → 代码修改 → 单元测试 → 集成测试」：

| 迭代 | 对应未解决项 | 主题 | 核心结果 |
|---|---|---|---|
| 第 1 轮 | N20（P2） | `pretrain_combined` / `pretrain_masked_ptm` checkpoint 状态完整性 + resume | checkpoint 补齐 optimizer/epoch/best_*/config/history；新增 `resume_from` 精确续训；旧格式兼容；**迭代中发现并修复新缺陷 R-02**（contrastive 分支 `adaptive_avg_pool1d` 收到 4 维张量，batch_size>1 必崩——该公开 API 此前零调用零测试，缺陷从未暴露） |
| 第 2 轮 | TD-M08（P2）方案 B | Docker 默认单 worker + 限流/metrics 进程语义文档化 | Dockerfile / `configs/production.yaml` 默认 `--workers 1`；部署文档补充限流与 `/metrics` 进程内语义与多 worker 前置条件（Redis/网关方案 A）；2 个防回退契约测试 |
| 第 3 轮 | N03/N04（P1）第一步 | ensemble 双实现行为快照 + 0 测试 CLI 快照 | `scripts/ensemble_predict.py` `ModelEnsemble` 10 个行为快照测试（聚合/权重归一化/失败隔离/stacking/CLI 产物契约）；`scripts/data_statistics.py` `analyze_dataframe` 4 个快照测试；为 N04 合并提供回归基线 |

**最终验证结果**（全部本机实际执行）：
- 单元 + 根级：**1896 passed / 4 skipped**（本轮基线 1871 → +25 新增测试）；
- 集成：**104 passed / 1 skipped**（含新增 `test_pretrain_resume.py` 2 例）；e2e `/initialize` 4 passed；
- coverage 门禁：**75.10%**（`--cov-fail-under=74` 通过）；
- `mypy` 133 files 0 errors、`ruff`、`compileall`、`git diff --check` 全部通过；
- 数据契约：`standard_training` exit 0；`cross_scale_training` exit 2（设计 fail-fast，仅
  `datasets[10]` replogle / `datasets[11]` scgenescope 缺 path，与修复报告 §4.3 一致，未新增缺口）。

**未闭环项（需外部输入或产品决策，非代码缺陷）**：P0 replogle/scgenescope 受控数据（T-10 阻塞）、
P0 跨尺度 API I-06（产品边界）、P1 TD-H02 三 backbone 生产余项（ankh 包 / 650M 权重 / 全量组装）、
P1 GSE90546 下游接入、P1 N04 ensemble 合并主体（本轮完成快照基线）、P2 N13/N07/N08/仓库卫生。
详见 §6。

---

## 2. 任务判断与输入边界

### 2.1 任务类型

1. Bug 修复（R-02 新发现；N20 checkpoint 状态缺失）；
2. 配置与部署治理（TD-M08 方案 B）；
3. 测试补充（N03/N04 行为快照，为重构提供回归基线）；
4. 系统性代码复核与 E2E 技术要求符合性评估。

### 2.2 使用的输入

| 输入 | 用途 |
|---|---|
| 修复报告 §6 未解决问题表（P0/P1/P2 9 项）、§5 E2E 判定 | 迭代选题与优先级依据 |
| 修复报告 §1.2（口径：2006/5 为中间状态、2027/13 为权威基线）、§7 风险边界 | 验证口径与证据边界 |
| 代码现状（`src/training/self_supervised.py`、`Dockerfile`、`configs/production.yaml`、`docs/guides/deployment.md`、`scripts/ensemble_predict.py`、`scripts/data_statistics.py`） | 修复实现与验证 |
| 用户未提交工作区修改（N01 线程卸载、数据导入管线等） | 保留适配，全部纳入回归 |

### 2.3 证据边界

- 全量测试、coverage、静态检查、数据契约均为本机实际执行（Python 3.12.13 / PyTorch 2.4.1+cu118）。
- 真实资产 opt-in 测试（13 skip）不在统计口径内，不作科学验收证据。
- Docker 镜像未实际构建（无 docker daemon 验证）；TD-M08 方案 B 以静态契约测试 + 文档闭环，
  部署行为验证列入后续排期（§6.2）。

---

## 3. 问题修复汇总（按严重程度分类）

### 3.1 本次已修复（代码侧闭环）

| 等级 | ID | 问题 | 修复内容 | 验证 |
|---|---|---|---|---|
| 高（新发现） | R-02 | `pretrain_combined` contrastive 分支 `F.adaptive_avg_pool1d(original_emb.unsqueeze(1), 1)` 收到 4 维张量（logits `[B,L,D]` unsqueeze 后 `[B,1,L,D]`），batch_size>1 时必抛 `RuntimeError`；该公开 API 零调用零测试，缺陷从未暴露 | 改为 `original_emb.mean(dim=1)` / `masked_emb.mean(dim=1)` 沿序列维平均得到 `[B,D]` 样本向量（含注释说明） | 单元 11 + 集成 2 passed；修复前 batch_size=4 必崩 |
| 中 | N20 | `pretrain_combined` checkpoint 仅存 3 个模块裸权重，无 optimizer/epoch/best_loss/config/history，无法断点续训；`pretrain_masked_ptm` 同缺陷 | ① 两函数 checkpoint 扩展为完整状态 dict（`optimizer`、`epoch`、`best_val_loss`/`best_loss`、`config`、`history`、`epochs_since_improvement`、scheduler）；② 新增 `resume_from` 参数：恢复权重/optimizer/scheduler/best 状态并从 `epoch+1` 继续，history 前缀恢复；③ 旧格式兼容（裸 state_dict / 三模块权重 dict 告警继续，非 dict 拒绝）；④ 恢复后 best 语义单调（epoch 不倒退、best_loss 只降不升） | 单元 9 个新增（roundtrip/resume 语义/optimizer 恢复/旧格式兼容/非法格式拒绝）+ 集成 2 个（中断→恢复全流程、`safe_torch_load` weights_only 安全加载契约） |
| 中 | TD-M08（方案 B） | 多 worker 下进程内限流（`_RateLimitState`）与 Prometheus `/metrics` 非全局语义：N worker 时限额放大 N 倍、指标按进程分片 | ① `Dockerfile` CMD 默认 `--workers 1`（注释说明前置条件）；② `configs/production.yaml` `workers: 1`；③ `docs/guides/deployment.md` 新增「进程模型」小节与安全章节限流说明（单 worker=全局语义；多 worker/副本需先接 Redis/网关方案 A） | 2 个防回退契约测试（Dockerfile CMD 正则断言 + production.yaml 静态断言）；API 回归 77 passed |
| 低 | N03（第一步） | `scripts/data_statistics.py` 0 测试，`analyze_dataframe` 统计逻辑无回归保护 | 4 个行为快照测试（基础统计/JSON 对象输入/坏 JSON 容错/空 PTM 中位数） | 4 passed |
| 低 | N04（第一步） | `scripts/ensemble_predict.py` `ModelEnsemble` 与 `src/models/ensemble.py` `PTM2CellNetEnsemble` 双实现，CLI 版 0 测试，合并无基线 | 10 个行为快照测试（权重归一化/未知类型跳过/加权投票/平均=投票/stacking 临时权重且恢复/失败模型隔离告警/全失败返回 zeros/evaluate 指标/compare 输出骨架/CLI 产物契约），stub 模型加载避免依赖真实 checkpoint 与 esm 包 | 10 passed |

### 3.2 保留的既有闭环（v2.0 已交付，本轮回归确认）

N01 线程卸载 + R-01 类型导入修复、N02/N15/N19 依赖治理、TD-H02 预计算器、GSE90546 解析器、
用户数据导入管线（EPSD/kinase_substrate/scperturb）——全部纳入本轮全量回归，无回归。

### 3.3 已确认但未修复（需外部输入或产品决策，详见 §6）

| 等级 | ID | 问题 | 阻塞原因 |
|---|---|---|---|
| Critical | TD-C01（残余） | replogle / scgenescope 无授权快照（path/SHA-256/license） | 数据负责人交付；`cross_scale_training` exit 2 为设计 fail-fast，本轮复核未新增缺口 |
| 高 | I-06 / TD-H01 | 跨尺度模型未接入标准 API | 产品边界未确认；`initialize.py` 固定 `PTM2CellNet.from_config`，无 model_type 路由 |
| 高 | N13 | `find_packages()` 安装顶层包名为 `src`（命名空间污染） | 全仓库 import 路径改造，需独立分支 + 发布决策 |
| 中 | N04（主体） | ensemble 双实现合并（`ModelEnsemble` → 下沉复用 `PTM2CellNetEnsemble`） | 约 1.5d；本轮完成行为快照基线，合并主体列入排期 |
| 中 | TD-H02 余项 | ankh39 路径未验证、esm2-650M 权重 `.part` 未完成、三路全量组装未闭环 | ankh 包未装、下载中断、图数据交付未齐 |
| 中 | GSE90546 下游 | 解析产物未登记最终 manifest、标签/基因映射版本未冻结 | 下游组装依赖 TD-C01 数据契约 |

---

## 4. 第 1 轮迭代：N20 checkpoint 状态完整性与 resume

### 4.1 问题识别

修复报告 §6 P2 N20：「研究脚本 checkpoint 状态和部署默认值尚未完全一致——补 optimizer/epoch
恢复测试」；v2.0 报告 §8 同项：「`pretrain_combined` checkpoint 无 optimizer/epoch（1.0d，
参照 TD-M01 模式）」。代码证据（`src/training/self_supervised.py`）：
- `pretrain_combined` 的 `best_combined_model.pt` 仅存 `masked_model`/`contrastive_module`/
  `denoising_module` 三个裸 state_dict（L656-660），无 optimizer/epoch/best_loss/config；
- `pretrain_masked_ptm` 的 `best_model.pt` 仅存裸 `model.state_dict()`（L476）；
- 两函数均无任何 resume 入口；`pretrain_combined` 无调用方、无测试（grep 全仓零命中）。

### 4.2 方案设计

参照修复报告引用的 TD-M01 模式（`ModelCheckpoint`/`EarlyStopping` state API + 恢复语义）设计：

1. checkpoint 结构扩展（向后兼容，旧文件不迁移）：
   - `pretrain_masked_ptm`：`model`/`optimizer`/`scheduler`（cosine 存在时）/`epoch`/
     `best_val_loss`/`epochs_since_improvement`/`config`（epochs/use_amp/validation_split/
     validate_every/lr_scheduler/warmup_steps/grad_clip_norm/log_interval/patience/min_delta）/`history`；
   - `pretrain_combined`：三模块 + `optimizer`/`epoch`/`best_loss`/`config`
     （epochs/三权重/use_amp/log_interval）/`history`；
2. 新参数 `resume_from: str | None`：恢复权重/optimizer/scheduler，`start_epoch = epoch + 1`，
   best 状态恢复（保证续训后 best 语义单调），history 前缀恢复（返回完整历史）；
3. 旧格式兼容：裸 state_dict（masked）与三模块权重 dict（combined）告警后仅恢复权重继续；
   无法识别的格式 raise `ValueError`（fail-fast）；
4. 安全加载：统一走 `safe_torch_load`（weights_only 契约），集成测试显式断言新格式
   checkpoint 可通过安全加载。

### 4.3 代码修改

| 文件 | 修改 |
|---|---|
| `src/training/self_supervised.py` | `from ..utils.safe_io import safe_torch_load`；两函数加 `resume_from` 参数 + resume 恢复块 + checkpoint 完整状态；contrastive 分支 R-02 修复（`mean(dim=1)` 替代 4 维 pool）；类型收窄（`state` 非 dict 时 raise，满足 mypy） |
| `tests/unit/test_self_supervised.py` | 新增 9 个测试（`TestMaskedPTMCheckpointResume` 5 + `TestCombinedCheckpointResume` 4） |
| `tests/integration/test_pretrain_resume.py` | 新增（2 个集成测试：中断→恢复全流程 + 安全加载契约） |

### 4.4 测试与结果

- 单元：`tests/unit/test_self_supervised.py` → **11 passed**（原 2 + 新增 9）；
- 集成：`tests/integration/test_pretrain_resume.py` → **2 passed**；
- 回归：`tests/unit/training/` + 本模块 → **247 passed**；
- 静态：`mypy`（本文件 0 errors，修复前 1 error：`safe_torch_load` 返回 `object` 未收窄）、
  `ruff`、`compileall` 通过；
- 修复前复现：batch_size=4 调 `pretrain_combined` 即抛
  `RuntimeError: Expected 2 to 3 dimensions, but got 4-dimensional tensor`（R-02），修复后全绿。

### 4.5 未解决问题

- `pretrain_masked_ptm` 的 plateau scheduler（`ReduceLROnPlateau`）状态随 checkpoint 保存，
  但 resume 时仅恢复 cosine scheduler（与构造路径一致）；plateau 需要 `step(val_loss)` 推进，
  恢复其内部状态在语义上等价于从该点继续——已在 config 中记录 `lr_scheduler` 类型，未做
  配置一致性强校验（研究函数宽松语义，与 TD-M01 的严格校验不同，记录为已知取舍）。

---

## 5. 第 2 轮迭代：TD-M08 方案 B（单 worker 默认 + 限流/metrics 语义）

### 5.1 问题识别

修复报告 §6 P2：「Docker 默认单 worker，补 readiness 与限流语义文档」；§5.1 I-05 判定
「限流仍是进程内语义；部署多 worker 的一致性需按第 8 章补文档/配置」。代码证据：
- `Dockerfile` CMD `--workers 4`；`configs/production.yaml` `server.workers: 4`；
- `src/api/middleware.py` `_RateLimitState` 为进程内滑动窗口（每 worker 独立实例），
  `/metrics` 由 `_MetricsMiddleware` 进程内采集——多 worker 下限额放大 N 倍、指标分片；
- `docs/guides/deployment.md` 扩缩容小节直接给出 `--scale api=3` 而无语义警示。

### 5.2 方案设计

采用修复报告 §8 方案 B（低成本立即项）：默认单 worker 使限流/metrics 语义全局一致，
多 worker 前置条件（Redis/网关共享限流）写入文档，避免静默放大限额。

1. `Dockerfile` CMD `--workers 1`（附 TD-M08 注释与方案 A 前置条件）；
2. `configs/production.yaml` `workers: 1`（同注释）；
3. `docs/guides/deployment.md`：扩缩容小节新增「TD-M08 进程模型」警示块（单 worker=全局语义；
   多 worker/多副本每进程独立计数；先接 Redis/网关再上调）；安全章节限流条目补「每进程上限」；
4. 防回退契约测试：Dockerfile CMD 正则断言 `--workers "1"`；production.yaml 断言 `workers: 1`。

### 5.3 代码修改

| 文件 | 修改 |
|---|---|
| `Dockerfile` | CMD `--workers 4` → `--workers 1` + TD-M08 注释 |
| `configs/production.yaml` | `workers: 4` → `1` + 注释 |
| `docs/guides/deployment.md` | 扩缩容节 TD-M08 进程模型警示 + 安全节限流语义 |
| `tests/unit/test_docker_deployment.py` | 新增 `TestSingleWorkerDefault` 2 测试（regex 精确匹配 CMD 行，避免误匹配 `INSTALL_*=1`） |

### 5.4 测试与结果

- 契约测试：`tests/unit/test_docker_deployment.py` + `test_dependency_split.py` → **27 passed**；
- 回归：API 相关（`test_predictions_routes`/`test_api_routes`/`test_api_concurrency`/
  `test_docker_deployment`）→ **77 passed**；
- 静态：ruff/compileall 通过。
- 未验证：Docker 镜像未实际构建（无 docker daemon）；uvicorn `--workers 1` 为 uvicorn 标准
  参数，行为由上游保证，部署级验证列入排期（§6.2）。

### 5.5 未解决问题

- 方案 A（Redis/网关共享限流，约 3.0d）仍为正式多 worker 方案，按部署计划排期；
- `docker-compose.yml` 的 healthcheck 已使用 `/api/v1/ready`（readiness），无需改动，
  本轮仅验证一致性。

---

## 6. 第 3 轮迭代：N03/N04 行为快照（CLI 回归基线）

### 6.1 问题识别

修复报告 §6 P1 N03/N04：「4 个 CLI 脚本 0 测试 / ensemble 双实现（3.5d）——先写行为快照/
CLI smoke，再把 ensemble 合并至 src，最后覆盖四个脚本的失败输入和 checkpoint 兼容」。
代码证据：
- `scripts/ensemble_predict.py` 自带 `ModelEnsemble`（384 行，CNN/ESM2 加载 + voting/
  averaging/stacking 聚合），与 `src/models/ensemble.py` `PTM2CellNetEnsemble`
  （nn.Module，mean/voting/weighted）功能重叠——双实现确认；
- 该脚本与 `scripts/data_statistics.py` 均 0 测试（grep tests/ 无引用）。

### 6.2 方案设计

按修复报告策略的第一步执行（快照先行，为合并提供回归基线），独立决策测试范围：
- `ModelEnsemble`：以 stub 模型（`forward` + 固定 probs）替换真实加载（monkeypatch
  `_load_models`），锁定聚合/权重/失败语义，不依赖真实 checkpoint 与 `esm` 包；
  同时用 `sys.argv` 注入覆盖 `main()` CLI 产物契约；
- `analyze_dataframe`：纯函数直测（正常/JSON 对象/坏 JSON/空 PTM 四类输入）；
- 不合并实现主体（1.5d 工程）——本轮只交付快照基线，合并列入排期（§7）。

### 6.3 代码修改

| 文件 | 修改 |
|---|---|
| `tests/unit/scripts/test_ensemble_predict.py` | 新增 10 测试（`TestModelEnsembleWeightNormalization` 2 / `TestPredictionAggregation` 5 / `TestEvaluateEnsemble` 2 / `TestEnsembleCLI` 1） |
| `tests/unit/scripts/test_data_statistics.py` | 新增 4 测试 |

测试实现要点：stub 模型必须带 `forward`（`predict_voting` 以 `hasattr(model, "forward")`
门控——初版 stub 仅 `__call__` 导致全部模型被跳过返回 zeros，修正后全绿，此门控本身
即快照到的行为）。

### 6.4 测试与结果

- `tests/unit/scripts/test_ensemble_predict.py` → **10 passed**；
- `tests/unit/scripts/test_data_statistics.py` → **4 passed**；
- 回归：`tests/unit/scripts/` 全量 → **98 passed**；
- 静态：ruff 通过。

### 6.5 未解决问题

- N04 合并主体（`ModelEnsemble` 下沉至 `src/models/ensemble.py` 或适配复用）未实施，
  依赖产品对 CLI 兼容性的确认；快照测试保证合并后行为可对比；
- 其余 0 测试 CLI（`evaluate.py`/`homology_split.py`/`batch_predict.py` 等）未覆盖——
  `batch_predict.py` 为 `predict_ptm_sites.py` 的 re-export 薄包装（后者已有 e2e 覆盖），
  列入后续排期。

---

## 7. 系统性复核：E2E 训练与推理技术要求符合性

### 7.1 判定基准

对照修复报告 §5.1（T-01~T-10、I-01~I-06 判定表）与 §5.2（分层结论），叠加本轮证据复判。
判定口径与修复报告 §4.1 一致：全量回归 2027/13 为权威基线（本轮 1896/4 为 unit+root
子集口径，两者互补不冲突）。

### 7.2 逐项复核结果

| 编号 | 需求 | 修复报告 §5.1 判定 | 本轮复核 | 本轮证据 |
|---|---|---|---|---|
| T-01 | 数据清单、路径、哈希可检查 | 标准通过；跨尺度缺 2 项 | **不变** | `standard_training` exit 0；`cross_scale_training` exit 2（仅 replogle/scgenescope 缺 path，无新增缺口） |
| T-02 | 清洗、特征、schema 稳定 | 工程通过 | 不变 | 既有 schema/NPZ 契约测试通过 |
| T-03 | 划分、同源控制、seed | 工程通过 | 不变 | 既有测试；真实分层验收待数据 |
| T-04 | forward/loss/反向 | 工程通过 | **增强** | R-02 修复前 `pretrain_combined` batch>1 必崩；修复后全量回归含该路径 |
| T-05 | optimizer/scheduler/AMP/裁剪/累积 | 工程通过 | 不变 | 既有测试 |
| T-06 | checkpoint 全状态恢复 | 工程通过 | **增强** | N20：自监督 pretrain 两函数 checkpoint 补齐 optimizer/epoch/best/config/history + resume（此前仅主训练链路 TD-M01 闭环）；`safe_torch_load` 安全加载契约测试 |
| T-07 | best/last/early stopping 与 resume 一致 | 工程通过 | 不变（保持） | resume 语义测试（epoch 单调、best_loss 只降） |
| T-08 | 日志/manifest/config/provenance | 通过 | 不变 | checkpoint `config` 字段新增（N20）补充 provenance |
| T-09 | 训练结果可被推理消费 | 标准通过 | 不变 | 既有 train→predict 契约 |
| T-10 | 真实数据训练与科学验收 | **未通过** | **未通过（外部阻塞）** | 两项受控数据缺失（§6.1 P0） |
| I-01 | 安全加载/schema/维度校验 | 通过 | 不变 | `safe_torch_load` 新格式 checkpoint 加载断言 |
| I-02 | 单样本/批量/异常输入 | 通过 | 不变 | API 并发/异常测试 |
| I-03 | 标签/分数/版本/provenance 输出 | 通过 | 不变 | 既有契约 |
| I-04 | 初始化/健康/预测/metrics | 通过 | 不变 | e2e initialize 4 passed |
| I-05 | 安全边界/体积/限流 | 通过现有测试 | **改善** | TD-M08 方案 B：默认单 worker 使限流/metrics 语义全局一致；部署文档明确多 worker 前置条件 |
| I-06 | 跨尺度 API 预测 | **未实现** | **未实现（产品决策阻塞）** | 无 model_type 路由；§6.1 P0 |

### 7.3 是否满足总体 E2E 要求

分层结论（与修复报告 §5.2 一致，本轮增量标注）：

1. **标准训练/推理/API 工程链路满足 E2E 要求**：训练 smoke、checkpoint 消费、预测 CLI、
   API 初始化/并发/健康检查/安全门禁均通过；本轮 N20 补齐自监督训练链路的 checkpoint
   状态完整性（T-04/T-06 增强），TD-M08 收敛部署默认值（I-05 改善）。
2. **跨尺度离线工程链路 smoke-ready**：输入解析、真实 ESM-2 编码、失败台账、manifest、
   resume、Dataset 消费契约齐备（v2.0 TD-H02 交付，本轮回归确认无破坏）。
3. **项目整体尚未满足全部 E2E/科学要求**：T-10 受 replogle/scgenescope 两项受控数据与
   科学验收基线阻塞；I-06 未实现且产品范围未确认——两者均为外部/决策项，非代码缺陷。

### 7.4 残余问题与根本原因分析

| ID | 问题 | 根本原因 | 代码侧现状 |
|---|---|---|---|
| TD-C01（残余） | replogle/scgenescope 无 path/SHA/license | 数据许可与交付不在代码权限内；NPZ 链路本身完整（`CrossScaleNPZDataset` fail-fast 正常触发） | manifest 已登记其余 6 项，缺口 2 项纯属数据资产 |
| I-06/TD-H01 | 跨尺度 API 未实现 | 产品是否纳入服务边界未决策；`initialize.py` 固定 `PTM2CellNet.from_config` 无 model_type 路由 | 建议契约已在 v2.0 §8 定义，未越权实现 |
| TD-H02 余项 | ankh39 未验证、650M 权重 `.part`、全量组装未闭环 | ankh 运行包缺失、权重下载中断、图/细胞数据契约未齐 | 预计算器已支持三 backbone；环境/资产就绪即可验证 |
| GSE90546 下游 | 解析产物未接入最终 manifest/组装 | 下游组装依赖 TD-C01 数据契约与标签版本冻结 | 解析器已完成（v2.0），3/3 实验 parsed |
| N04 主体 | ensemble 双实现未合并 | 约 1.5d 工程 + CLI 兼容性确认 | 快照基线已建立（本轮），合并可安全执行 |
| N13 | 安装包名 `src` 命名空间污染 | 历史 src-layout + `find_packages()` 未显式包列表 | 需独立分支 + 发布决策（约 2.0d） |
| N07/N08 | iterrows 热路径 / UniProt 同步轮询 | 性能债，无功能缺陷 | 未启动（排期 08-25~08-29） |

---

## 8. 未解决问题与后续解决策略（含时间节点）

依据修复报告 §6 表逐项更新（日期为排期节点，非已完成事实）。标注 **[代码侧]** 项可由
工程独立推进；[外部] 项依赖交付/决策。

| 优先级 | 问题 | 策略 | 实施步骤 | 时间节点 | 责任人 |
|---|---|---|---|---|---|
| P0 | TD-C01 残余：replogle/scgenescope [外部] | 数据负责人交付授权快照（修复报告 §6 同项） | ① release/license/owner 表（0.5d）；② 固定快照 + SHA-256 登记 manifest（1.0d）；③ 两 profile 验收附件（0.5d）；④ 真实训练前 schema/哈希验收（0.5d） | 2026-08-17 ~ 08-21 | 生物信息数据负责人 |
| P0 | 跨尺度 API I-06 [外部] | 产品决策 → 若纳入则实施（v2.0 §8 接口契约） | ① 冻结 `model_type`/artifact/schema（0.5d）；② loader/状态（1.5d）；③ 单样本/批量 route（1.5d）；④ 资源限制/错误映射（1.0d）；⑤ 集成测试 + OpenAPI（1.0d） | 决策 08-19 前；实现 08-20 ~ 08-27 | 产品负责人 + 后端 |
| P1 | TD-H02 三 backbone 生产 [外部+代码侧] | ankh 环境锁定；权重恢复；全量组装 | ① 安装并锁定 ankh 版本 + 冒烟（0.5d）；② 恢复 650M 权重并校验 SHA（0.5d，网络）；③ 三路全量 cache→NPZ 组装与 `[N,L,D]`/mask/provenance 核对（1.5d） | 08-18 ~ 08-25 | 数据工程 |
| P1 | GSE90546 下游接入 [代码侧] | 解析产物登记 + 组装接线 | ① 固定 gene/perturbation 映射版本（0.5d）；② per-GSM NPZ + 聚合 TSV 登记 manifest（0.5d）；③ 跨尺度组装与真实标签分层测试（1.0d） | 解析器已完；接入 08-18 ~ 08-22 | 数据工程 |
| P1 | **[代码侧]** N04 ensemble 合并主体 | 以本轮快照为基线下沉实现 | ① `ModelEnsemble` 聚合语义迁移至 `src/models/ensemble.py` 或适配复用（1.0d）；② CLI 保持兼容（0.5d）；③ 快照测试迁移/扩展（0.5d） | 08-18 ~ 08-22 | 后端 |
| P2 | **[代码侧]** 其余 0 测试 CLI 快照（evaluate/homology_split 等） | 按 N03 模式逐个补齐 | ① 行为快照（0.5d/个）；② 失败输入与 checkpoint 兼容用例（0.5d/个） | 08-21 ~ 08-26 | 后端 |
| P2 | **[代码侧]** N13 包名改造 | 独立分支显式包列表 | ① src-layout + `packages=` 显式列表（0.5d）；② 全量 import 替换（1.0d）；③ editable install + 发布回归（0.5d） | 08-24 ~ 08-28 | 发布负责人 |
| P2 | **[代码侧]** N07/N08 性能 | benchmark 先行 | ① multitask iterrows 热路径向量化（1.0d）；② UniProt 指数退避（1.0d）；③ 指标/限额不变回归（0.5d） | 08-25 ~ 08-29 | 研究工程 |
| P2 | **[代码侧]** 仓库卫生 N16/N17/N18 | 可恢复归档 + 归属确认 | ① 根目录事故文件（0.5d）；② experimental 脚本移除（0.5d）；③ `.planning/` 归档决策（0.5d） | 08-19 ~ 08-21 | 发布负责人 |
| P2 | **[代码侧]** 部署级验证（TD-M08 后续） | docker 实构 + 单 worker 行为验收 | ① 有 docker 环境时实构镜像（0.5d）；② 单 worker 限流/metrics 冒烟（0.5d） | 按部署窗口 | 部署维护 |

### 8.1 立即行动顺序

1. 产品边界决策（跨尺度 API）——阻塞 I-06 一切工作；
2. 数据负责人交付 replogle/scgenescope——阻塞 T-10 与 P0-03；
3. ✅ 本轮已闭环 N20、TD-M08 方案 B、N03/N04 快照基线（原行动项 4/5 的代码侧部分）；
4. N04 合并主体（快照已就绪，可直接执行）；
5. GSE90546 下游接入（解析器已完）。

---

## 9. 验证证据汇总（全部本机实际执行）

| 命令 | 结果 |
|---|---|
| `pytest tests/unit tests/test_*.py -m "not slow and not gpu"` | **1896 passed, 4 skipped**（本轮基线 1871 → +25） |
| `pytest tests/integration` | **104 passed, 1 skipped**（+2 新增 pretrain_resume） |
| `pytest tests/e2e/test_api_initialize_checkpoint.py` | **4 passed** |
| 同口径 `--cov=src --cov-branch --cov-fail-under=74` | **1896 passed, 75.10%**（门禁 74% 通过） |
| `python -m mypy src --show-error-codes` | 133 files, 0 errors |
| `python -m ruff check <本轮修改文件>` | All checks passed |
| `python -m compileall -q src scripts tests` | 通过 |
| `git diff --check` | 通过 |
| `python scripts/validate_data_manifest.py --profile standard_training --check-files --verify-hashes` | exit 0（ok=true） |
| 同命令 `--profile cross_scale_training` | exit 2（设计 fail-fast）；仅 datasets[10]/[11]（replogle/scgenescope）缺 path，未新增缺口 |
| R-02 复现（修复前） | batch_size=4 `pretrain_combined` 抛 4 维 tensor RuntimeError |

**本轮变更文件**（9 个）：`src/training/self_supervised.py`、`Dockerfile`、
`configs/production.yaml`、`docs/guides/deployment.md`、`tests/unit/test_self_supervised.py`、
`tests/unit/test_docker_deployment.py`、新增 `tests/integration/test_pretrain_resume.py`、
`tests/unit/scripts/test_ensemble_predict.py`、`tests/unit/scripts/test_data_statistics.py`
（+399/-28 行于 6 个已跟踪文件，3 个新测试文件 27 个新测试）。用户未提交工作区修改保持原样。

---

## 10. 置信度与结论

**置信度：高（代码回归）/ 中（真实数据科学结论）**。全部测试/coverage/静态检查/数据契约
均为本机实际执行；每轮迭代均有单元 + 集成测试支撑；R-02 以失败复现、修复后通过闭环。

**结论**：
1. 修复报告 §6 未解决表中代码侧可独立闭环项已推进：N20（checkpoint 完整性 + resume，
   含 R-02 新缺陷修复）、TD-M08 方案 B（单 worker 默认 + 语义文档）、N03/N04 第一步
   （行为快照回归基线）；
2. 全量回归：单元+根级 1896 passed（+25），集成 104 passed（+2），coverage 75.10% 过门禁，
   mypy/ruff/compileall/diff 全绿；数据契约缺口保持 2 项（replogle/scgenescope），未新增；
3. 标准链路满足工程级 E2E 技术要求（T-01~T-09、I-01~I-05），且本轮在 T-04/T-06/I-05
   三处增强；跨尺度离线链路 smoke-ready 保持；
4. 真实科学验收（T-10）与跨尺度 API（I-06）仍受外部数据交付与产品决策阻塞，非代码缺陷，
   策略与时间节点见 §8；
5. 未越权实现未经确认的功能（跨尺度 API、受控数据 loader），与修复报告 §7 边界一致。

---

## 11. 相关文档引用索引

| 引用 | 章节 |
|---|---|
| `project_repair_report_20260816.md`（根目录，依据文档） | §1.2 口径、§2.2 严重程度、§3 三轮迭代、**§5.1 T-01~T-10/I-01~I-06 判定表**、§5.2 E2E 分层结论、**§6 未解决问题与后续策略表**、§6.1 执行顺序、§7 风险边界、§8 交付物 |
| `archive/20260816/reports/project_repair_report_20260809.md` | §8 未解决问题与后续解决策略（TD-M01/M02/M08 方案表）、§7.5 E2E 结论（历史基线） |
| `project_analysis_20260816.md` v2.0（被本报告取代） | §4 N01/R-01、§5 依赖治理、§6 TD-H02、§8 未解决问题排期（N20/TD-M08/N03/N04/N13 等） |
| `docs/E2E训练与推理现状分析_2026-08-08.md`（已归档） | §2.1 T-01~T-10、§2.2 I-01~I-06（判定基准来源） |
| `src/api/middleware.py` | `_RateLimitState` 进程内限流实现（TD-M08 依据） |
| `data/manifests/datasets.yaml` | cross_scale_training profile（datasets[10]/[11] 缺 path） |

---

**报告版本**：v3.0（2026-08-16）
**编制**：主智能体（3 轮迭代 + 系统性复核，依据最新修复报告重新执行）
**下一份权威报告**：待 P0 数据项落地、跨尺度 API 决策或 N04/N13 合并完成后刷新

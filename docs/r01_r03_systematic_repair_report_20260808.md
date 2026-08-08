# R-01～R-03 系统性修复与复核报告

- **报告日期**：2026-08-08
- **参考基线**：[`project_analysis_20260808_pre_cross_scale.md`](../archive/20260808/reports/project_analysis_20260808_pre_cross_scale.md)（历史快照）
- **当前综合结论**：[`E2E训练与推理现状分析_2026-08-08.md`](E2E训练与推理现状分析_2026-08-08.md)
- **修复范围**：R-01、R-02、R-03；R-04～R-07 仅做现状审计，不将未授权范围误报为已完成。
- **代码基线**：本地 `main` 工作区；本报告只描述 R-01～R-03 的范围，不把 Git 提交状态当作科学验收证据。

## 1. 任务判断与证据边界

本任务同时属于新功能实现、需求转代码设计、代码审查/技术债治理、测试补充和集成验证。结论依据为：参考报告、原始需求归档、当前源代码/调用链、CodeGraph 索引、实际测试与静态检查结果。

本报告区分“工程契约已实现”和“真实科学资产已验证”：没有下载外部 pLM 权重，没有导入受限图数据，也没有把随机初始化模型或 synthetic fixture 当作生物学证据。

## 2. 全量未达标项审计

| 编号 | 基线完成度 | 本次状态 | 当前缺口 |
|---|---:|---|---|
| R-01 三路 pLM | 约 25%（缺 Ankh39 和融合） | **工程契约约 90%** | 真实 Ankh39/ESM-2/ProtT5 权重与科学验收环境未提供；不能宣称复刻原始论文权重 |
| R-02 PTM Adapter/tokens | 约 80%（组件存在但契约未闭合） | **工程契约约 95%** | PTM-Mamba 原始权重/训练方案未提供；当前是项目内可复现等价 adapter |
| R-03 CIGNN/PPI 信号桥 | 约 25%（启发式 pathway mapping） | **工程契约约 90%** | 真实 STRING/BioPlex/激酶-底物图、图版本和生物学因果验收未提供；dense sensitivity 对大图有限制 |
| R-04 扰动表达/状态头 | 约 35% | 约 65% 工程接口 | 已有 `CellGraphCompassHead` 等价 decoder，但没有真实扰动训练和 Cell-Graph-Compass 权重验收；本次不扩展为独立 R-04 任务 |
| R-05 指定数据源 | 约 35% | 约 45% | manifest 和本地导入契约已存在，PTMAtlas、ProteomeTools/PROSPECT、scPerturb、scGeneScope 等仍受控或未接入完整 loader |
| R-06 图数据覆盖 | 约 40% | 约 50% | 图 schema 支持 typed edge，但 BioPlex/RegNetwork 等真实 release loader/版本校验仍需数据许可与资产 |
| R-07 PMADS Ridge | 未实现 | **已具备离线基线** | 真实 PMADS refresh 和复杂模型同 split 生物学比较仍待数据所有者提供资产 |

R-04～R-06 是当前明确的残余需求差距；它们没有被本次 R-01～R-03 交付物隐式宣称为真实科学功能。R-07 相关 manifest/baseline 是当前工作区已有的配套交付，本报告对其做回归验证。

## 3. 对照参考报告第 7 节的修复策略

已落实的实施建议如下：

1. 冻结蛋白节点、PTM 位点、信号图、细胞图、表达/状态 target 的 shape 和错误处理契约。
2. `MultiPLMEncoder` 统一 Ankh39、ESM-2、ProtT5 的预计算 embedding 和本地 Hugging Face 权重入口，投影到共同维度，默认冻结权重；缺失资产默认 fail-fast，fallback 必须显式开启。
3. `PTMTokenAdapter` 将 1-based PTM type/position 转为 residue-aligned token field；活动位点类型、位置、mask 不再静默裁剪。
4. `CIGNNSignalBridge` 使用显式 `[2, E]` source→destination 图、edge weight/type 和可微消息传递，公开 `S=(1-alpha)(I-alpha G')^-1`；默认禁止全连接近似。
5. `CellGraphCompassHead` 接收 signal-to-gene map 和 cell graph，输出 masked `delta_expression`、cell-state logits、loss 和 metrics。
6. 通过固定 fixture 覆盖维度、梯度、敏感度矩阵不变量、PTM 边界、配置/checkpoint round-trip 和 CPU benchmark。
7. forward provenance 记录模型版本、manifest 路径/摘要、图版本/摘要、seed、pLM asset status、PTM adapter 状态、fallback 和 approximation 状态。

## 4. 文件级交付物

| 文件 | 修改/新增内容 | 影响范围 |
|---|---|---|
| [`src/models/cross_scale.py`](../src/models/cross_scale.py) | 三路 pLM 融合、统一 raw-sequence residue path、图桥、敏感度矩阵、cell decoder、配置和 provenance；严格校验边/掩码/lazy 参数 | 仅 opt-in `CrossScalePTM2CellNet`，不改变默认 `PTM2CellNetBase` |
| [`src/models/ptm_modules.py`](../src/models/ptm_modules.py) | 新增 `PTMTokenAdapter`，独立处理 PTM tokens 和 1-based 位点 scatter | 可被跨尺度模型复用；原有 PTMModule 接口保持兼容 |
| [`src/models/__init__.py`](../src/models/__init__.py) | 导出 `PTMTokenAdapter` 及跨尺度组件 | 包级公共导入 |
| [`configs/cross_scale/smoke.yaml`](../configs/cross_scale/smoke.yaml) | 固化三路模型标识、required/freeze 策略和 sensitivity 上限 | opt-in smoke/config round-trip |
| [`tests/unit/models/test_cross_scale.py`](../tests/unit/models/test_cross_scale.py) | PTM token、raw sequence、typed edge、公式/梯度、mask、config policy 回归 | R-01/R-02/R-03 单元门禁 |
| [`tests/integration/test_cross_scale_pipeline.py`](../tests/integration/test_cross_scale_pipeline.py) | 完整 protein/PTM→graph→cell forward/loss/backward/checkpoint/provenance | 跨尺度集成门禁 |
| [`docs/r01_r03_systematic_repair_report_20260808.md`](r01_r03_systematic_repair_report_20260808.md) | 本次审计、三轮修复和残余风险记录 | 需求追踪和复核入口 |

工作区已有的 [`src/data/data_manifest.py`](../src/data/data_manifest.py)、PMADS Ridge、manifest CLI 和技术总结作为 TD-02 配套能力被保留并纳入回归；没有用受限数据替代真实 loader。

## 5. 三轮实际迭代记录

### 第 1 轮：pLM 与 PTM token 契约

- **识别**：PTM 是组合模型内临时 scatter；活动位点越界/未知类型会被裁剪；raw sequence 只能产生一个 pooled node，无法定位 PTM；mask 归一化后可能被 LayerNorm bias 污染。
- **方案与修改**：新增 `PTMTokenAdapter`；raw sequence 改为 residue-level padded nodes；显式 special-token/padding mask；pLM 缺失资产和 tokenizer residue 数量不匹配时 fail-fast；冻结 pLM 时保持 eval。
- **验证**：`python -m pytest -q tests/integration/test_cross_scale_pipeline.py tests/integration/test_cross_scale_benchmark.py tests/unit/models/test_cross_scale.py` —— **13 passed**；compileall、ruff、mypy 通过。

### 第 2 轮：图传播与配置策略

- **识别**：typed `edge_type` 会把浮点标签静默转成整数；错误边 shape 可能在空图路径被吞掉；node/gene mask 不校验范围；`from_config()` 丢失 pLM model names/required/freeze；注入 encoder 的 fallback provenance 可能与真实策略不一致。
- **方案与修改**：要求 edge/type 使用整数 dtype；提前验证边 shape、权重和 mask；配置完整转发并以实际注入 encoder policy 为 provenance 来源；补充 edge-weight 梯度与显式公式断言。
- **验证**：跨尺度单元 **13 passed**、集成/性能 **2 passed**；ruff 和 mypy 均通过。

### 第 3 轮：资产、checkpoint 与最终质量门禁

- **识别**：三路 pLM 缺少统一加载入口；ProtT5 输入约定未显式处理；manifest/graph/fallback provenance 不完整；部分 pLM 维度产生 LazyLinear 时 `get_model_info()` 在 forward 前崩溃。
- **方案与修改**：新增 `load_pretrained_backbones()`、ProtT5 空格化 tokenizer、asset status、manifest SHA-256、graph digest 字段、`max_sensitivity_nodes` 配置；对未初始化参数显式返回计数，不伪造参数量。
- **验证**：跨尺度测试与当前回归测试通过；本次最终全仓库 `pytest` **1897 passed、13 skipped、28 warnings**；ruff、mypy、compileall 通过。

## 6. 最终验收矩阵

| 要求 | 证据 | 结论 |
|---|---|---|
| R-01 三模型统一编码 | `MultiPLMEncoder`、三路 fixture、加载入口、residue alignment、asset status 测试 | **工程实现通过；真实权重未验证** |
| R-02 PTM Adapter/tokens | `PTMTokenAdapter`、1-based 位点校验、raw/precomputed 两条路径、输出 token/provenance 测试 | **工程实现通过；原始论文训练等价性未验证** |
| R-03 CIGNN/敏感度矩阵 | `CIGNNSignalBridge`、typed graph、edge gradient、显式公式、cell integration 测试 | **工程实现通过；真实图和生物学因果有效性未验证** |
| 默认模型兼容性 | 全量 pytest 中既有 PTM/DAVF/API/E2E 通过；跨尺度为 opt-in | **通过** |
| 质量门禁 | `ruff`、`mypy src`、`compileall`、`git -c core.whitespace=cr-at-eol diff --check` | **通过** |

## 7. 实际验证命令与结果

```text
python -m pytest -q
1910 collected; 1897 passed, 13 skipped, 28 warnings; 466.34s

python -m ruff check src scripts tests
All checks passed!

python -m mypy src --show-error-codes
Success: no issues found in 133 source files

python -m compileall -q src scripts tests
exit 0

python -m pytest -q tests/unit/models/test_cross_scale.py tests/integration/test_cross_scale_pipeline.py tests/integration/test_cross_scale_benchmark.py
16 passed

python scripts/validate_data_manifest.py --manifest data/manifests/datasets.yaml --check-files --verify-hashes
ok=true; dataset_count=15; errors=[]; warnings=[]

python scripts/benchmark_cross_scale.py --iterations 5 --warmup 1 --threads 1
CPU fixture; parameters=7094; mean=1.1680 ms; p95=1.2544 ms; RSS delta=0.125 MB
```

真实资产测试的 13 个 skip 是环境门控结果；它们不证明真实 pLM、外部图、CPTAC 或服务已经验收。跨尺度模型也没有替换默认 API/训练路径。

## 8. 复核结论与后续边界

对本次明确范围 R-01～R-03，系统性复核未发现剩余阻塞、严重或中等级工程缺口，因此不启动额外三轮范围外修复。R-04～R-06、依赖 lock、多 worker 限流、DAVF 真实资产和弃用警告等残余项仍按参考报告记录，不能因本次 R-01～R-03 通过而被标记为全部清零。

当前交付物的正确表述是：**提供一个 opt-in、可测试、可复现契约的项目内跨尺度等价模型；真实资产与生物学结论仍需授权数据、权重、图版本和独立验收。**

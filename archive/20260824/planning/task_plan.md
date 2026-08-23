# PTM2CellNet 代码修复任务计划

## 目标
依据 `project_analysis_20260816.md` 的“未解决问题与后续解决策略”，完成至少三轮完整修复迭代（问题识别、方案设计、代码修改、单元测试、集成测试），随后复核 E2E 训练/推理并生成 `project_repair_report_20260816.md`。

- [x] 基线与分析：读取最新分析文档、提取未解决项、建立当前测试基线
- [ ] 迭代一：N04 ensemble 公共聚合复用并完成单元/集成验证
- [ ] 迭代二：GSE90546 产物 manifest catalog 并完成单元/集成验证
- [ ] 迭代三：multitask 热路径向量化并完成单元/集成验证
- [ ] 系统复核与交付：E2E 训练/推理审计、遗留策略、报告生成与校验

## 约束与决策
- 使用当前工作区真实状态；不把已有历史报告中的结论当作本轮已验证证据。
- 每轮必须保留问题识别、设计、改动、单测、集成测试、遗留问题记录。
- 优先处理分析文档中可由代码独立闭环的高风险项；外部资产/产品决策项只记录，不伪造完成。
- 代码修改保持最小范围；每次修改后运行针对性单元测试与集成测试。
- 迭代一复用聚合行为但保持 `ModelEnsemble` 的加载和 CLI 签名；迭代二只补产物 provenance，不修改解析数学；迭代三只替换 pandas 行迭代，不改变样本顺序/抽样语义。

## 错误记录
| 错误 | 尝试 | 处理 |
|---|---:|---|
| `tests/unit/models/test_ensemble.py` 不存在 | 1 | 改用 `tests/unit/scripts/test_ensemble_predict.py` 与现有集成测试；通过 |

## 文件变更记录
| 阶段 | 文件 | 变更 |
|---|---|---|
| 基线 | `findings.md`, `progress.md` | 记录分析章节、代码证据和基线测试 |
| 迭代一 | 待定 | 待实施 |
| 迭代二 | 待定 | 待实施 |
| 迭代三 | 待定 | 待实施 |

---

## 2026-08-22 DAVF × PerturbGen 双路径整合

- [x] 重新审阅 v2.0 “可直接执行的阶段计划”并建立 Gate 证据
- [x] M1：契约、数据 preflight、严格基因词表
- [x] M2：隔离 runner、配置、manifest/resume、双路径 CLI（mocked）
- [x] M3：rescue/null/FDR/严格 AND/报告（mocked）
- [ ] M0：真实权重、Python 3.11 环境、真实 cohort、离线 smoke（外部资产缺失）
- [ ] M4：DAVF runtime 注入、重训与 Gate-E（受 Gate-0 阻断）
- [ ] M5–M7：真实 smoke、科学验收、移除 Geneformer、归档

停止条件：真实 checkpoint key 与 vocabulary row 未验证前，不修改 DAVF runtime，不删除 Geneformer，不声称统一底座完成。

### 2026-08-22 补充收口

- [x] M2：上游 tokenise 公式产物、动态 checkpoint/h5ad 唯一发现、跨阶段 artifact 引用与 hash 防篡改
- [x] M5 验证基础：benchmark JSON 与显式 opt-in real-assets smoke
- [x] M3 可执行入口：真实 h5ad/manifest → PathResult → 双路径候选报告 CLI
- [x] M5 正式门禁：峰值显存/多轮隔离 benchmark、release evidence 硬门、独立 GPU workflow
- [ ] Gate-4：真实资产/GPU evidence（受 Gate-0 缺失阻断）
- [ ] M6–M7：科学验收、Gate-E/Gate-5、移除 Geneformer 与归档

# 2026-08-27 综合审计任务计划

## 目标

完成项目代码、现行需求/设计文档、检测报告和技术债的证据化复核；生成
`project_analysis_20260827.md`，将确认过时的活动文档/报告归档到
`archive/20260827/`，并完成本地版本控制与可行验证。

## 执行项

- [completed] 盘点仓库、现行需求、报告、归档目录和 Git 状态。
- [completed] 并行复核需求差异、代码集成链路、技术债与文档新鲜度。
- [completed] 汇总引用、生成流程图、记录未实现/部分实现和技术债策略。
- [completed] 归档确认过时文件，写入清单与版本信息。
- [completed] 提交修改，确认 `main` 与 `origin/main` 的同步关系，执行编译/测试/静态检查。
- [completed] Review 查 Bug，复核报告链接和结论，更新经验记忆。

## 当前基线

- 执行日期：2026-08-27。
- 工作区初始状态：干净，当前分支为 `main`。
- 复核前 HEAD：`a8c480ca97a9902e7f34871358102b18bfebdd3f`。
- 复核前远程 `origin/main`：`c76fff881f268f9bd0b39d68db8ba547115ec4cf`。
- `git rev-list --left-right --count main...origin/main`：`43 0`，本地包含远程全部提交且额外领先 43 个提交。

## 最终结果

- 审计内容提交：`f0be1abbceca7105d5605bb9d8c989c619a71ea7`。
- 后续审计元数据提交：`fce11aebb30e8d339d8c94aca43cdc73a68b2552`、`3656921c401e5000b74446d080e7779cb35a8b44`、`b1a315572ecd2bdd2b6dfe013e117ec8f81d20df`。
- 最终复核：`git merge --ff-only origin/main` 输出 `Already up to date.`；最终 `main...origin/main` 为 `47 0`；工作树干净。
- 最终验证：compileall、Ruff check、pytest collect（2345）和 requirements consistency 均通过；全量离线回归为 2329 passed / 16 skipped / 47 warnings。

## 2026-09-01 续作收口

- [completed] Review DAVF → PerturbGen 方向门控、API/CLI batch 边界和 legacy checkpoint 错误路径。
- [completed] 提交代码与测试：`19024a1`（前一版本 `b1a3155`）。
- [completed] `git fetch origin main` 与 `git merge --ff-only origin/main`；输出 `Already up to date.`，关系 `origin/main...main = 0 48`。
- [completed] 完整编译、Ruff check、定向回归和全量离线回归。
- [completed] 生成 `project_analysis_20260901.md`，归档旧报告至 `archive/20260901/`，更新活动文档链接。

### 续作验证基线

- 全量：2373 collected，2357 passed / 16 skipped / 47 warnings，536.65s。
- 定向：107 passed / 1 skipped / 15 warnings，155.87s。
- compileall 与 Ruff check 通过；mypy 5 errors、pip check 3 conflicts、format check 298 files 待格式化，均已写入当前报告。

## 2026-09-10 综合复核收口

- [completed] 5 组逻辑提交合入 main（LatentDAVF 管线 / GSE 队列 / IBD QC / orchestrator E2E / 清单状态），ff-only 无冲突。
- [completed] compileall + ruff + 全量离线回归（2475 passed/0 failed）+ mypy/pip check/requirements 验证并记录。
- [completed] 归档 8 个过时报告至 `archive/20260910/`，生成 MANIFEST/README，更新 index.rst 与活动文档链接。
- [completed] 3 个并行子代理对抗审查（未实现项 / 部分实现 / 技术债），发布 `project_analysis_20260910.md`（含子代理统计章节）。

### 20260910 验证基线

- 全量：2475 passed / 15 skipped / 7 deselected / 54 warnings，729.03s，exit 0。
- compileall、ruff check、requirements consistency 通过；mypy 80 errors（TD-NEW-16）；pip check 3 conflicts（既有）。

## 2026-09-13 架构澄清与研究闭环

上述 2026-09-10 的完成项仍然成立，但它们表示工程契约、桥接和检查完成，不能直接解释为正式生物学验收。2026-09-13 的 U-01～U-07 修复已补齐若干代码接口；本节把现状、目标、待办和验收条件分开记录。

### 当前现状

- 候选准入链为 `PTM proposal/candidate_spec → scVI/PTMDirectionMapper → DAVF decode 方向 → 三方方向 gate → CandidateEvidence → PerturbGenInvocation`。PTM classifier 只预测 site presence；candidate 的 proposed direction 来自用户假设或逐 site override。当前 E2E 消费候选 JSON 的目标与方向，不从原始 site 自动推断表达或因果方向。
- 准备/运行链是六阶段 `tokenise → train_mask → train_decoder → perturb → export_gene_embeddings → report`。`source_intervention=[src]` 是状态转移前场景，`within_state=[tgt]+pert_tps` 是目标状态内场景；`source_intervention/within_state` 是 PerturbGen 内部两个实验场景，与桥接/准备执行的职责分层及 Workflow A/B 资产生命周期划分分别讨论。当前每个候选重做 `tokenise/train_mask/train_decoder`、两路径 `perturb` 和 `export_gene_embeddings/report`，跨候选 reuse 只是目标，不能写成已实现。
- 统计验收是独立终点：report 只汇总 `stage_manifest`，E2E 目前不会自动接续匹配 null、候选 empirical-p/q、未扰动质量和 dual-path 统计。U-01～U-05 的生成、聚合、formal 隔离、质量提取和 donor split 接口已在代码中闭合，但真实运行与自动接续仍待完成；普通 CLI 仍需通过现有 `--e2e-gate-report`/invocation 边界进入 runner。
- 观测方向是 donor-level disease−normal，DAVF 方向是 `decode(z_intervened)−decode(z_context)`。当前 gate 直接比较 up/down，统一参考轴仍待定义；正式记录必须带 context、intervention、比较基准和研究目标。病程一致性、状态逆转、治疗因果性不能混称。三路同号不自动构成三个独立证据，来源复用和 cohort/donor/训练划分必须披露并留证。
- encoder → 静态冻结 embedding asset → LatentDAVF 重训/Gate-E 属于独立资产生命周期（Workflow B）；gate → PerturbGen utility 属于 Workflow A。`export_gene_embeddings` 读取固定基础 encoder checkpoint，不消费候选结果或回灌当前 DAVF。当前正式 normal/disease cohort 为 0 个合规候选；正式合并使用 canonical Ensembl ID，PerturbGen token index 与 scVI decoder index 不得混用。

### 研究目标

- 固定正式 cohort、词表、训练配置和资产版本完成一次公共准备；候选运行阶段只做 perturb 和评估。是否引入跨候选复用需后续方案和代码共同确认，当前不新增 hash、调度框架或任务开关。
- 在统一参考轴上分别评估 source/within-state 场景；正式 dual-path 采用 AND。单路通过只表示对应场景或探索结果，不表示普遍治疗效用，也不构成正式双路 PASS。
- 让统计验收只消费带 provenance 的真实 null、未扰动质量和效用结果；工程/synthetic 结果与正式科学结果分开。

### 待实现/待验证

- [ ] 提供满足 Gate-0 的真实 `normal/disease` raw-count cohort：canonical Ensembl、显式 donor、至少 3 个共享 donor，并登记 manifest。
- [ ] 为每次方向证据固定记录 context、intervention、比较基准和研究目标，明确是在检验病程一致性还是状态逆转；保留 proposal、observed 和 DAVF 的来源标签。
- [ ] 用真实 cohort 完成训练-only/held-out donor 划分，证明与 frozen manifest 一致；历史 checkpoint 未绑定该证明的不能追认。
- [ ] 披露独立表达观测、DAVF 训练 cohort/donor 与外部方向假设的复用关系；三路同号不能在缺少划分证明时写成三个独立证据。
- [ ] 让 E2E 在现有 invocation/runner 边界内接续 matched-null 生成、未扰动质量提取、候选 p/q 和 dual-path 统计；不把底层 `StagePlan` runner 自检描述为全局 gate。
- [ ] 完成真实多 seed、双场景和 ≥99 matched-null GPU 运行，随后执行 Gate-E ≥200、M6/OE 及 formal evidence；在此之前保留 Geneformer。

### 验收条件

只有同时具备三方方向证据、真实 donor 隔离证明、冻结 scVI gene order/embedding manifest、真实 null/质量/效用统计和双路径 AND 结果，才可进入正式 PASS。synthetic、smoke、bridge 或单路通过只能作为工程/探索结果；不得据单个 gene 方向宣称治疗因果性。

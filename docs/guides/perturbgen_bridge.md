# PerturbGen 桥接指南（DAVF × PerturbGen 双路径整合）

> **文档版本**：v1.2（2026-09-01，补充 DAVF 方向 gate 与生产接线边界）
> **权威方案**：[`docs/DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md`](../DAVF_PerturbGen_双路径整合方案与测试方案_2026-08-21.md)（v2.0）
> **状态基线**：[`project_analysis_20260901.md`](../../project_analysis_20260901.md)（本报告按代码闭合度评估 DAVF 方向推理 77.0% / PerturbGen runner 68.0%，真实资产与 Gate 另计）

本指南面向需要运行 PerturbGen 训练/扰动链路或 DAVF 嵌入底座迁移的操作者，
给出环境、数据契约、六阶段 pipeline、嵌入资产与评估的入口命令。
**所有命令均为当前仓库真实存在的脚本与参数**（未实现的项明确标注）。

---

## 1. 架构一页纸

两条工作流（方案 §4.1）：

| 工作流 | 内容 | 入口 |
|---|---|---|
| A：PerturbGen 双路径扰动模拟 | candidate → preflight → 独立环境训练 → `src`/`tgt` 扰动 → rescue/null/donor 统计 → PASS/FAIL/INCONCLUSIVE | `scripts/run_perturbgen_pipeline.py` |
| B：DAVF 底座迁移 | PerturbGen encoder ckpt → 静态 gene embedding 资产导出 → schema v2 注入 → DAVF 重训 → Gate-E → 删除 Geneformer 主链 | `scripts/export_perturbgen_gene_embeddings.py` + `scripts/finetune_davf_e2e.py --embedding-asset` |

主进程与独立环境边界（方案 §4.2，硬约束）：

- 主项目环境（如 `SSH_unit`）：输入契约、路径安全、symbol↔ENSG 解析、配置生成、manifest、stage 调度、输出 schema 校验、统计与报告。
- PerturbGen 独立环境（conda env `perturbgen`，Python 3.11）：官方 tokenisation、masking/count decoder 训练、`src/tgt` 扰动推理、gene embedding 导出。
- **主进程绝不 `import perturbgen`**；所有跨环境调用为参数数组（禁止 `shell=True`），stage 带 timeout、退出码与输出 schema 校验。

当前 `run_perturbgen_pipeline.py` 只负责六阶段 runner 调度；DAVF 方向 gate 和
`evaluate_davf_perturbgen_candidate()` 已作为严格库契约实现，但尚未由该 CLI
自动调用。因此正式 candidate manifest 仍需按未实现项补齐，不能把 mocked
pipeline 输出当成方向 gate 或真实科学闭环。

## 2. 环境准备（一次性）

```bash
# 1) 独立 Python 3.11 环境（复用已存在的 perturbgen env；绝不触碰共享环境）
bash scripts/setup_perturbgen_env.sh

# 2) 离线 wheel 预下载（网络受限机器转机安装用）
bash scripts/download_perturbgen_wheels.sh

# 3) 环境证据采集（GPU/CUDA/包版本快照，写入 outputs/ 供 manifest 引用）
python scripts/collect_perturbgen_env_evidence.py
```

前提：`ref/Perturbgen-src`（浅克隆）与用户手动放置的 PerturbGen encoder
权重（HuggingFace `lotfollahi-lab/PerturbGen`）。**agent 不得自动拉取大权重**。

## 3. 数据契约（Gate-0，先于一切训练）

donor cohort 硬要求（方案 §4.6-1；lessons.md L-2026-0822-06）：

1. 显式 `donor`/`patient` 列（`sample`/`batch`/`replicate`/CRISPR control 不算）；
2. normal/disease 配对、raw counts、Ensembl ID（ENSG）、`cell_type`/`state` 元数据；
3. 目标 cell type 每状态 ≥3 个可评估 donor。

```bash
# 审计本地 h5ad 是否有合规候选（结果写 outputs/perturbgen/spike/<date>_donor_audit/evidence.json）
python scripts/audit_perturbgen_cohort.py
```

**当前状态（2026-09-01）**：30 文件审计 0 合规候选 —— M0⑥ 是全链唯一
外部数据硬阻断（U-01）。Gate-0 未过时，M4 重训/M6/Gate-4 按方案 §7.3
有意挂起，不得跳过。

## 4. 六阶段 pipeline（工作流 A）

```bash
python scripts/run_perturbgen_pipeline.py \
  --config configs/integration/perturbgen.yaml \
  --stages tokenise train_mask train_decoder perturb export_gene_embeddings report \
  --path both \
  --dry-run \
  --gpu-lock-file /tmp/pg.gpu.lock
```

- stage 顺序固定：`tokenise → train_mask → train_decoder → perturb → export_gene_embeddings → report`；
- 产物路径按上游公式 + 唯一 glob 解析后写入 manifest（路径 + hash），**禁止
  latest-mtime 猜测**；零匹配/多匹配/hash 变化直接失败（lessons.md L-2026-0822-04）；
- 上例用 `--dry-run` 只打印计划；正式续跑时移除 `--dry-run`，需要从 manifest
  继续时再添加 `--resume`。

## 5. 嵌入资产导出与 DAVF 注入（工作流 B）

```bash
# 参数模板：执行前将尖括号替换为真实路径；tensor-key 必填，工具从不猜测 checkpoint 布局
python scripts/export_perturbgen_gene_embeddings.py \
    --checkpoint <encoder.ckpt> \
    --tensor-key <精确的 embedding 参数键> \
    --vocabulary <gene_id→row 的 JSON/pickle> \
    --output-dir outputs/perturbgen/embedding_asset/
```

产出 `gene_embeddings.safetensors + vocabulary.json + manifest.json`。其中嵌入资产
`manifest.json` 使用 `schema_version: 1`（含 sha256 与维度）；DAVF 配置文件另使用
配置 schema v2。主环境加载零 PerturbGen 依赖。

```bash
# schema v2 注入 + DAVF 重训（参数模板；缺失资产 fail-fast，不随机 fallback）
python scripts/finetune_davf_e2e.py \
  --data <ptm_training.csv> \
  --checkpoint <davf_checkpoint.pt> \
  --embedding-asset outputs/perturbgen/embedding_asset/
```

注入链路：`src/models/davf_inference.py`（`embedding_asset_path` 字段 +
sha256/schema 校验）→ `LatentDAVF(pretrained_gene_embeddings=...)`；
旧 `geneformer_path` 配置在 `architectures.py` 迁移闸门直接 ValueError。
**Gate-E 未过前不删除 Geneformer 主链（M7）**。

## 6. 评估、报告与发布证据

```bash
# 双路径评估（rescue/null/FDR/AND 判定）
python scripts/evaluate_perturbgen_dual_path.py --input-json <candidates.json> --output-dir <outdir>

# 基准测试（engineering fixture 可离线；real 需真资产）
python scripts/benchmark_perturbgen.py --config <cfg> --fixture-type engineering|real [--iterations N]

# Gate-4 发布证据校验（单次运行自洽，不接受多次残缺 benchmark 的并集）
python scripts/check_perturbgen_release_evidence.py --evidence <evidence.json> [--benchmark-json <bench.json>] [--mode formal|smoke]
```

判定口径（lessons.md L-2026-0821-01）：双路径 **AND** 标准——`src` 与 `tgt`
两路 rescue 均稳定为正（排除目标基因本身、≥3 donor 方向一致、跨 seed/mask-pad-delete
模式一致）才进实验验证候选清单。

## 7. 门禁状态速查（截至 2026-09-01）

| Gate | 内容 | 状态 |
|---|---|---|
| Gate-0 M0⑤ | 独立环境 smoke（perturb 51.88s / 1757 MiB / h5ad schema 通过） | ✅ 已过（`outputs/perturbgen/spike/20260823_m0_smoke/evidence.json`） |
| Gate-0 M0⑥ | ≥3 donor 合规 cohort | ❌ 阻断（0 合规候选，U-01，外部数据依赖） |
| Gate-1~3 | 契约 / runner / 双路径统计 | ⚠️ 工程组件完成；方向 gate→runner 自动接线仍开放 |
| Gate-E | DAVF 新底座回归（≥200 PTM 基准 + bootstrap CI + 下游非劣） | ⏸ 等 M4 重训（数据阻断） |
| Gate-4 | 真实 smoke / 正式 release evidence | ⏸ 等真资产 workflow 运行 |
| Gate-5 | 冻结队列科学验收（3 seeds / held-out / ≥99 null / BH-FDR） | ⏸ 未开始（M6） |

## 8. 常见陷阱

- **不要把 scPerturb 通用数据当 donor cohort**（L-2026-0822-06）；
- **不要混装 scgpt 与 scvi-tools>=1.2 到同一环境**（TD-M05：scgpt 需要独立环境，且已从 `requirements-lock.txt` 移除）；
- **PerturbGen 不暴露 HTTP API**（方案 §9 有意决策：长任务不进同步 API）；
- 正式证据必须可提交可留存：benchmark JSON 写 `outputs/real_assets/`，不是 pytest 临时目录（L-2026-0822-05）。

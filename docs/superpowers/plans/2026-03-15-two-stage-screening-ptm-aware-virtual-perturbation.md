# Two-Stage Screening And PTM-Aware Virtual Perturbation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 先构建一个稳定的“两阶段筛选/解释”管线，用 PTM2CellNet 排序候选 PTM/蛋白事件并交给 GenKI 做下游网络解释；随后在不重写主模型的前提下，把 GenKI 的 hard KO 扩展为 PTM-aware soft perturbation。

**Architecture:** 推荐采用“外接式编排”而不是“端到端联训”。阶段 1 保持 PTM2CellNet 主干不变，用 leave-one-PTM-out / leave-one-protein-out 的后验打分做候选筛选，再用桥接层把候选映射到 gene 并调用 GenKI 的 virtual KO；阶段 2 继续复用同一桥接层，只把 hard KO 替换成“节点活性衰减 + 网络边权缩放后重新阈值化”的软扰动。

**Tech Stack:** Python, PyTorch, pandas, numpy, PyYAML, pytest, 本地 `ref/GenKI-master-src/GenKI-master` 参考实现。

---

## Recommended Approach

### Option A: 外接式两阶段集成（推荐）
- PTM2CellNet 只负责样本级预测和候选事件排序。
- 新增桥接层负责 protein/PTM -> gene 映射、GenKI 调用、解释报告生成。
- 优点：风险最低，最符合 `docs/guides/2026-03-15-GenKI源码与PTM2CellNet整合分析.md` 的建议顺序。

### Option B: 直接把 GenKI 图分支塞入 `PTM2CellNet`
- 需要同时改 `src/models/architectures.py`、训练逻辑、数据接口。
- 优点是未来可能更强；缺点是当前没有足够监督信号，验证难度高。
- 本计划不采用。

### Option C: 先完整 vendor GenKI，再做所有改造
- 会过早引入大体量第三方代码维护成本。
- 只有在阶段 1 证明桥接价值后，才值得考虑更深 vendor。
- 本计划只包装必要接口，不先整体迁移。

## Scope And Exit Criteria

### 阶段 1 完成标准
- 可以从 PTM2CellNet 输出中得到样本级候选 PTM/蛋白事件排名。
- 可以把候选映射为 gene 列表，并驱动 GenKI 跑 virtual KO。
- 可以产出包含 candidate score、KO distance、下游 gene/pathway 排名的结构化结果。
- 至少有 1 个端到端 smoke test 和 1 份 Markdown 汇总报告。

### 阶段 2 完成标准
- 同一候选输入同时支持 `hard_ko` 和 `soft_ptm_perturbation` 两种模式。
- soft 模式至少实现两类扰动：
  - 节点表达/活性衰减。
  - 目标 gene 相关边的权重缩放，并通过重新阈值化更新 `edge_index`。
- 产出 hard vs soft 的差异报告，至少包含 distance shift、top downstream genes overlap、解释稳定性比较。

### 非目标
- 不在此轮把 GenKI 训练过程重写成 Lightning。
- 不在此轮做 PTM2CellNet 与 GenKI 的联合训练。
- 不在此轮扩展 API 服务端接口，先完成离线脚本与评估闭环。

## File Structure

### 新增文件
- `src/integration/__init__.py`
- `src/integration/contracts.py`
- `src/integration/ptm_gene_mapper.py`
- `src/integration/genki_adapter.py`
- `src/integration/ptm_virtual_perturbation.py`
- `src/evaluation/explainers.py`
- `configs/integration/two_stage_explanation.yaml`
- `configs/integration/ptm_virtual_perturbation.yaml`
- `scripts/run_two_stage_explanation.py`
- `scripts/run_ptm_virtual_perturbation.py`
- `tests/unit/integration/test_contracts.py`
- `tests/unit/integration/test_ptm_gene_mapper.py`
- `tests/unit/evaluation/test_explainers.py`
- `tests/unit/integration/test_ptm_virtual_perturbation.py`
- `tests/integration/test_two_stage_pipeline.py`
- `tests/integration/test_soft_perturbation_pipeline.py`
- `tests/fixtures/genki/mock_gene_list.txt`
- `tests/fixtures/genki/mock_network.npy`
- `tests/fixtures/genki/mock_counts.npy`

### 可能修改文件
- `src/utils/config.py`
  - 如果需要为 `configs/integration/*.yaml` 增加便捷读取或配置 merge helper。
- `src/utils/io.py`
  - 如果需要新增统一的 Markdown / JSON / CSV 结果落盘函数。
- `README.md`
  - 在最后补一段“解释/扰动脚本”的运行入口。

### 责任边界
- `src/evaluation/explainers.py`
  - 只负责候选打分、排序、解释编排，不直接处理第三方源码细节。
- `src/integration/genki_adapter.py`
  - 隔离 `ref/GenKI-master-src/GenKI-master` 的加载、入参转换、KO 执行、结果标准化。
- `src/integration/ptm_virtual_perturbation.py`
  - 只负责 soft perturbation profile 构建，不负责 PTM2CellNet 推理和报告渲染。

## Data Contracts

### CandidateRecord

```python
@dataclass
class CandidateRecord:
    sample_id: str
    protein_id: str
    ptm_type: str
    ptm_position: int
    baseline_label: str
    baseline_probability: float
    perturbed_probability: float
    delta_probability: float
```

### GenePerturbationRequest

```python
@dataclass
class GenePerturbationRequest:
    gene_symbol: str
    source_protein_id: str
    source_ptm_type: str
    source_ptm_position: int
    magnitude: float
    mode: str  # "hard_ko" | "soft_ptm"
```

### PerturbationResult

```python
@dataclass
class PerturbationResult:
    gene_symbol: str
    mode: str
    distance_score: float
    ranked_genes: list[str]
    metadata: dict[str, Any]
```

## Chunk 1: 基础桥接层与契约

### Task 1: 建立 integration 契约与最小桥接骨架

**Files:**
- Create: `src/integration/__init__.py`
- Create: `src/integration/contracts.py`
- Create: `src/integration/ptm_gene_mapper.py`
- Create: `src/integration/genki_adapter.py`
- Test: `tests/unit/integration/test_contracts.py`
- Test: `tests/unit/integration/test_ptm_gene_mapper.py`

- [ ] **Step 1: 先写契约测试，固定输入输出格式**

```python
from src.integration.contracts import CandidateRecord, GenePerturbationRequest


def test_candidate_record_delta_is_explicit() -> None:
    record = CandidateRecord(
        sample_id="S1",
        protein_id="P04637",
        ptm_type="phosphorylation",
        ptm_position=15,
        baseline_label="activated",
        baseline_probability=0.91,
        perturbed_probability=0.34,
        delta_probability=0.57,
    )
    assert record.delta_probability == 0.57


def test_gene_mapper_deduplicates_gene_symbols() -> None:
    from src.integration.ptm_gene_mapper import ProteinGeneMapper

    mapper = ProteinGeneMapper({"P04637": "TP53", "TP53_HUMAN": "TP53"})
    mapped = mapper.map_many(["P04637", "TP53_HUMAN"])
    assert mapped == ["TP53"]
```

- [ ] **Step 2: 运行单元测试，确认当前确实失败**

Run:
```bash
pytest tests/unit/integration/test_contracts.py tests/unit/integration/test_ptm_gene_mapper.py -v
```

Expected: FAIL，提示 `src.integration` 或相关类尚不存在。

- [ ] **Step 3: 实现最小 dataclass、mapper 与 adapter 壳层**

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class CandidateRecord:
    sample_id: str
    protein_id: str
    ptm_type: str
    ptm_position: int
    baseline_label: str
    baseline_probability: float
    perturbed_probability: float
    delta_probability: float


class ProteinGeneMapper:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def map_many(self, protein_ids: list[str]) -> list[str]:
        seen: list[str] = []
        for protein_id in protein_ids:
            gene = self.mapping.get(protein_id)
            if gene and gene not in seen:
                seen.append(gene)
        return seen
```

- [ ] **Step 4: 给 `GenKIAdapter` 先做“接口通、实现薄”的第一版**

```python
class GenKIAdapter:
    def __init__(self, ref_root: str) -> None:
        self.ref_root = ref_root

    def run_virtual_ko(self, gene_symbol: str, top_k: int = 20) -> PerturbationResult:
        raise NotImplementedError("实现阶段 1 时接入本地 GenKI 参考代码")
```

- [ ] **Step 5: 重新运行测试，确认契约层通过**

Run:
```bash
pytest tests/unit/integration/test_contracts.py tests/unit/integration/test_ptm_gene_mapper.py -v
```

Expected: PASS。

- [ ] **Step 6: 记录本地 checkpoint**

Run:
```bash
mkdir -p outputs/results/checkpoints
printf "chunk1-task1 complete\n" > outputs/results/checkpoints/two_stage_chunk1_task1.txt
```

Expected: 生成本地检查点文件。

### Task 2: 为 GenKI 参考实现建立稳定适配器

**Files:**
- Modify: `src/integration/genki_adapter.py`
- Test: `tests/unit/integration/test_contracts.py`
- Test: `tests/fixtures/genki/mock_gene_list.txt`
- Test: `tests/fixtures/genki/mock_network.npy`
- Test: `tests/fixtures/genki/mock_counts.npy`

- [ ] **Step 1: 写适配器测试，先只验证加载和请求路径，不跑真训练**

```python
from src.integration.genki_adapter import GenKIAdapter


def test_genki_adapter_accepts_local_fixture_root(tmp_path) -> None:
    adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
    payload = adapter.build_request(gene_symbol="TP53", mode="hard_ko", magnitude=1.0)
    assert payload.gene_symbol == "TP53"
    assert payload.mode == "hard_ko"
```

- [ ] **Step 2: 跑测试确认失败**

Run:
```bash
pytest tests/unit/integration/test_contracts.py -v
```

Expected: FAIL，提示 `build_request` 缺失。

- [ ] **Step 3: 实现三个明确接口，禁止把 GenKI 细节泄漏到上层**

```python
class GenKIAdapter:
    def build_request(self, gene_symbol: str, mode: str, magnitude: float) -> GenePerturbationRequest:
        return GenePerturbationRequest(
            gene_symbol=gene_symbol,
            source_protein_id="",
            source_ptm_type="",
            source_ptm_position=-1,
            magnitude=magnitude,
            mode=mode,
        )

    def load_reference_data(self) -> dict[str, Any]:
        ...

    def run(self, request: GenePerturbationRequest) -> PerturbationResult:
        ...
```

- [ ] **Step 4: 接入 `ref/GenKI-master-src/GenKI-master/GenKI/dataLoader.py` 的最小依赖**

Run:
```bash
python - <<'PY'
from src.integration.genki_adapter import GenKIAdapter

adapter = GenKIAdapter(ref_root="ref/GenKI-master-src/GenKI-master")
print(adapter.ref_root)
PY
```

Expected: 正常打印路径，不报 import error。

- [ ] **Step 5: 重新跑适配器测试**

Run:
```bash
pytest tests/unit/integration/test_contracts.py -v
```

Expected: PASS。

## Chunk 2: 两阶段筛选/解释

### Task 3: 实现 PTM 候选打分器

**Files:**
- Create: `src/evaluation/explainers.py`
- Test: `tests/unit/evaluation/test_explainers.py`

- [ ] **Step 1: 先写失败测试，固定“留一 PTM 置空”打分行为**

```python
import pandas as pd

from src.evaluation.explainers import LeaveOnePTMOutScorer


class MockModel:
    def __call__(self, batch):
        active_sites = int(batch["ptm_mask"].sum().item())
        positive = 0.2 + 0.3 * active_sites
        return {
            "probabilities": __import__("torch").tensor([[1 - positive, positive]]),
            "predictions": __import__("torch").tensor([1 if positive >= 0.5 else 0]),
        }


def test_leave_one_ptm_out_ranks_larger_probability_drop_first() -> None:
    scorer = LeaveOnePTMOutScorer(target_class_index=1)
    row = pd.Series(
        {
            "sample_id": "S1",
            "protein_id": "P04637",
            "sequence": "ACDEFG",
            "ptm_sites": '[{"position": 2, "type": "phosphorylation"}, {"position": 5, "type": "acetylation"}]',
        }
    )
    ranked = scorer.rank_row(MockModel(), row)
    assert ranked[0].delta_probability >= ranked[1].delta_probability
```

- [ ] **Step 2: 运行测试验证失败**

Run:
```bash
pytest tests/unit/evaluation/test_explainers.py -v
```

Expected: FAIL，提示 `LeaveOnePTMOutScorer` 不存在。

- [ ] **Step 3: 只实现最小可用候选评分逻辑**

```python
class LeaveOnePTMOutScorer:
    def __init__(self, target_class_index: int) -> None:
        self.target_class_index = target_class_index

    def rank_row(self, model, row):
        baseline = self._predict(model, row)
        candidates = []
        for site in self._iter_sites(row):
            perturbed = self._predict(model, row, drop_site=site)
            candidates.append(
                CandidateRecord(
                    sample_id=str(row["sample_id"]),
                    protein_id=str(row["protein_id"]),
                    ptm_type=site["type"],
                    ptm_position=site["position"],
                    baseline_label=str(baseline["label"]),
                    baseline_probability=baseline["prob"],
                    perturbed_probability=perturbed["prob"],
                    delta_probability=baseline["prob"] - perturbed["prob"],
                )
            )
        return sorted(candidates, key=lambda item: item.delta_probability, reverse=True)
```

- [ ] **Step 4: 补一个 protein-level 聚合器，避免只停在 site-level**

```python
def aggregate_by_protein(candidates: list[CandidateRecord]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for candidate in candidates:
        scores[candidate.protein_id] = max(scores.get(candidate.protein_id, 0.0), candidate.delta_probability)
    return scores
```

- [ ] **Step 5: 跑单测确认 site-level 和 protein-level 行为正确**

Run:
```bash
pytest tests/unit/evaluation/test_explainers.py -v
```

Expected: PASS。

### Task 4: 把候选打分接到 GenKI，产出两阶段解释结果

**Files:**
- Modify: `src/evaluation/explainers.py`
- Modify: `src/integration/genki_adapter.py`
- Create: `configs/integration/two_stage_explanation.yaml`
- Create: `scripts/run_two_stage_explanation.py`
- Test: `tests/integration/test_two_stage_pipeline.py`

- [ ] **Step 1: 先写集成测试，定义编排层输出**

```python
import pandas as pd

from src.evaluation.explainers import TwoStageExplanationPipeline


class FakeScorer:
    def rank_row(self, model, row):
        from src.integration.contracts import CandidateRecord
        return [
            CandidateRecord(
                sample_id="S1",
                protein_id="P04637",
                ptm_type="phosphorylation",
                ptm_position=15,
                baseline_label="activated",
                baseline_probability=0.91,
                perturbed_probability=0.34,
                delta_probability=0.57,
            )
        ]


class FakeMapper:
    def map_candidate(self, candidate):
        return "TP53"


class FakeGenKI:
    def run(self, request):
        from src.integration.contracts import PerturbationResult
        return PerturbationResult(
            gene_symbol=request.gene_symbol,
            mode=request.mode,
            distance_score=1.23,
            ranked_genes=["BAX", "MDM2"],
            metadata={"pathways": ["apoptosis"]},
        )


def test_two_stage_pipeline_writes_markdown_summary(tmp_path) -> None:
    pipeline = TwoStageExplanationPipeline(FakeScorer(), FakeMapper(), FakeGenKI())
    df = pd.DataFrame([{"sample_id": "S1", "protein_id": "P04637", "sequence": "ACD", "ptm_sites": "[]"}])
    outputs = pipeline.run(model=object(), df=df, output_dir=tmp_path)
    assert outputs[0].gene_symbol == "TP53"
    assert (tmp_path / "two_stage_summary.md").exists()
```

- [ ] **Step 2: 跑集成测试，确认当前失败**

Run:
```bash
pytest tests/integration/test_two_stage_pipeline.py -v
```

Expected: FAIL，提示 `TwoStageExplanationPipeline` 尚未实现。

- [ ] **Step 3: 实现编排层，严格保持依赖方向**

```python
class TwoStageExplanationPipeline:
    def __init__(self, scorer, mapper, genki_adapter) -> None:
        self.scorer = scorer
        self.mapper = mapper
        self.genki_adapter = genki_adapter

    def run(self, model, df, output_dir):
        results = []
        for _, row in df.iterrows():
            ranked = self.scorer.rank_row(model, row)
            top_candidate = ranked[0]
            gene_symbol = self.mapper.map_candidate(top_candidate)
            request = self.genki_adapter.build_request(
                gene_symbol=gene_symbol,
                mode="hard_ko",
                magnitude=top_candidate.delta_probability,
            )
            results.append(self.genki_adapter.run(request))
        self._write_markdown_summary(results, output_dir)
        return results
```

- [ ] **Step 4: 增加 CLI 脚本，复用现有脚本风格**

```python
parser.add_argument("--model", required=True)
parser.add_argument("--config", default="configs/integration/two_stage_explanation.yaml")
parser.add_argument("--input", required=True)
parser.add_argument("--output", default="outputs/results/two_stage")
parser.add_argument("--top-k", type=int, default=10)
```

- [ ] **Step 5: 创建配置文件，先把外部依赖路径写清楚**

```yaml
integration:
  genki_ref_root: "ref/GenKI-master-src/GenKI-master"
  mapping_file: "data/external/protein_gene_mapping.csv"
  candidate_strategy: "leave_one_ptm_out"
  top_k_candidates: 10
  perturbation_mode: "hard_ko"
```

- [ ] **Step 6: 跑单测和集成测试**

Run:
```bash
pytest tests/unit/integration/test_contracts.py tests/unit/evaluation/test_explainers.py tests/integration/test_two_stage_pipeline.py -v
```

Expected: PASS。

- [ ] **Step 7: 跑一次 CLI smoke test**

Run:
```bash
python scripts/run_two_stage_explanation.py \
  --model outputs/models/best_model.pt \
  --config configs/integration/two_stage_explanation.yaml \
  --input data/processed/example_candidates.csv \
  --output outputs/results/two_stage_smoke
```

Expected: 输出目录下至少出现：
- `candidate_scores.csv`
- `genki_explanations.json`
- `two_stage_summary.md`

## Chunk 3: PTM-aware Virtual Perturbation

### Task 5: 抽象 soft perturbation profile，并保持与 hard KO 并存

**Files:**
- Create: `src/integration/ptm_virtual_perturbation.py`
- Test: `tests/unit/integration/test_ptm_virtual_perturbation.py`

- [ ] **Step 1: 先写测试，固定 soft profile 的核心行为**

```python
import numpy as np

from src.integration.ptm_virtual_perturbation import PTMPerturbationProfile, apply_soft_perturbation


def test_soft_perturbation_scales_node_and_incident_edges() -> None:
    counts = np.ones((4, 3), dtype=float)
    net = np.array(
        [
            [0.0, 0.9, 0.0],
            [0.9, 0.0, 0.7],
            [0.0, 0.7, 0.0],
        ]
    )
    profile = PTMPerturbationProfile(target_gene_index=1, node_decay=0.4, edge_scale=0.5)
    counts_new, net_new = apply_soft_perturbation(counts, net, profile)
    assert counts_new[:, 1].mean() < counts[:, 1].mean()
    assert net_new[0, 1] == 0.45
    assert net_new[1, 2] == 0.35
```

- [ ] **Step 2: 跑测试确认失败**

Run:
```bash
pytest tests/unit/integration/test_ptm_virtual_perturbation.py -v
```

Expected: FAIL。

- [ ] **Step 3: 实现 profile dataclass 与纯函数式扰动逻辑**

```python
@dataclass
class PTMPerturbationProfile:
    target_gene_index: int
    node_decay: float
    edge_scale: float


def apply_soft_perturbation(counts, net, profile):
    counts_new = counts.copy()
    net_new = net.copy()
    counts_new[:, profile.target_gene_index] *= profile.node_decay
    net_new[:, profile.target_gene_index] *= profile.edge_scale
    net_new[profile.target_gene_index, :] *= profile.edge_scale
    return counts_new, net_new
```

- [ ] **Step 4: 再补一个“退化为 hard KO”的兼容测试**

```python
def test_soft_profile_can_degenerate_to_hard_ko() -> None:
    profile = PTMPerturbationProfile(target_gene_index=0, node_decay=0.0, edge_scale=0.0)
    counts_new, net_new = apply_soft_perturbation(np.ones((2, 2)), np.ones((2, 2)), profile)
    assert counts_new[:, 0].sum() == 0.0
    assert net_new[0, :].sum() == 0.0
    assert net_new[:, 0].sum() == 0.0
```

- [ ] **Step 5: 跑单测**

Run:
```bash
pytest tests/unit/integration/test_ptm_virtual_perturbation.py -v
```

Expected: PASS。

### Task 6: 把 soft perturbation 接回 GenKI adapter，并形成对比执行脚本

**Files:**
- Modify: `src/integration/genki_adapter.py`
- Modify: `src/integration/ptm_virtual_perturbation.py`
- Create: `configs/integration/ptm_virtual_perturbation.yaml`
- Create: `scripts/run_ptm_virtual_perturbation.py`
- Test: `tests/integration/test_soft_perturbation_pipeline.py`

- [ ] **Step 1: 先写集成测试，要求同时跑 hard 和 soft**

```python
from src.integration.contracts import GenePerturbationRequest
from src.integration.genki_adapter import GenKIAdapter


def test_adapter_supports_hard_and_soft_modes(tmp_path) -> None:
    adapter = GenKIAdapter(ref_root="tests/fixtures/genki")
    hard = adapter.run(GenePerturbationRequest("TP53", "", "", -1, 1.0, "hard_ko"))
    soft = adapter.run(GenePerturbationRequest("TP53", "", "", -1, 0.6, "soft_ptm"))
    assert hard.mode == "hard_ko"
    assert soft.mode == "soft_ptm"
```

- [ ] **Step 2: 跑测试确认失败**

Run:
```bash
pytest tests/integration/test_soft_perturbation_pipeline.py -v
```

Expected: FAIL，提示 `soft_ptm` 未实现。

- [ ] **Step 3: 在 adapter 内新增 mode 分发，不直接改 GenKI 主训练器**

```python
def run(self, request: GenePerturbationRequest) -> PerturbationResult:
    if request.mode == "hard_ko":
        data_v = self._build_hard_ko_data(request.gene_symbol)
    elif request.mode == "soft_ptm":
        data_v = self._build_soft_ptm_data(request.gene_symbol, request.magnitude)
    else:
        raise ValueError(f"Unsupported mode: {request.mode}")
    return self._score_virtual_state(request, data_v)
```

- [ ] **Step 4: soft 模式优先通过“缩放网络矩阵后重新阈值化”复用 GenKI 现有逻辑**

```python
def _build_soft_ptm_data(self, gene_symbol: str, magnitude: float):
    gene_idx = self._lookup_gene_index(gene_symbol)
    profile = PTMPerturbationProfile(
        target_gene_index=gene_idx,
        node_decay=max(0.0, 1.0 - magnitude),
        edge_scale=max(0.0, 1.0 - magnitude),
    )
    counts_new, net_new = apply_soft_perturbation(self.counts, self.net, profile)
    return self._to_genki_data(counts_new, net_new)
```

- [ ] **Step 5: 增加对比脚本，至少输出三类结果**

```python
comparison = {
    "hard_distance": hard.distance_score,
    "soft_distance": soft.distance_score,
    "top_gene_overlap": overlap_at_k(hard.ranked_genes, soft.ranked_genes, k=10),
}
```

脚本输出目录要求：
- `hard_ko_results.json`
- `soft_ptm_results.json`
- `perturbation_comparison.md`

- [ ] **Step 6: 创建配置文件**

```yaml
integration:
  genki_ref_root: "ref/GenKI-master-src/GenKI-master"
  perturbation_mode: "soft_ptm"
  default_node_decay: 0.4
  default_edge_scale: 0.5
  compare_against_hard_ko: true
```

- [ ] **Step 7: 跑单测与集成测试**

Run:
```bash
pytest tests/unit/integration/test_ptm_virtual_perturbation.py tests/integration/test_soft_perturbation_pipeline.py -v
```

Expected: PASS。

- [ ] **Step 8: 运行 CLI smoke test**

Run:
```bash
python scripts/run_ptm_virtual_perturbation.py \
  --config configs/integration/ptm_virtual_perturbation.yaml \
  --gene TP53 \
  --magnitude 0.6 \
  --output outputs/results/ptm_virtual_perturbation_smoke
```

Expected: 输出 hard/soft 对比结果文件，并生成 `perturbation_comparison.md`。

## Chunk 4: 验证与收尾

### Task 7: 统一验证、结果审查与文档补全

**Files:**
- Modify: `README.md`
- Modify: `src/utils/io.py`（如果需要）

- [ ] **Step 1: 跑完整测试集合**

Run:
```bash
pytest \
  tests/unit/integration/test_contracts.py \
  tests/unit/integration/test_ptm_gene_mapper.py \
  tests/unit/evaluation/test_explainers.py \
  tests/unit/integration/test_ptm_virtual_perturbation.py \
  tests/integration/test_two_stage_pipeline.py \
  tests/integration/test_soft_perturbation_pipeline.py \
  -v
```

Expected: 全部 PASS。

- [ ] **Step 2: 做一次真实路径 smoke run**

Run:
```bash
python scripts/run_two_stage_explanation.py \
  --model outputs/models/best_model.pt \
  --config configs/integration/two_stage_explanation.yaml \
  --input data/processed/example_candidates.csv \
  --output outputs/results/final_two_stage

python scripts/run_ptm_virtual_perturbation.py \
  --config configs/integration/ptm_virtual_perturbation.yaml \
  --gene TP53 \
  --magnitude 0.6 \
  --output outputs/results/final_soft_perturbation
```

Expected:
- 第一个脚本输出 candidate ranking 与 GenKI explanation。
- 第二个脚本输出 hard vs soft 对比。

- [ ] **Step 3: 审查结果是否满足生物学和工程上的最低可信度**

检查清单：
- top candidate 是否来自真实存在的 PTM site。
- protein -> gene 映射是否出现大量丢失。
- soft 模式是否只是 hard KO 的数值复刻。
- top downstream genes overlap 是否过高或过低到不合理。

- [ ] **Step 4: 在 README 增加运行入口**

```markdown
### 两阶段解释
python scripts/run_two_stage_explanation.py --model ... --input ...

### PTM-aware virtual perturbation
python scripts/run_ptm_virtual_perturbation.py --gene TP53 --magnitude 0.6
```

- [ ] **Step 5: 写一份执行后报告**

输出建议：
- `outputs/results/final_two_stage/two_stage_summary.md`
- `outputs/results/final_soft_perturbation/perturbation_comparison.md`

## Risks And Mitigations

- `protein/PTM -> gene` 映射缺失率高。
  - 先在阶段 1 引入显式 `unmapped_candidates.csv`，不要静默丢弃。
- GenKI 参考实现路径或依赖不稳定。
  - 通过 `GenKIAdapter` 隔离动态 import，并先用 fixtures 覆盖契约测试。
- soft perturbation 与 hard KO 结果差异过小。
  - 先记录 `node_decay`、`edge_scale`、删边比例三种中间指标，再决定是否增加更细粒度 PTM-type-specific 参数。
- leave-one-PTM-out 计算成本偏高。
  - 优先只对 top-N confidence 样本和 top-M PTM site 运行；必要时做 batch 化掩码推理。

## Suggested Execution Order

1. Chunk 1 Task 1-2
2. Chunk 2 Task 3-4
3. Chunk 3 Task 5-6
4. Chunk 4 Task 7

## Plan Review Loop

- 针对每个 Chunk 单独做文档审查，优先检查：
  - 文件边界是否清晰。
  - 测试是否先于实现。
  - 阶段 1 与阶段 2 是否保持解耦。
- 若执行环境后续具备 git，再把每个 checkpoint 步骤替换成正常 `git add` / `git commit`。

Plan complete and saved to `docs/superpowers/plans/2026-03-15-two-stage-screening-ptm-aware-virtual-perturbation.md`. Ready to execute?

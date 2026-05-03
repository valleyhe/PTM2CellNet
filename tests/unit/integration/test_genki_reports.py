import pandas as pd

from src.integration.contracts import PerturbationResult
from src.integration.genki_reports import (
    build_comparison_summary_payload,
    build_generank_dataframe,
    build_gsea_ranked_dataframe,
    build_result_summary_payload,
    build_significant_gene_dataframe,
    build_two_stage_summary_payload,
    render_comparison_summary_markdown,
    render_two_stage_summary_markdown,
    save_gsea_ranked_tsv,
)


def _make_result(
    gene_symbol: str = "TP53",
    mode: str = "hard_ko",
    distance_score: float = 2.5,
    ranked_genes: list[str] | None = None,
    metadata: dict | None = None,
) -> PerturbationResult:
    if ranked_genes is None:
        ranked_genes = ["BAX", "MDM2", "EGFR"]
    if metadata is None:
        metadata = {
            "gene_scores": {"BAX": 1.2, "MDM2": 0.9, "EGFR": 0.4},
            "gene_indices": {"BAX": 2, "MDM2": 3, "EGFR": 0},
            "empirical_pvalues": {"BAX": 0.01, "MDM2": 0.04, "EGFR": 0.4},
            "adjusted_pvalues": {"BAX": 0.02, "MDM2": 0.05, "EGFR": 0.4},
            "bagging_hits": {"BAX": 10, "MDM2": 8, "EGFR": 1},
            "bagging_frequencies": {"BAX": 1.0, "MDM2": 0.8, "EGFR": 0.1},
            "significant_genes": ["BAX", "MDM2"],
            "scoring_method": "shift",
            "null_distribution_summary": {"n_permutations": 8},
        }
    return PerturbationResult(
        gene_symbol=gene_symbol,
        mode=mode,
        distance_score=distance_score,
        ranked_genes=ranked_genes,
        metadata=metadata,
    )


class TestReportGeneration:
    def test_generate_summary_report(self) -> None:
        result = _make_result()
        df = build_generank_dataframe(result)
        assert list(df.columns) == [
            "affected_gene",
            "dis",
            "rank",
            "gene_index",
            "hit",
            "frequency",
            "empirical_pvalue",
            "adjusted_pvalue",
            "is_significant",
        ]
        assert df.iloc[0]["affected_gene"] == "BAX"
        assert df.iloc[0]["dis"] == 1.2
        assert df.iloc[0]["rank"] == 1

    def test_generate_detailed_report(self) -> None:
        result = _make_result()
        df = build_significant_gene_dataframe(result)
        assert isinstance(df, pd.DataFrame)
        assert df["affected_gene"].tolist() == ["BAX", "MDM2"]
        assert df["is_significant"].all()

    def test_report_contains_gene_symbol(self) -> None:
        result = _make_result(gene_symbol="KRAS")
        payload = build_result_summary_payload(result, top_k=2)
        assert payload["gene_symbol"] == "KRAS"

    def test_report_contains_distance_score(self) -> None:
        result = _make_result(distance_score=3.14)
        payload = build_result_summary_payload(result, top_k=2)
        assert payload["distance_score"] == 3.14


class TestReportFormatting:
    def test_format_ranked_genes(self) -> None:
        result = _make_result(ranked_genes=["A", "B", "C"])
        df = build_generank_dataframe(result)
        assert df["affected_gene"].tolist() == ["A", "B", "C"]
        assert df["rank"].tolist() == [1, 2, 3]

    def test_format_significant_genes(self) -> None:
        result = _make_result(ranked_genes=["A", "B"], metadata={"significant_genes": ["A"]})
        df = build_significant_gene_dataframe(result)
        assert df["affected_gene"].tolist() == ["A"]

    def test_format_pvalues(self) -> None:
        result = _make_result()
        df = build_generank_dataframe(result)
        assert 0.0 <= df.iloc[0]["empirical_pvalue"] <= 1.0
        assert 0.0 <= df.iloc[0]["adjusted_pvalue"] <= 1.0

    def test_format_metadata(self) -> None:
        result = _make_result(metadata={"custom_key": "custom_value"})
        payload = build_result_summary_payload(result, top_k=2)
        assert payload["scoring_method"] is None


class TestReportSaving:
    def test_save_report_tsv(self, tmp_path) -> None:
        result = _make_result()
        path = tmp_path / "gsea.tsv"
        save_gsea_ranked_tsv(result, str(path))
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "affected_gene" in content

    def test_file_created(self, tmp_path) -> None:
        result = _make_result()
        path = tmp_path / "output.tsv"
        save_gsea_ranked_tsv(result, str(path))
        assert path.is_file()


class TestDataAggregation:
    def test_aggregate_multiple_results(self) -> None:
        results = [
            _make_result(gene_symbol="TP53", mode="hard_ko"),
            _make_result(gene_symbol="KRAS", mode="soft_ptm"),
        ]
        payload = build_two_stage_summary_payload(results, top_k=2)
        assert payload["total_results"] == 2
        assert set(payload["unique_genes"]) == {"KRAS", "TP53"}

    def test_aggregate_statistics(self) -> None:
        results = [
            _make_result(mode="hard_ko"),
            _make_result(mode="hard_ko"),
            _make_result(mode="soft_ptm"),
        ]
        payload = build_two_stage_summary_payload(results, top_k=2)
        assert payload["mode_counts"]["hard_ko"] == 2
        assert payload["mode_counts"]["soft_ptm"] == 1

    def test_top_genes_across_results(self) -> None:
        results = [
            _make_result(gene_symbol="G1", distance_score=3.0),
            _make_result(gene_symbol="G2", distance_score=1.0),
        ]
        payload = build_two_stage_summary_payload(results, top_k=1)
        assert payload["top_results"][0]["gene_symbol"] == "G1"


class TestReportEdgeCases:
    def test_empty_results(self) -> None:
        payload = build_two_stage_summary_payload([], top_k=2)
        assert payload["total_results"] == 0
        assert payload["unique_genes"] == []

    def test_single_result(self) -> None:
        results = [_make_result()]
        payload = build_two_stage_summary_payload(results, top_k=2)
        assert payload["total_results"] == 1

    def test_missing_optional_fields(self) -> None:
        result = PerturbationResult(
            gene_symbol="TP53",
            mode="hard_ko",
            distance_score=2.5,
            ranked_genes=["BAX"],
        )
        df = build_generank_dataframe(result)
        assert bool(df["is_significant"].iloc[0]) is False


class TestComparisonMarkdown:
    def test_render_comparison_summary_markdown(self) -> None:
        hard = _make_result(gene_symbol="TP53", mode="hard_ko", distance_score=2.5)
        soft = _make_result(gene_symbol="TP53", mode="soft_ptm", distance_score=1.8)
        payload = build_comparison_summary_payload(
            hard=hard,
            soft=soft,
            comparison={"top_gene_overlap_at_3": 2, "top_gene_overlap_at_10": 1},
            backend_info={"backend": "array_files"},
        )
        md = render_comparison_summary_markdown(payload)
        assert "PTM-Aware Virtual Perturbation Summary" in md
        assert "Gene: TP53" in md
        assert "Hard KO distance: 2.5000" in md
        assert "Soft PTM distance: 1.8000" in md
        assert "Top-3 overlap: 2" in md
        assert "Top-10 overlap: 1" in md
        assert "Hard KO" in md
        assert "Soft PTM" in md


class TestTwoStageMarkdown:
    def test_render_two_stage_summary_markdown(self) -> None:
        results = [
            _make_result(gene_symbol="TP53", mode="hard_ko", distance_score=2.5),
            _make_result(gene_symbol="KRAS", mode="soft_ptm", distance_score=1.2),
        ]
        md = render_two_stage_summary_markdown(results, top_k=1)
        assert "Two-Stage Explanation Summary" in md
        assert "Total results: 2" in md
        assert "TP53" in md
        assert "KRAS" in md

    def test_build_two_stage_payload_ordering(self) -> None:
        results = [
            _make_result(gene_symbol="G2", distance_score=1.0),
            _make_result(gene_symbol="G1", distance_score=3.0),
        ]
        payload = build_two_stage_summary_payload(results, top_k=2)
        assert payload["top_results"][0]["gene_symbol"] == "G1"

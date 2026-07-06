"""
信号网络映射模块
功能: 将PTM变化映射到信号通路和细胞状态
"""

import pandas as pd
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    from src.analysis.pathway_integration import PathwayDatabaseIntegration
    PATHWAY_INTEGRATION_AVAILABLE = True
except ImportError:
    PATHWAY_INTEGRATION_AVAILABLE = False
    logger.warning(
        "sspa library not installed. Using built-in pathway database only. "
        "Install sspa (pip install sspa) for KEGG/Reactome integration."
    )

# ---------------------------------------------------------------------------
# Built-in KEGG/Reactome pathway data (used as fallback when sspa is
# unavailable and as the default baseline for all SignalingNetworkMapper
# instances).
# ---------------------------------------------------------------------------
_BUILTIN_PATHWAY_DATA: Dict[str, Dict[str, Any]] = {
    "MAPK_signaling": {
        "description": "MAPK/ERK signaling pathway (built-in fallback)",
        "key_kinases": ["MAPK1", "MAPK3", "MAP2K1", "MAP2K2", "RAF1", "BRAF"],
        "key_substrates": ["ELK1", "MYC", "FOS", "JUN", "RSK", "MNK"],
        "ptm_types": ["phosphorylation"],
        "output_genes": ["ELK1", "MYC", "FOS", "JUN", "DUSP6", "SPRY2"],
    },
    "PI3K_AKT_signaling": {
        "description": "PI3K/AKT/mTOR signaling pathway (built-in fallback)",
        "key_kinases": ["PIK3CA", "AKT1", "AKT2", "MTOR", "PDK1"],
        "key_substrates": ["FOXO1", "FOXO3", "GSK3B", "TSC2", "BAD", "CASP9"],
        "ptm_types": ["phosphorylation"],
        "output_genes": ["FOXO1", "FOXO3", "GSK3B", "RPS6KB1", "EIF4EBP1"],
    },
    "JAK_STAT_signaling": {
        "description": "JAK/STAT signaling pathway (built-in fallback)",
        "key_kinases": ["JAK1", "JAK2", "JAK3", "TYK2"],
        "key_substrates": ["STAT1", "STAT2", "STAT3", "STAT4", "STAT5A", "STAT5B", "STAT6"],
        "ptm_types": ["phosphorylation"],
        "output_genes": ["STAT1", "STAT3", "SOCS1", "SOCS3", "IRF1", "MYC"],
    },
    "NF_kB_signaling": {
        "description": "NF-kB signaling pathway (built-in fallback)",
        "key_kinases": ["CHUK", "IKBKB", "IKBKG"],
        "key_substrates": ["NFKBIA", "NFKB1", "RELA"],
        "ptm_types": ["phosphorylation"],
        "output_genes": ["NFKBIA", "TNF", "IL6", "IL1B", "CCL2", "BCL2"],
    },
    "Wnt_signaling": {
        "description": "Wnt/beta-catenin signaling pathway (built-in fallback)",
        "key_kinases": ["GSK3B", "CK1", "NLK"],
        "key_substrates": ["CTNNB1", "AXIN1", "APC", "TCF7L2"],
        "ptm_types": ["phosphorylation"],
        "output_genes": ["CTNNB1", "MYC", "CCND1", "AXIN2", "LEF1", "TCF7"],
    },
    "TGF_beta_signaling": {
        "description": "TGF-beta signaling pathway (built-in fallback)",
        "key_kinases": ["TGFBR1", "TGFBR2", "ACVR1", "BMPR1A"],
        "key_substrates": ["SMAD2", "SMAD3", "SMAD1", "SMAD5", "SMAD9"],
        "ptm_types": ["phosphorylation"],
        "output_genes": ["SMAD7", "SERPINE1", "CDKN1A", "CDKN2B", "SNAI1", "SNAI2"],
    },
    "p53_signaling": {
        "description": "p53 signaling pathway (built-in fallback)",
        "key_kinases": ["ATM", "ATR", "CHEK1", "CHEK2"],
        "key_substrates": ["TP53", "MDM2", "CDKN1A"],
        "ptm_types": ["phosphorylation", "acetylation"],
        "output_genes": ["CDKN1A", "BAX", "BBC3", "GADD45A", "MDM2", "RRM2B"],
    },
    "Apoptosis_signaling": {
        "description": "Apoptosis signaling pathway (built-in fallback)",
        "key_kinases": ["CASP8", "CASP9", "CASP3"],
        "key_substrates": ["BID", "PARP1", "LMNA", "DFFA"],
        "ptm_types": ["phosphorylation", "cleavage"],
        "output_genes": ["BCL2", "BAX", "CASP3", "CASP9", "XIAP", "BIRC5"],
    },
    "Cell_cycle_signaling": {
        "description": "Cell cycle regulation pathway (built-in fallback)",
        "key_kinases": ["CDK1", "CDK2", "CDK4", "CDK6", "PLK1", "AURKA"],
        "key_substrates": ["RB1", "E2F1", "CDC25A", "WEE1"],
        "ptm_types": ["phosphorylation"],
        "output_genes": ["CCND1", "CCNE1", "CDKN1A", "CDKN1B", "E2F1", "MYC"],
    },
    "DNA_damage_response": {
        "description": "DNA damage response pathway (built-in fallback)",
        "key_kinases": ["ATM", "ATR", "DNAPK", "CHEK1", "CHEK2"],
        "key_substrates": ["H2AX", "TP53", "BRCA1", "RAD51"],
        "ptm_types": ["phosphorylation", "ubiquitination"],
        "output_genes": ["TP53", "CDKN1A", "GADD45A", "BRCA1", "RAD51", "PCNA"],
    },
}

# 下游基因表达变化受通路活性的影响因子（假设中等影响）。
DOWNSTREAM_GENE_IMPACT_FACTOR = 0.5


class SignalingNetworkMapper:
    """
    信号网络映射器

    将PTM变化映射到信号通路，预测下游效应
    """

    # Optional[PathwayDatabaseIntegration] when sspa is installed, else None.
    # Declared as Any because the type is only importable when the optional
    # sspa dependency is present.
    pathway_integration: Any

    # 主要信号通路定义
    SIGNALING_PATHWAYS = {
        'MAPK/ERK': {
            'description': 'RAS-RAF-MEK-ERK通路',
            'key_kinases': ['BRAF', 'RAF1', 'MAP2K1', 'MAP2K2', 'MAPK1', 'MAPK3'],
            'key_substrates': ['EGFR', 'KRAS', 'NRAS', 'HRAS'],
            'ptm_types': ['Phosphorylation'],
            'output_genes': ['FOS', 'JUN', 'MYC', 'EGR1'],
        },
        'PI3K/AKT': {
            'description': 'PI3K-AKT-mTOR通路',
            'key_kinases': ['PIK3CA', 'PIK3CB', 'AKT1', 'AKT2', 'MTOR'],
            'key_substrates': ['PTEN', 'PDK1', 'TSC2', 'GSK3B'],
            'ptm_types': ['Phosphorylation'],
            'output_genes': ['FOXO1', 'FOXO3', 'S6K', '4EBP1'],
        },
        'JAK/STAT': {
            'description': 'JAK-STAT信号通路',
            'key_kinases': ['JAK1', 'JAK2', 'JAK3', 'TYK2'],
            'key_substrates': ['STAT1', 'STAT2', 'STAT3', 'STAT5A', 'STAT5B'],
            'ptm_types': ['Phosphorylation'],
            'output_genes': ['SOCS1', 'SOCS3', 'IRF1', 'IRF9'],
        },
        'NF-kB': {
            'description': 'NF-kB炎症信号通路',
            'key_kinases': ['IKBKB', 'IKBKA', 'CHUK'],
            'key_substrates': ['NFKBIA', 'NFKBIB', 'RELA', 'NFKB1'],
            'ptm_types': ['Phosphorylation', 'Ubiquitination'],
            'output_genes': ['TNF', 'IL6', 'IL8', 'CXCL10'],
        },
        'Wnt/beta-catenin': {
            'description': 'Wnt信号通路',
            'key_kinases': ['GSK3B', 'CSNK1A1', 'CSNK2A1'],
            'key_substrates': ['CTNNB1', 'APC', 'AXIN1'],
            'ptm_types': ['Phosphorylation', 'Ubiquitination'],
            'output_genes': ['MYC', 'CCND1', 'AXIN2', 'LEF1'],
        },
        'Cell Cycle': {
            'description': '细胞周期调控',
            'key_kinases': ['CDK1', 'CDK2', 'CDK4', 'CDK6'],
            'key_substrates': ['RB1', 'TP53', 'CCND1', 'CCNE1'],
            'ptm_types': ['Phosphorylation', 'Ubiquitination'],
            'output_genes': ['E2F1', 'E2F2', 'CDKN1A', 'CDKN1B'],
        },
        'Apoptosis': {
            'description': '细胞凋亡通路',
            'key_kinases': ['CASP3', 'CASP8', 'CASP9'],
            'key_substrates': ['BCL2', 'BAX', 'PARP1', 'XIAP'],
            'ptm_types': ['Phosphorylation', 'Ubiquitination'],
            'output_genes': ['BCL2L11', 'PMAIP1', 'BBC3'],
        },
        'DNA Damage': {
            'description': 'DNA损伤修复',
            'key_kinases': ['ATM', 'ATR', 'CHEK1', 'CHEK2', 'TP53'],
            'key_substrates': ['H2AX', 'BRCA1', 'BRCA2', 'RAD51'],
            'ptm_types': ['Phosphorylation', 'Ubiquitination', 'Sumoylation'],
            'output_genes': ['CDKN1A', 'GADD45A', 'BAX'],
        },
        'TGF-beta': {
            'description': 'TGF-beta signaling pathway',
            'key_kinases': ['TGFBR1', 'TGFBR2', 'SMAD2', 'SMAD3', 'SMAD4'],
            'key_substrates': ['SERPINE1', 'CTGF', 'TGFB1'],
            'ptm_types': ['Phosphorylation', 'Ubiquitination'],
            'output_genes': ['PAI1', 'JUN', 'CDKN1A'],
        },
        'Hippo': {
            'description': 'Hippo signaling pathway',
            'key_kinases': ['STK3', 'STK4', 'LATS1', 'LATS2'],
            'key_substrates': ['YAP1', 'WWTR1', 'TEAD1'],
            'ptm_types': ['Phosphorylation', 'Ubiquitination'],
            'output_genes': ['CTGF', 'CYR61', 'ANKRD1'],
        },
        'Notch': {
            'description': 'Notch signaling pathway',
            'key_kinases': ['NOTCH1', 'NOTCH2', 'ADAM17', 'PSEN1'],
            'key_substrates': ['RBPJ', 'MAML1', 'HES1'],
            'ptm_types': ['Phosphorylation', 'Ubiquitination'],
            'output_genes': ['HES1', 'HEY1', 'NRARP'],
        },
        'mTOR': {
            'description': 'mTOR signaling pathway',
            'key_kinases': ['MTOR', 'RICTOR', 'RPTOR', 'AKT1S1'],
            'key_substrates': ['RPS6KB1', 'EIF4EBP1', 'ULK1'],
            'ptm_types': ['Phosphorylation', 'Ubiquitination'],
            'output_genes': ['S6K', '4EBP1', 'HIF1A'],
        },
    }

    # PTM类型功能影响
    PTM_FUNCTIONAL_IMPACT = {
        'Phosphorylation': {
            'gain': '激活激酶活性或蛋白互作',
            'loss': '失去活性或互作能力',
        },
        'Ubiquitination': {
            'gain': '促进蛋白降解',
            'loss': '蛋白稳定性增加',
        },
        'Acetylation': {
            'gain': '调节蛋白功能或稳定性',
            'loss': '失去调控',
        },
        'Methylation': {
            'gain': '调节蛋白互作或定位',
            'loss': '失去调控',
        },
        'Sumoylation': {
            'gain': '调节核定位或转录活性',
            'loss': '失去核定位',
        },
        'Succinylation': {
            'gain': '代谢调控',
            'loss': '失去代谢调控',
        },
    }

    def __init__(
        self,
        pathway_db_path: Optional[str] = None,
        organism: str = "hsa",
        organism_name: str = "Homo sapiens",
    ):
        """
        初始化信号网络映射器

        参数:
            pathway_db_path: 通路数据库标识或缓存目录路径（可选）。
                - "kegg": 通过 sspa 加载 KEGG 通路
                - "reactome": 通过 sspa 加载 Reactome 通路
                - "both": 同时加载 KEGG 与 Reactome
                - 其它字符串: 视为缓存目录路径，尝试从缓存加载
            organism: KEGG 物种代码（默认 "hsa" 人类）
            organism_name: Reactome 物种名（默认 "Homo sapiens"）
        """
        self.pathway_db_path = pathway_db_path
        self.organism = organism
        self.organism_name = organism_name
        self.pathways = dict(self.SIGNALING_PATHWAYS)

        # Always load built-in pathway data as baseline
        for pw_name, pw_data in _BUILTIN_PATHWAY_DATA.items():
            self.pathways[pw_name] = dict(pw_data)  # shallow copy

        # External pathway database integration (FEAT-02)
        if PATHWAY_INTEGRATION_AVAILABLE:
            self.pathway_integration = PathwayDatabaseIntegration()
        else:
            self.pathway_integration = None

        if pathway_db_path:
            self._load_pathway_db(pathway_db_path)

        # 构建蛋白到通路的映射
        self.protein_to_pathway = self._build_protein_mapping()
        self._validation_report: Optional[Dict] = None

        logger.info(f"加载 {len(self.pathways)} 个信号通路")

    def _load_pathway_db(self, db_path: str) -> None:
        """加载外部通路数据库（KEGG/Reactome，FEAT-02 完整集成）。

        ``db_path`` 可以是以下之一:
            - "kegg"     : 通过 sspa 加载 KEGG 通路
            - "reactome" : 通过 sspa 加载 Reactome 通路
            - "both"     : 同时加载 KEGG 与 Reactome
            - 其它字符串  : 视为缓存目录路径

        当 sspa 不可用或网络/缓存失败时，回退到内置通路并发出明确警告，
        不会抛出 NotImplementedError（CONF-03）。
        """
        if self.pathway_integration is None:
            logger.warning(
                "PathwayDatabaseIntegration 不可用（pathway_integration 模块"
                "导入失败）。回退到内置信号通路数据。如需 KEGG/Reactome 集成，"
                "请安装 sspa 及其依赖（rpy2）。"
            )
            return

        directive = str(db_path).lower()
        added = 0

        loaders = []
        if directive in ("kegg", "both"):
            loaders.append(("KEGG", self.pathway_integration.load_kegg_pathways))
        if directive in ("reactome", "both"):
            loaders.append(("Reactome", self.pathway_integration.load_reactome_pathways))
        if not loaders:
            # Treat as cache directory path
            try:
                self.pathway_integration.cache_dir = Path(db_path)
                loaders.append(("KEGG", self.pathway_integration.load_kegg_pathways))
                loaders.append(("Reactome", self.pathway_integration.load_reactome_pathways))
            except (FileNotFoundError, OSError, ValueError) as e:
                logger.warning(
                    "无法识别的通路数据库路径 '%s' (%s)。回退到内置通路。",
                    db_path, e,
                )
                return

        for name, loader_fn in loaders:
            try:
                pw = loader_fn()
            except (ImportError, RuntimeError, ValueError, OSError) as e:
                logger.warning(
                    "加载 %s 通路失败: %s。%s 通路将仅使用内置数据。",
                    name, e, name,
                )
                continue
            # Merge external pathways into the in-memory pathway map.
            # External entries map pathway_id -> gene list; we adapt them to
            # the internal schema so downstream mapping still works.
            for pathway_id, genes in (pw or {}).items():
                gene_list = list(genes) if not isinstance(genes, dict) else list(genes.keys())
                if pathway_id in self.pathways:
                    continue  # don't shadow built-in pathways
                self.pathways[f"{name}:{pathway_id}"] = {
                    'description': f"{name} pathway {pathway_id}",
                    'key_kinases': [],
                    'key_substrates': gene_list,
                    'ptm_types': [],
                    'output_genes': gene_list,
                }
                added += 1

        logger.info(
            "从外部数据库加载完成：新增 %d 条通路（当前总数 %d）。",
            added, len(self.pathways),
        )

    def get_missing_pathway_warning(self, pathway_names: List[str]) -> List[str]:
        """Check which pathways are missing from the mapper.

        Args:
            pathway_names: List of pathway names to check.

        Returns:
            List of pathway names that are not available.
        """
        missing = [name for name in pathway_names if name not in self.pathways]
        if missing:
            logger.warning(
                "Missing pathways: %s. Available: %s. "
                "Install sspa for KEGG/Reactome integration.",
                missing, list(self.pathways.keys()),
            )
        return missing

    def _build_protein_mapping(self) -> Dict[str, Set[str]]:
        """构建蛋白到通路的映射"""
        mapping = defaultdict(set)

        for pathway, info in self.pathways.items():
            for protein in info['key_kinases']:
                mapping[protein].add(pathway)
            for protein in info['key_substrates']:
                mapping[protein].add(pathway)

        return dict(mapping)

    def map_ptm_to_pathway(
        self,
        gene_symbol: str,
        ptm_type: str,
        effect: str,
    ) -> Dict[str, float]:
        """
        将PTM变化映射到信号通路

        参数:
            gene_symbol: 基因符号
            ptm_type: PTM类型
            effect: 效应类型 ('gain'/'loss'/'neutral')

        返回:
            通路影响分数
        """
        pathway_impacts: Dict[str, Any] = {}

        if gene_symbol not in self.protein_to_pathway:
            return pathway_impacts

        affected_pathways = self.protein_to_pathway[gene_symbol]

        for pathway in affected_pathways:
            pathway_info = self.pathways[pathway]

            # 检查PTM类型是否相关
            if ptm_type not in pathway_info['ptm_types']:
                continue

            # 计算影响分数
            if gene_symbol in pathway_info['key_kinases']:
                # 激酶变异影响更大
                impact_score = 0.8 if effect != 'neutral' else 0.1
            elif gene_symbol in pathway_info['key_substrates']:
                impact_score = 0.5 if effect != 'neutral' else 0.1
            else:
                impact_score = 0.2 if effect != 'neutral' else 0.0

            pathway_impacts[pathway] = impact_score

        return pathway_impacts

    def predict_pathway_activity(
        self,
        ptm_effects: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        预测信号通路活性变化

        参数:
            ptm_effects: PTM效应DataFrame

        返回:
            通路活性变化分数
        """
        pathway_activities: Dict[str, float] = defaultdict(float)
        pathway_counts: Dict[str, int] = defaultdict(int)

        for _, row in ptm_effects.iterrows():
            gene = row.get('gene_symbol', '')
            ptm_type = row.get('ptm_type', '')
            effect = row.get('effect', 'neutral')
            delta_prob = row.get('delta_prob', 0)

            if gene and effect != 'neutral':
                impacts = self.map_ptm_to_pathway(gene, ptm_type, effect)

                for pathway, score in impacts.items():
                    # 加权累积
                    weight = abs(delta_prob) * score
                    if effect == 'gain':
                        pathway_activities[pathway] += weight
                    else:
                        pathway_activities[pathway] -= weight
                    pathway_counts[pathway] += 1

        # 归一化
        for pathway in pathway_activities:
            if pathway_counts[pathway] > 0:
                pathway_activities[pathway] /= pathway_counts[pathway]

        return dict(pathway_activities)

    def predict_downstream_genes(
        self,
        pathway_activities: Dict[str, float],
        threshold: float = 0.3,
    ) -> Dict[str, float]:
        """
        预测下游基因表达变化

        参数:
            pathway_activities: 通路活性变化
            threshold: 影响阈值

        返回:
            基因表达变化预测
        """
        gene_changes: Dict[str, float] = defaultdict(float)

        for pathway, activity in pathway_activities.items():
            if abs(activity) < threshold:
                continue

            pathway_info = self.pathways.get(pathway)
            if not pathway_info:
                continue

            # 下游基因受通路活性影响
            for gene in pathway_info['output_genes']:
                gene_changes[gene] += activity * DOWNSTREAM_GENE_IMPACT_FACTOR  # 假设中等影响

        return dict(gene_changes)

    def _validate_builtin_pathways(self) -> Dict[str, Dict]:
        """Validate built-in pathways for data integrity.

        Performs a self-consistency check on all loaded pathways:
        checks that required fields (key_kinases, key_substrates,
        output_genes, ptm_types) are non-empty. This always works
        without external dependencies.

        Returns:
            Validation report with source: "builtin" annotation.
        """
        validation_report = {}

        for pathway_name, pathway_info in self.pathways.items():
            issues = []
            has_kinases = bool(pathway_info.get('key_kinases'))
            has_substrates = bool(pathway_info.get('key_substrates'))
            has_output = bool(pathway_info.get('output_genes'))
            has_ptm_types = bool(pathway_info.get('ptm_types'))

            if not has_kinases:
                issues.append("no key_kinases")
            if not has_substrates:
                issues.append("no key_substrates")
            if not has_output:
                issues.append("no output_genes")
            if not has_ptm_types:
                issues.append("no ptm_types")

            gene_count = (
                len(pathway_info.get('key_kinases', [])) +
                len(pathway_info.get('key_substrates', [])) +
                len(pathway_info.get('output_genes', []))
            )

            validation_report[pathway_name] = {
                'validated': len(issues) == 0,
                'source': 'builtin',
                'gene_count': gene_count,
                'kinase_count': len(pathway_info.get('key_kinases', [])),
                'substrate_count': len(pathway_info.get('key_substrates', [])),
                'issues': issues,
            }

        return validation_report

    def validate_against_databases(
        self,
        min_coverage: float = 0.7,
    ) -> Dict[str, Dict]:
        """
        Validate built-in pathways against KEGG/Reactome databases.

        When sspa/rpy2 is unavailable, falls back to built-in
        self-consistency validation via ``_validate_builtin_pathways``.

        Args:
            min_coverage: Minimum gene coverage for validation (0-1)

        Returns:
            Validation report for each pathway
        """
        if self.pathway_integration is None:
            logger.warning(
                "Pathway integration not available; "
                "using built-in self-consistency validation"
            )
            self._validation_report = self._validate_builtin_pathways()
            return self._validation_report

        try:
            self._validation_report = self.pathway_integration.validate_builtin_pathways(
                min_coverage=min_coverage
            )

            # Log results
            validated_count = sum(
                1 for r in self._validation_report.values() if r['validated']
            )
            logger.info(
                f"Pathway validation: {validated_count}/{len(self.pathways)} "
                f"pathways meet {min_coverage:.0%} coverage threshold"
            )

            for pathway_name, report in self._validation_report.items():
                status = "✓" if report['validated'] else "✗"
                logger.info(
                    f"  {status} {pathway_name}: {report['coverage']:.1%} coverage "
                    f"(match: {report['reactome_match_id']})"
                )

            return self._validation_report

        except (RuntimeError, ValueError, KeyError, OSError) as e:
            logger.error(f"Pathway validation failed: {e}")
            # Return empty report on failure
            return {}

    def is_pathway_validated(self, pathway_name: str) -> bool:
        """Check if a pathway passed external validation."""
        if self._validation_report is None:
            return False
        return bool(self._validation_report.get(pathway_name, {}).get('validated', False))

    def get_validation_summary(self) -> Dict:
        """Get summary of pathway validation results."""
        if self._validation_report is None:
            if self.pathways:
                return {
                    'status': 'builtin_only',
                    'total_pathways': len(self.pathways),
                    'note': 'External KEGG/Reactome validation not available. '
                            'Install sspa (pip install sspa) for full validation.',
                }
            return {'status': 'not_validated'}

        total = len(self._validation_report)
        validated = sum(1 for r in self._validation_report.values() if r['validated'])

        return {
            'status': 'validated',
            'total_pathways': total,
            'validated_pathways': validated,
            'validation_rate': validated / total if total > 0 else 0,
            'pathways': self._validation_report,
        }

    def generate_network_report(
        self,
        ptm_effects: pd.DataFrame,
        include_validation: bool = True,
    ) -> Dict:
        """
        生成网络效应报告

        Args:
            ptm_effects: PTM效应DataFrame
            include_validation: Whether to include pathway validation status

        返回:
            完整的网络效应报告
        """
        # 预测通路活性
        pathway_activities = self.predict_pathway_activity(ptm_effects)

        # 预测下游基因
        downstream_genes = self.predict_downstream_genes(pathway_activities)

        # 识别关键通路
        key_pathways = [
            (p, a) for p, a in sorted(
                pathway_activities.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            ) if abs(a) > 0.2
        ][:5]

        # 生成报告
        report = {
            'summary': {
                'total_ptm_changes': len(ptm_effects),
                'significant_changes': len(ptm_effects[ptm_effects['effect'] != 'neutral']),
                'affected_pathways': len([a for a in pathway_activities.values() if abs(a) > 0.1]),
            },
            'pathway_activities': pathway_activities,
            'key_pathways': key_pathways,
            'downstream_genes': downstream_genes,
            'interpretation': self._generate_interpretation(pathway_activities, downstream_genes),
        }

        # Add validation info if available
        if include_validation and self._validation_report:
            report['pathway_validation'] = self.get_validation_summary()

        return report

    def _generate_interpretation(
        self,
        pathway_activities: Dict[str, float],
        downstream_genes: Dict[str, float],
    ) -> str:
        """生成生物学解释"""
        interpretations = []

        # 分析主要通路变化
        for pathway, activity in sorted(
            pathway_activities.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]:
            if abs(activity) > 0.2:
                direction = "激活" if activity > 0 else "抑制"
                pathway_name = self.pathways[pathway]['description']
                interpretations.append(f"- {pathway_name}通路可能被{direction} (分数: {activity:.2f})")

        if not interpretations:
            interpretations.append("- 未检测到显著的信号通路变化")

        return "\n".join(interpretations)


class PTMNetworkAnalyzer:
    """PTM网络分析器"""

    def __init__(self):
        self.mapper = SignalingNetworkMapper()

    def analyze_variant(
        self,
        gene_symbol: str,
        uniprot_id: str,
        position: int,
        ref_aa: str,
        alt_aa: str,
        ptm_effects: Dict[str, Dict],
    ) -> Dict:
        """
        分析变异的信号网络效应

        参数:
            gene_symbol: 基因符号
            uniprot_id: UniProt ID
            position: 变异位置
            ref_aa: 参考氨基酸
            alt_aa: 变异氨基酸
            ptm_effects: 各PTM类型的效应预测结果

        返回:
            完整分析结果
        """
        # 转换为DataFrame
        effects_list = []
        for ptm_type, effect in ptm_effects.items():
            effects_list.append({
                'gene_symbol': gene_symbol,
                'ptm_type': ptm_type,
                'effect': effect.get('effect', 'neutral'),
                'delta_prob': effect.get('delta_prob', 0),
            })

        ptm_df = pd.DataFrame(effects_list)

        # 生成网络报告
        network_report = self.mapper.generate_network_report(ptm_df)

        return {
            'variant': {
                'gene_symbol': gene_symbol,
                'uniprot_id': uniprot_id,
                'position': position,
                'ref_aa': ref_aa,
                'alt_aa': alt_aa,
            },
            'ptm_effects': ptm_effects,
            'network_effects': network_report,
        }


def create_pathway_visualization(
    pathway_activities: Dict[str, float],
    output_path: str,
):
    """创建通路活性可视化"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use('Agg')  # 非交互式后端

        pathways = list(pathway_activities.keys())
        activities = list(pathway_activities.values())

        # 颜色映射
        colors = ['green' if a > 0 else 'red' for a in activities]

        fig, ax = plt.subplots(figsize=(10, 6))
        _ = ax.barh(pathways, activities, color=colors, alpha=0.7)

        ax.set_xlabel('Activity Change')
        ax.set_title('Signaling Pathway Activity Changes')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"可视化保存至: {output_path}")

    except ImportError:
        logger.warning("matplotlib未安装，跳过可视化")

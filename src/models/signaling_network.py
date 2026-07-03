"""
信号网络映射模块
功能: 将PTM变化映射到信号通路和细胞状态
"""

# mypy: disable-error-code="arg-type,assignment,dict-item,operator,return-value,name-defined"
import pandas as pd
from typing import Any, Dict, Optional, Set
from collections import defaultdict
from pathlib import Path
import logging

try:
    from src.analysis.pathway_integration import PathwayDatabaseIntegration
    PATHWAY_INTEGRATION_AVAILABLE = True
except ImportError:
    PATHWAY_INTEGRATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class SignalingNetworkMapper:
    """
    信号网络映射器

    将PTM变化映射到信号通路，预测下游效应
    """

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
        use_cache: bool = True,
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
            use_cache: 是否优先使用缓存数据
        """
        self.pathway_db_path = pathway_db_path
        self.organism = organism
        self.organism_name = organism_name
        self.use_cache = use_cache
        self.pathways = dict(self.SIGNALING_PATHWAYS)

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
            except Exception as e:
                logger.warning(
                    "无法识别的通路数据库路径 '%s' (%s)。回退到内置通路。",
                    db_path, e,
                )
                return

        for name, loader_fn in loaders:
            try:
                pw = loader_fn()
            except Exception as e:
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
                gene_changes[gene] += activity * 0.5  # 假设中等影响

        return dict(gene_changes)

    def validate_against_databases(
        self,
        min_coverage: float = 0.7,
        use_cache: bool = True,
    ) -> Dict[str, Dict]:
        """
        Validate built-in pathways against KEGG/Reactome databases.

        Args:
            min_coverage: Minimum gene coverage for validation (0-1)
            use_cache: Whether to use cached pathway data

        Returns:
            Validation report for each pathway
        """
        if self.pathway_integration is None:
            logger.warning("Pathway integration not available, skipping validation")
            return {}

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

        except Exception as e:
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

"""
扩展信号通路知识库 - 第二部分
功能: 添加更多通路、药物-靶点关系、疾病关联
"""

from typing import Any, Dict, List, Set, Tuple

# ============================================================
# 药物-靶点关系数据库
# ============================================================

DRUG_TARGET_DATABASE = {
    # 激酶抑制剂
    'kinase_inhibitors': {
        'imatinib': {
            'targets': ['ABL1', 'KIT', 'PDGFR'],
            'indication': ['CML', 'GIST'],
            'pathway_effect': {'MAPK/ERK': 'inhibition', 'PI3K/AKT': 'inhibition'},
            'resistance_mechanisms': ['BCR-ABL T315I', 'KIT D816V'],
        },
        'erlotinib': {
            'targets': ['EGFR'],
            'indication': ['NSCLC', 'Pancreatic cancer'],
            'pathway_effect': {'MAPK/ERK': 'inhibition', 'PI3K/AKT': 'inhibition'},
            'resistance_mechanisms': ['EGFR T790M', 'MET amplification'],
        },
        'gefitinib': {
            'targets': ['EGFR'],
            'indication': ['NSCLC'],
            'pathway_effect': {'MAPK/ERK': 'inhibition', 'PI3K/AKT': 'inhibition'},
            'resistance_mechanisms': ['EGFR T790M', 'KRAS mutation'],
        },
        'vemurafenib': {
            'targets': ['BRAF'],
            'indication': ['Melanoma'],
            'pathway_effect': {'MAPK/ERK': 'inhibition'},
            'resistance_mechanisms': ['NRAS mutation', 'MEK mutation', 'BRAF splice variants'],
        },
        'dabrafenib': {
            'targets': ['BRAF'],
            'indication': ['Melanoma', 'NSCLC'],
            'pathway_effect': {'MAPK/ERK': 'inhibition'},
            'resistance_mechanisms': ['MEK1/2 mutation', 'NRAS mutation'],
        },
        'trametinib': {
            'targets': ['MEK1', 'MEK2'],
            'indication': ['Melanoma', 'NSCLC'],
            'pathway_effect': {'MAPK/ERK': 'inhibition'},
            'resistance_mechanisms': ['BRAF amplification', 'NRAS mutation'],
        },
        'selumetinib': {
            'targets': ['MEK1', 'MEK2'],
            'indication': ['NF1', 'Melanoma'],
            'pathway_effect': {'MAPK/ERK': 'inhibition'},
            'resistance_mechanisms': ['KRAS mutation', 'PI3K activation'],
        },
        'everolimus': {
            'targets': ['mTOR'],
            'indication': ['Renal cell carcinoma', 'Breast cancer'],
            'pathway_effect': {'mTOR': 'inhibition', 'Autophagy': 'activation'},
            'resistance_mechanisms': ['AKT activation', '4EBP1 mutation'],
        },
        'temsirolimus': {
            'targets': ['mTOR'],
            'indication': ['Renal cell carcinoma'],
            'pathway_effect': {'mTOR': 'inhibition'},
            'resistance_mechanisms': ['PI3K mutation', 'AKT activation'],
        },
        'palbociclib': {
            'targets': ['CDK4', 'CDK6'],
            'indication': ['HR+ Breast cancer'],
            'pathway_effect': {'Cell Cycle': 'inhibition'},
            'resistance_mechanisms': ['RB1 loss', 'Cyclin E amplification', 'CDK2 activation'],
        },
        'ribociclib': {
            'targets': ['CDK4', 'CDK6'],
            'indication': ['HR+ Breast cancer'],
            'pathway_effect': {'Cell Cycle': 'inhibition'},
            'resistance_mechanisms': ['RB1 mutation', 'CCNE1 amplification'],
        },
        'abemaciclib': {
            'targets': ['CDK4', 'CDK6', 'CDK9'],
            'indication': ['HR+ Breast cancer'],
            'pathway_effect': {'Cell Cycle': 'inhibition'},
            'resistance_mechanisms': ['RB1 loss', 'CDK2 activation'],
        },
    },

    # PI3K/AKT通路抑制剂
    'pi3k_akt_inhibitors': {
        'alpelisib': {
            'targets': ['PIK3CA'],
            'indication': ['HR+/HER2- Breast cancer'],
            'pathway_effect': {'PI3K/AKT': 'inhibition'},
            'resistance_mechanisms': ['PTEN loss', 'AKT mutation'],
        },
        'buparlisib': {
            'targets': ['PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3CG'],
            'indication': ['Breast cancer'],
            'pathway_effect': {'PI3K/AKT': 'inhibition'},
            'resistance_mechanisms': ['RTK activation', 'mTOR activation'],
        },
        'ipatasertib': {
            'targets': ['AKT'],
            'indication': ['Breast cancer', 'Prostate cancer'],
            'pathway_effect': {'PI3K/AKT': 'inhibition'},
            'resistance_mechanisms': ['PI3K mutation', 'PTEN loss'],
        },
        'capivasertib': {
            'targets': ['AKT'],
            'indication': ['Breast cancer'],
            'pathway_effect': {'PI3K/AKT': 'inhibition'},
            'resistance_mechanisms': ['mTOR activation', 'MAPK activation'],
        },
    },

    # PARP抑制剂
    'parp_inhibitors': {
        'olaparib': {
            'targets': ['PARP1', 'PARP2'],
            'indication': ['Ovarian cancer', 'Breast cancer', 'Pancreatic cancer', 'Prostate cancer'],
            'pathway_effect': {'DNA Damage': 'synthetic_lethality'},
            'resistance_mechanisms': ['BRCA reversion', '53BP1 loss', 'REV7 loss'],
        },
        'rucaparib': {
            'targets': ['PARP1', 'PARP2', 'PARP3'],
            'indication': ['Ovarian cancer', 'Prostate cancer'],
            'pathway_effect': {'DNA Damage': 'synthetic_lethality'},
            'resistance_mechanisms': ['BRCA1/2 reversion', 'HR restoration'],
        },
        'niraparib': {
            'targets': ['PARP1', 'PARP2'],
            'indication': ['Ovarian cancer'],
            'pathway_effect': {'DNA Damage': 'synthetic_lethality'},
            'resistance_mechanisms': ['BRCA reversion', 'Shieldin complex loss'],
        },
        'talazoparib': {
            'targets': ['PARP1', 'PARP2'],
            'indication': ['Breast cancer'],
            'pathway_effect': {'DNA Damage': 'synthetic_lethality'},
            'resistance_mechanisms': ['BRCA reversion', 'PARG loss'],
        },
    },

    # 免疫检查点抑制剂
    'immune_checkpoint_inhibitors': {
        'pembrolizumab': {
            'targets': ['PD-1'],
            'indication': ['Melanoma', 'NSCLC', 'Multiple cancers'],
            'pathway_effect': {'Immune': 'activation'},
            'resistance_mechanisms': ['JAK1/2 mutation', 'B2M loss', 'PTEN loss'],
        },
        'nivolumab': {
            'targets': ['PD-1'],
            'indication': ['Melanoma', 'NSCLC', 'RCC', 'Multiple cancers'],
            'pathway_effect': {'Immune': 'activation'},
            'resistance_mechanisms': ['IFN-γ pathway loss', 'WNT/β-catenin activation'],
        },
        'atezolizumab': {
            'targets': ['PD-L1'],
            'indication': ['NSCLC', 'Bladder cancer'],
            'pathway_effect': {'Immune': 'activation'},
            'resistance_mechanisms': ['PD-L1 amplification', 'STK11 mutation'],
        },
        'ipilimumab': {
            'targets': ['CTLA-4'],
            'indication': ['Melanoma', 'RCC'],
            'pathway_effect': {'Immune': 'activation'},
            'resistance_mechanisms': ['IFN-γ pathway mutation', 'T-cell exclusion'],
        },
    },

    # BCL2抑制剂
    'bcl2_inhibitors': {
        'venetoclax': {
            'targets': ['BCL2'],
            'indication': ['CLL', 'AML'],
            'pathway_effect': {'Apoptosis': 'activation'},
            'resistance_mechanisms': ['BCL2 mutation', 'MCL1 upregulation', 'BCL-XL upregulation'],
        },
    },
}

# ============================================================
# PTM与疾病关联数据库
# ============================================================

PTM_DISEASE_ASSOCIATIONS = {
    # 磷酸化相关疾病
    'Phosphorylation': {
        'gain_of_function': {
            'BRAF V600E': {
                'disease': 'Melanoma, Thyroid cancer',
                'pathway': 'MAPK/ERK',
                'effect': 'Constitutive activation of BRAF kinase',
                'frequency': '50% in melanoma',
            },
            'EGFR L858R': {
                'disease': 'NSCLC',
                'pathway': 'EGFR/MAPK/PI3K',
                'effect': 'Increased EGFR kinase activity',
                'frequency': '15% in NSCLC',
            },
            'AKT1 E17K': {
                'disease': 'Breast cancer, Ovarian cancer',
                'pathway': 'PI3K/AKT',
                'effect': 'Membrane localization and activation',
                'frequency': '5% in breast cancer',
            },
            'JAK2 V617F': {
                'disease': 'Polycythemia vera, Myelofibrosis',
                'pathway': 'JAK/STAT',
                'effect': 'Constitutive JAK2 activation',
                'frequency': '95% in PV',
            },
        },
        'loss_of_function': {
            'PTEN loss': {
                'disease': 'Multiple cancers',
                'pathway': 'PI3K/AKT',
                'effect': 'Loss of PI3K negative regulation',
                'frequency': '30-40% in endometrial cancer',
            },
            'LKB1/STK11 loss': {
                'disease': 'Lung cancer, Peutz-Jeghers',
                'pathway': 'AMPK/mTOR',
                'effect': 'Loss of AMPK activation',
                'frequency': '20% in NSCLC',
            },
        },
    },

    # 泛素化相关疾病
    'Ubiquitination': {
        'degradation_loss': {
            'NFKBIA mutation': {
                'disease': 'Hodgkin lymphoma',
                'pathway': 'NF-kB',
                'effect': 'Prevents IκBα degradation, blocks NF-κB',
                'frequency': '30% in cHL',
            },
            'VHL loss': {
                'disease': 'Renal cell carcinoma',
                'pathway': 'HIF/mTOR',
                'effect': 'Loss of HIF degradation',
                'frequency': '75% in clear cell RCC',
            },
            'FBXW7 mutation': {
                'disease': 'Multiple cancers',
                'pathway': 'Cell Cycle',
                'effect': 'Stabilization of oncoproteins (MYC, NOTCH)',
                'frequency': '10% in solid tumors',
            },
        },
        'degradation_gain': {
            'MDM2 amplification': {
                'disease': 'Sarcoma, Glioma',
                'pathway': 'p53',
                'effect': 'Enhanced p53 degradation',
                'frequency': '7% in cancers',
            },
        },
    },

    # 乙酰化相关疾病
    'Acetylation': {
        'histone_modifications': {
            'H3K27M': {
                'disease': 'Diffuse midline glioma',
                'pathway': 'Epigenetic',
                'effect': 'Inhibition of PRC2, loss of H3K27me3',
                'frequency': '80% in DIPG',
            },
            'CREBBP mutation': {
                'disease': 'Lymphoma, Rubinstein-Taybi',
                'pathway': 'Epigenetic',
                'effect': 'Loss of histone acetylation',
                'frequency': '40% in DLBCL',
            },
        },
    },

    # 甲基化相关疾病
    'Methylation': {
        'histone_modifications': {
            'EZH2 Y641 mutation': {
                'disease': 'Lymphoma',
                'pathway': 'Epigenetic/Polycomb',
                'effect': 'Increased H3K27 trimethylation',
                'frequency': '25% in GCB DLBCL',
            },
            'DNMT3A mutation': {
                'disease': 'AML',
                'pathway': 'DNA methylation',
                'effect': 'Aberrant DNA methylation patterns',
                'frequency': '25% in AML',
            },
        },
    },

    # SUMO化相关疾病
    'Sumoylation': {
        'defects': {
            'RNF4 loss': {
                'disease': 'Cancer',
                'pathway': 'DNA Damage',
                'effect': 'Impaired SUMO-targeted degradation',
                'frequency': 'Variable',
            },
        },
    },
}

# ============================================================
# 激酶组分类数据库
# ============================================================

KINOME_CLASSIFICATION = {
    # AGC激酶家族
    'AGC': {
        'description': 'PKA/PKG/PKC family',
        'members': ['AKT1', 'AKT2', 'AKT3', 'PKA', 'PKC', 'PKG', 'SGK', 'RSK', 'PDK1'],
        'key_substrate_motif': 'RXRXX[ST]',
        'pathway_involvement': ['PI3K/AKT', 'MAPK/ERK', 'Cell Cycle'],
    },

    # CAMK激酶家族
    'CAMK': {
        'description': 'Ca2+/Calmodulin-dependent kinases',
        'members': ['CAMK1', 'CAMK2A', 'CAMK2B', 'CAMK2D', 'CAMK4', 'DAPK'],
        'key_substrate_motif': '[KR]..[ST]',
        'pathway_involvement': ['Calcium signaling', 'Synaptic plasticity'],
    },

    # CK1激酶家族
    'CK1': {
        'description': 'Casein kinase 1 family',
        'members': ['CSNK1A1', 'CSNK1D', 'CSNK1E', 'CSNK1G1', 'CSNK1G2', 'CSNK1G3'],
        'key_substrate_motif': 'D..[ST]',
        'pathway_involvement': ['Wnt/beta-catenin', 'Hedgehog', 'Circadian rhythm'],
    },

    # CMGC激酶家族
    'CMGC': {
        'description': 'CDK/MAPK/GSK3/CLK family',
        'members': ['CDK1', 'CDK2', 'CDK4', 'CDK6', 'CDK7', 'MAPK1', 'MAPK3', 'MAPK14',
                    'GSK3B', 'CLK1', 'CLK2', 'CLK3', 'CLK4', 'DYRK1A', 'DYRK1B'],
        'key_substrate_motif': '[ST]P or P.[ST]',
        'pathway_involvement': ['Cell Cycle', 'MAPK/ERK', 'Wnt/beta-catenin', 'DNA Damage'],
    },

    # STE激酶家族
    'STE': {
        'description': 'STE20/STE11/STE7 family',
        'members': ['MAP2K1', 'MAP2K2', 'MAP2K3', 'MAP2K4', 'MAP2K5', 'MAP2K6', 'MAP2K7',
                    'MAP3K1', 'MAP3K2', 'MAP3K3', 'MAP3K5', 'STK3', 'STK4', 'MST1', 'MST2'],
        'key_substrate_motif': 'Variable',
        'pathway_involvement': ['MAPK/ERK', 'Hippo', 'JNK', 'p38'],
    },

    # TK激酶家族
    'TK': {
        'description': 'Tyrosine kinases',
        'members': ['EGFR', 'ERBB2', 'ERBB3', 'ERBB4', 'ABL1', 'ABL2', 'JAK1', 'JAK2', 'JAK3',
                    'TYK2', 'SRC', 'FYN', 'YES1', 'MET', 'ALK', 'ROS1', 'RET', 'FGFR1-4', 'VEGFR1-3'],
        'key_substrate_motif': 'YXX[LIKV] or [DE]Y',
        'pathway_involvement': ['EGFR', 'JAK/STAT', 'MAPK/ERK', 'PI3K/AKT'],
    },

    # TKL激酶家族
    'TKL': {
        'description': 'Tyrosine kinase-like',
        'members': ['BRAF', 'RAF1', 'ARAF', 'TGFBR1', 'TGFBR2', 'ACVR1', 'ACVR2A', 'BMPR1A', 'BMPR2'],
        'key_substrate_motif': '[ST]P or S/T in activation loop',
        'pathway_involvement': ['MAPK/ERK', 'TGF-beta'],
    },

    # RGC激酶家族
    'RGC': {
        'description': 'Receptor guanylyl cyclase',
        'members': ['NPR1', 'NPR2'],
        'key_substrate_motif': 'N/A (produces cGMP)',
        'pathway_involvement': ['cGMP signaling'],
    },

    # Atypical激酶家族
    'Atypical': {
        'description': 'Atypical protein kinases',
        'members': ['mTOR', 'ATM', 'ATR', 'DNAPK', 'PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3CG',
                    'TBK1', 'IKBKB', 'IKBKE', 'PRKAA1', 'PRKAA2'],
        'key_substrate_motif': 'Variable',
        'pathway_involvement': ['mTOR', 'DNA Damage', 'PI3K/AKT', 'NF-kB', 'AMPK'],
    },
}

# ============================================================
# PTM保守性数据库（跨物种）
# ============================================================

PTM_CONSERVATION = {
    # 高度保守的PTM位点
    'highly_conserved': {
        'H3K4me3': {
            'function': 'Active promoter mark',
            'conservation': 'Eukaryotes',
            'disease_relevance': 'Developmental disorders, Cancer',
        },
        'H3K27me3': {
            'function': 'Polycomb repression mark',
            'conservation': 'Eukaryotes',
            'disease_relevance': 'Cancer, Developmental disorders',
        },
        'H3K9me3': {
            'function': 'Heterochromatin mark',
            'conservation': 'Eukaryotes',
            'disease_relevance': 'Cancer, Aging',
        },
        'H2AXS139ph': {
            'function': 'DNA damage marker (γH2AX)',
            'conservation': 'Eukaryotes',
            'disease_relevance': 'Genomic instability, Cancer',
        },
        'RB1S780ph': {
            'function': 'RB inactivation, cell cycle progression',
            'conservation': 'Vertebrates',
            'disease_relevance': 'Cancer',
        },
        'TP53S15ph': {
            'function': 'p53 activation by DNA damage',
            'conservation': 'Vertebrates',
            'disease_relevance': 'Cancer',
        },
    },

    # 功能重要的磷酸化位点
    'critical_phosphosites': {
        'AKT1_T308': {'conservation': 'Vertebrates', 'function': 'AKT activation'},
        'AKT1_S473': {'conservation': 'Mammals', 'function': 'Full AKT activation'},
        'mTOR_S2448': {'conservation': 'Mammals', 'function': 'mTORC1 activation'},
        'GSK3B_S9': {'conservation': 'Vertebrates', 'function': 'GSK3β inhibition'},
        'STAT3_Y705': {'conservation': 'Vertebrates', 'function': 'STAT3 dimerization'},
        'CTNNB1_S33': {'conservation': 'Metazoans', 'function': 'β-catenin degradation'},
        'CTNNB1_S37': {'conservation': 'Metazoans', 'function': 'β-catenin degradation'},
        'CHEK2_T68': {'conservation': 'Vertebrates', 'function': 'CHK2 activation'},
        'ATM_S1981': {'conservation': 'Mammals', 'function': 'ATM activation'},
    },
}

# ============================================================
# 代谢通路PTM调控
# ============================================================

METABOLIC_PTM_REGULATION = {
    # 糖酵解相关PTM
    'glycolysis': {
        'PFKFB3_S461ph': {
            'kinase': 'AMPK',
            'effect': 'Activation',
            'condition': 'Energy stress',
        },
        'PKM2_Y105ph': {
            'kinase': 'FGFR1',
            'effect': 'Inhibition (dimerization)',
            'condition': 'Cancer cell metabolism',
        },
        'PKM2_K305succinylation': {
            'effect': 'Enzyme inhibition',
            'condition': 'High succinate',
        },
    },

    # TCA循环相关PTM
    'tca_cycle': {
        'IDH2_K413ac': {
            'effect': 'Decreased activity',
            'condition': 'High acetyl-CoA',
        },
        'IDH3_S94ph': {
            'kinase': 'PKA',
            'effect': 'Inhibition',
            'condition': 'High energy',
        },
        'SDHA_K547succinylation': {
            'effect': 'Activity modulation',
            'condition': 'Metabolic stress',
        },
    },

    # 脂肪酸代谢相关PTM
    'fatty_acid_metabolism': {
        'ACC1_S79ph': {
            'kinase': 'AMPK',
            'effect': 'Inhibition',
            'condition': 'Energy stress',
        },
        'ACC2_S212ph': {
            'kinase': 'AMPK',
            'effect': 'Inhibition',
            'condition': 'Energy stress',
        },
        'CPT1A_S22ph': {
            'kinase': 'PKA',
            'effect': 'Inhibition',
            'condition': 'High energy',
        },
    },

    # 氧化磷酸化相关PTM
    'oxidative_phosphorylation': {
        'PDHA1_S293ph': {
            'kinase': 'PDK',
            'effect': 'Inhibition',
            'condition': 'Hypoxia, Cancer',
        },
        'ATP5A1_K104ac': {
            'effect': 'Activity regulation',
            'condition': 'Metabolic state',
        },
    },
}

# ============================================================
# 细胞周期检查点PTM
# ============================================================

CELL_CYCLE_CHECKPOINTS = {
    # G1/S检查点
    'G1_S_checkpoint': {
        'RB1_S780ph': {'kinase': 'CDK4/6', 'effect': 'RB inactivation, G1/S transition'},
        'RB1_S795ph': {'kinase': 'CDK4/6', 'effect': 'RB inactivation'},
        'RB1_S807ph': {'kinase': 'CDK2', 'effect': 'Full RB inactivation'},
        'CDKN1A_T145ph': {'kinase': 'AKT', 'effect': 'p21 cytoplasmic localization'},
        'CDC25A_S178ph': {'kinase': 'CHK1', 'effect': 'CDC25A degradation'},
    },

    # S期检查点
    'S_checkpoint': {
        'CDC45_S114ph': {'kinase': 'CDK2', 'effect': 'Origin firing'},
        'MCM2_S139ph': {'kinase': 'ATR', 'effect': 'Replication stress response'},
        'RPA2_S33ph': {'kinase': 'ATR', 'effect': 'DNA damage response'},
    },

    # G2/M检查点
    'G2_M_checkpoint': {
        'CDC25C_S216ph': {'kinase': 'CHK1/CHK2', 'effect': 'CDC25C inactivation'},
        'CDC25C_S287ph': {'kinase': 'PLK1', 'effect': 'CDC25C activation'},
        'CDK1_T14ph': {'kinase': 'WEE1', 'effect': 'CDK1 inhibition'},
        'CDK1_Y15ph': {'kinase': 'WEE1/MYT1', 'effect': 'CDK1 inhibition'},
        'CDK1_T161ph': {'kinase': 'CAK', 'effect': 'CDK1 activation'},
        'WEE1_S642ph': {'kinase': 'PLK1', 'effect': 'WEE1 inactivation'},
    },

    # 纺锤体检查点
    'spindle_checkpoint': {
        'BUB1 T589ph': {'kinase': 'CDK1', 'effect': 'Kinetochore localization'},
        'MPS1 S821ph': {'kinase': 'CDK1', 'effect': 'MPS1 activation'},
        'MAD2 S170ph': {'kinase': 'PLK1', 'effect': 'Checkpoint silencing'},
    },
}


class ExtendedPathwayKnowledgeBase:
    """扩展通路知识库查询接口。

    对 ``src/data/extended_pathway_kb.py`` 中定义的静态字典提供程序化
    查询与检测能力，包括代谢通路 PTM 调控、细胞周期检查点、药物-靶点、
    PTM-疾病关联、激酶组分类和 PTM 保守性数据。
    """

    def __init__(self) -> None:
        self._pathways: Dict[str, Dict[str, Any]] = {
            **{
                name: {"category": "metabolic_ptm_regulation", "sites": sites}
                for name, sites in METABOLIC_PTM_REGULATION.items()
            },
            **{
                name: {"category": "cell_cycle_checkpoint", "sites": sites}
                for name, sites in CELL_CYCLE_CHECKPOINTS.items()
            },
        }

    def get_pathway(self, name: str) -> Dict[str, Any]:
        """按名称查询通路。

        查询范围包括代谢通路 PTM 调控与细胞周期检查点。若未找到，返回空字典。

        Args:
            name: 通路名称，例如 ``"glycolysis"`` 或 ``"G2_M_checkpoint"``。

        Returns:
            包含 ``category`` 与 ``sites`` 的通路字典；未找到时为空字典。
        """
        return self._pathways.get(name, {})

    def search_pathways(self, query: str) -> List[Dict[str, Any]]:
        """按子串搜索通路。

        匹配对大小写不敏感，返回所有通路名称包含 ``query`` 的通路，
        每个结果包含 ``name`` 字段。

        Args:
            query: 搜索关键词。

        Returns:
            匹配到的通路字典列表。
        """
        query_lower = query.lower()
        results: List[Dict[str, Any]] = []
        for name, data in self._pathways.items():
            if query_lower in name.lower():
                results.append({"name": name, **data})
        return results

    def get_kinase_substrates(self, kinase: str) -> List[Dict[str, Any]]:
        """查询某个激酶调控的 PTM 底物位点。

        在代谢通路 PTM 调控与细胞周期检查点字典中搜索 ``kinase`` 字段
        匹配的条目（大小写不敏感）。

        Args:
            kinase: 激酶名称，例如 ``"AMPK"`` 或 ``"CDK1"``。

        Returns:
            底物位点注释列表，每个条目包含 ``pathway``、``category``、
            ``site`` 以及原始注释字段。
        """
        kinase_lower = kinase.lower()
        substrates: List[Dict[str, Any]] = []
        for category, source in (
            ("metabolic_ptm_regulation", METABOLIC_PTM_REGULATION),
            ("cell_cycle_checkpoint", CELL_CYCLE_CHECKPOINTS),
        ):
            for pathway_name, sites in source.items():
                for site_id, annotation in sites.items():
                    site_kinase = annotation.get("kinase", "")
                    if isinstance(site_kinase, str) and kinase_lower in site_kinase.lower():
                        substrates.append(
                            {
                                "pathway": pathway_name,
                                "category": category,
                                "site": site_id,
                                **annotation,
                            }
                        )
        return substrates

    def get_ptm_annotations(self, accession: str) -> List[Dict[str, Any]]:
        """查询与给定蛋白名称/登录号相关的 PTM 注释。

        根据 PTM 位点键的前缀匹配蛋白标识符（例如 ``"PFKFB3"`` 可匹配
        ``"PFKFB3_S461ph"``）。搜索范围包括代谢调控、细胞周期检查点、
        PTM 保守性和 PTM-疾病关联字典。

        Args:
            accession: 蛋白名称或登录号。

        Returns:
            PTM 注释列表。
        """
        accession_lower = accession.lower()
        annotations: List[Dict[str, Any]] = []

        for category, source in (
            ("metabolic_ptm_regulation", METABOLIC_PTM_REGULATION),
            ("cell_cycle_checkpoint", CELL_CYCLE_CHECKPOINTS),
        ):
            for pathway_name, sites in source.items():
                for site_id, annotation in sites.items():
                    if site_id.lower().startswith(accession_lower):
                        annotations.append(
                            {
                                "pathway": pathway_name,
                                "category": category,
                                "site": site_id,
                                **annotation,
                            }
                        )

        for conservation_level, sites in PTM_CONSERVATION.items():
            for site_id, annotation in sites.items():
                if site_id.lower().startswith(accession_lower):
                    annotations.append(
                        {
                            "category": "ptm_conservation",
                            "conservation_level": conservation_level,
                            "site": site_id,
                            **annotation,
                        }
                    )

        for ptm_type, effect_groups in PTM_DISEASE_ASSOCIATIONS.items():
            for effect_type, variants in effect_groups.items():
                for variant_id, annotation in variants.items():
                    if variant_id.lower().startswith(accession_lower):
                        annotations.append(
                            {
                                "category": "ptm_disease_association",
                                "ptm_type": ptm_type,
                                "effect_type": effect_type,
                                "variant": variant_id,
                                **annotation,
                            }
                        )

        return annotations

    def detect_crosstalk(
        self, pathway_a: str, pathway_b: str
    ) -> Dict[str, Any]:
        """检测两条通路之间潜在的串扰。

        比较两条通路（来自代谢调控或细胞周期检查点）的激酶集合与 PTM
        位点集合，返回共享激酶、共享位点及简单的重叠评分。

        Args:
            pathway_a: 第一条通路名称。
            pathway_b: 第二条通路名称。

        Returns:
            包含 ``shared_kinases``、``shared_sites``、``crosstalk_score``
            与 ``detected`` 的结果字典。若任一通路不存在，评分为 0。
        """
        info_a = self.get_pathway(pathway_a)
        info_b = self.get_pathway(pathway_b)

        if not info_a or not info_b:
            return {
                "pathway_a": pathway_a,
                "pathway_b": pathway_b,
                "shared_kinases": [],
                "shared_sites": [],
                "crosstalk_score": 0.0,
                "detected": False,
            }

        def _extract_sets(info: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
            sites = info.get("sites", {})
            kinases: Set[str] = set()
            site_ids: Set[str] = set()
            for site_id, annotation in sites.items():
                site_ids.add(site_id)
                site_kinase = annotation.get("kinase")
                if isinstance(site_kinase, str):
                    kinases.add(site_kinase.strip())
            return kinases, site_ids

        kinases_a, sites_a = _extract_sets(info_a)
        kinases_b, sites_b = _extract_sets(info_b)

        shared_kinases = sorted(kinases_a & kinases_b)
        shared_sites = sorted(sites_a & sites_b)

        total_unique = len(kinases_a | kinases_b) + len(sites_a | sites_b)
        shared_count = len(shared_kinases) + len(shared_sites)
        score = shared_count / total_unique if total_unique else 0.0

        return {
            "pathway_a": pathway_a,
            "pathway_b": pathway_b,
            "shared_kinases": shared_kinases,
            "shared_sites": shared_sites,
            "crosstalk_score": round(score, 4),
            "detected": bool(shared_kinases or shared_sites),
        }


__all__ = [
    "ExtendedPathwayKnowledgeBase",
    "DRUG_TARGET_DATABASE",
    "PTM_DISEASE_ASSOCIATIONS",
    "KINOME_CLASSIFICATION",
    "PTM_CONSERVATION",
    "METABOLIC_PTM_REGULATION",
    "CELL_CYCLE_CHECKPOINTS",
]

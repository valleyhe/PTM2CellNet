"""
扩展信号通路知识库
功能: 更全面的信号通路、激酶-底物关系、PTM位点注释
"""

# ============================================================
# 扩展信号通路定义
# ============================================================

EXTENDED_SIGNALING_PATHWAYS = {
    # ============ 原有通路（增强版） ============
    'MAPK/ERK': {
        'description': 'RAS-RAF-MEK-ERK通路 - 细胞增殖和分化',
        'key_kinases': ['BRAF', 'RAF1', 'ARAF', 'MAP2K1', 'MAP2K2', 'MAPK1', 'MAPK3', 'MAPK7'],
        'key_substrates': ['EGFR', 'ERBB2', 'KRAS', 'NRAS', 'HRAS', 'GRB2', 'SOS1', 'SHC1'],
        'ptm_types': ['Phosphorylation'],
        'output_genes': ['FOS', 'JUN', 'MYC', 'EGR1', 'ELK1', 'ETS1', 'DUSP1', 'DUSP6'],
        'diseases': ['黑色素瘤', '结直肠癌', '肺癌'],
        'crosstalk': ['PI3K/AKT', 'JAK/STAT', 'Wnt/beta-catenin'],
        'key_sites': {
            'BRAF': {'V600': 'activation_mutation'},
            'MAPK1': {'T185': 'activation', 'Y187': 'activation'},
            'MAPK3': {'T202': 'activation', 'Y204': 'activation'},
        },
    },

    'PI3K/AKT': {
        'description': 'PI3K-AKT-mTOR通路 - 细胞存活和代谢',
        'key_kinases': ['PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3CG', 'AKT1', 'AKT2', 'AKT3', 'MTOR', 'PDK1'],
        'key_substrates': ['PTEN', 'TSC2', 'GSK3B', 'FOXO1', 'FOXO3', 'BAD', 'CASP9'],
        'ptm_types': ['Phosphorylation'],
        'output_genes': ['FOXO1', 'FOXO3', 'RPS6KB1', 'EIF4EBP1', 'HIF1A', 'GLUT1'],
        'diseases': ['乳腺癌', '子宫内膜癌', '胶质瘤'],
        'crosstalk': ['MAPK/ERK', 'mTOR', 'NF-kB'],
        'key_sites': {
            'AKT1': {'T308': 'activation', 'S473': 'full_activation'},
            'PTEN': {'S380': 'stability', 'T382': 'stability'},
            'MTOR': {'S2448': 'activation', 'S2481': 'autophosphorylation'},
        },
    },

    'JAK/STAT': {
        'description': 'JAK-STAT信号通路 - 免疫和造血',
        'key_kinases': ['JAK1', 'JAK2', 'JAK3', 'TYK2'],
        'key_substrates': ['STAT1', 'STAT2', 'STAT3', 'STAT4', 'STAT5A', 'STAT5B', 'STAT6'],
        'ptm_types': ['Phosphorylation'],
        'output_genes': ['SOCS1', 'SOCS2', 'SOCS3', 'IRF1', 'IRF9', 'CXCL10', 'IFNG'],
        'diseases': ['骨髓增殖性疾病', '白血病', '自身免疫病'],
        'crosstalk': ['PI3K/AKT', 'MAPK/ERK', 'NF-kB'],
        'key_sites': {
            'STAT3': {'Y705': 'dimerization', 'S727': 'transcriptional_activity'},
            'STAT5A': {'Y694': 'activation'},
            'JAK2': {'Y1007': 'activation', 'Y1008': 'activation'},
        },
    },

    'NF-kB': {
        'description': 'NF-kB炎症信号通路 - 炎症和免疫',
        'key_kinases': ['IKBKB', 'IKBKA', 'CHUK', 'IKBKE', 'TBK1', 'NFKBIA'],
        'key_substrates': ['RELA', 'NFKB1', 'NFKB2', 'RELB', 'NFKBIA', 'NFKBIB', 'NFKBIE'],
        'ptm_types': ['Phosphorylation', 'Ubiquitination', 'Acetylation'],
        'output_genes': ['TNF', 'IL1B', 'IL6', 'IL8', 'CXCL10', 'CCL2', 'NOS2', 'COX2'],
        'diseases': ['炎症性疾病', '自身免疫病', '癌症'],
        'crosstalk': ['JAK/STAT', 'MAPK/ERK', 'Apoptosis'],
        'key_sites': {
            'RELA': {'S536': 'transactivation', 'K310': 'acetylation'},
            'IKBKB': {'S177': 'activation', 'S181': 'activation'},
            'NFKBIA': {'S32': 'degradation', 'S36': 'degradation'},
        },
    },

    'Wnt/beta-catenin': {
        'description': 'Wnt信号通路 - 发育和干细胞维持',
        'key_kinases': ['GSK3B', 'CSNK1A1', 'CSNK1D', 'CSNK1E', 'LRRK2'],
        'key_substrates': ['CTNNB1', 'APC', 'AXIN1', 'AXIN2', 'TCF7', 'LEF1'],
        'ptm_types': ['Phosphorylation', 'Ubiquitination'],
        'output_genes': ['MYC', 'CCND1', 'AXIN2', 'LEF1', 'TCF7', 'DKK1', 'SP5'],
        'diseases': ['结直肠癌', '肝癌', '骨骼疾病'],
        'crosstalk': ['Hippo', 'Notch', 'PI3K/AKT'],
        'key_sites': {
            'CTNNB1': {'S33': 'degradation', 'S37': 'degradation', 'T41': 'degradation', 'S45': 'priming'},
            'GSK3B': {'S9': 'inhibition', 'Y216': 'activation'},
        },
    },

    'Cell Cycle': {
        'description': '细胞周期调控 - 细胞分裂',
        'key_kinases': ['CDK1', 'CDK2', 'CDK4', 'CDK6', 'CDK7', 'PLK1', 'AURKA', 'AURKB'],
        'key_substrates': ['RB1', 'TP53', 'CCND1', 'CCNE1', 'CCNA2', 'CCNB1', 'CDC25A', 'CDC25C'],
        'ptm_types': ['Phosphorylation', 'Ubiquitination'],
        'output_genes': ['E2F1', 'E2F2', 'E2F3', 'CDKN1A', 'CDKN1B', 'CDKN2A', 'MCM2', 'PCNA'],
        'diseases': ['癌症', '发育障碍'],
        'crosstalk': ['DNA Damage', 'Apoptosis', 'PI3K/AKT'],
        'key_sites': {
            'RB1': {'S780': 'inactivation', 'S795': 'inactivation', 'S807': 'inactivation'},
            'TP53': {'S15': 'activation', 'S20': 'stabilization', 'K382': 'acetylation'},
            'CDK1': {'T14': 'inhibition', 'Y15': 'inhibition', 'T161': 'activation'},
        },
    },

    'Apoptosis': {
        'description': '细胞凋亡通路 - 程序性细胞死亡',
        'key_kinases': ['CASP3', 'CASP8', 'CASP9', 'CASP7', 'RIPK1', 'RIPK3'],
        'key_substrates': ['BCL2', 'BAX', 'BAK1', 'BCL2L1', 'PARP1', 'XIAP', 'MCL1'],
        'ptm_types': ['Phosphorylation', 'Ubiquitination', 'Cleavage'],
        'output_genes': ['BCL2L11', 'PMAIP1', 'BBC3', 'DIABLO', 'CYCS'],
        'diseases': ['癌症', '神经退行性疾病', '自身免疫病'],
        'crosstalk': ['NF-kB', 'Cell Cycle', 'DNA Damage'],
        'key_sites': {
            'BCL2': {'S70': 'inactivation', 'T56': 'inactivation'},
            'BAX': {'S184': 'activation'},
            'MCL1': {'S159': 'degradation', 'T163': 'stability'},
        },
    },

    'DNA Damage': {
        'description': 'DNA损伤修复 - 基因组稳定性',
        'key_kinases': ['ATM', 'ATR', 'CHEK1', 'CHEK2', 'TP53', 'DNAPK', 'PRKDC'],
        'key_substrates': ['H2AX', 'BRCA1', 'BRCA2', 'RAD51', 'NBN', 'MRE11', 'RBBP8'],
        'ptm_types': ['Phosphorylation', 'Ubiquitination', 'Sumoylation', 'Acetylation'],
        'output_genes': ['CDKN1A', 'GADD45A', 'GADD45B', 'BAX', 'PUMA', 'NOXA'],
        'diseases': ['癌症', '基因组不稳定综合征'],
        'crosstalk': ['Cell Cycle', 'Apoptosis', 'NF-kB'],
        'key_sites': {
            'ATM': {'S1981': 'activation'},
            'CHEK2': {'T68': 'activation'},
            'H2AX': {'S139': 'gamma_H2AX'},
            'TP53': {'S15': 'activation', 'S46': 'apoptosis'},
        },
    },

    # ============ 新增通路 ============

    'mTOR': {
        'description': 'mTOR复合物通路 - 细胞生长和代谢',
        'key_kinases': ['MTOR', 'RICTOR', 'RPTOR', 'MLST8', 'AKT1S1', 'DEPTOR'],
        'key_substrates': ['RPS6KB1', 'EIF4EBP1', 'ULK1', 'TFEB', 'SREBF1', 'PPARGC1A'],
        'ptm_types': ['Phosphorylation', 'Ubiquitination', 'Acetylation'],
        'output_genes': ['RPS6', 'EIF4E', 'LC3', 'LAMP1', 'GLUT4', 'HMGCR'],
        'diseases': ['癌症', '糖尿病', '神经退行性疾病'],
        'crosstalk': ['PI3K/AKT', 'AMPK', 'Autophagy'],
        'key_sites': {
            'MTOR': {'S2448': 'mTORC1_activation', 'S2481': 'mTORC2'},
            'RPS6KB1': {'T389': 'activation', 'T229': 'activation'},
            'ULK1': {'S757': 'inhibition', 'S317': 'activation'},
        },
    },

    'Hippo': {
        'description': 'Hippo信号通路 - 器官大小控制',
        'key_kinases': ['STK3', 'STK4', 'LATS1', 'LATS2', 'TAOK1', 'TAOK2', 'TAOK3'],
        'key_substrates': ['YAP1', 'WWTR1', 'TEAD1', 'TEAD2', 'TEAD3', 'TEAD4'],
        'ptm_types': ['Phosphorylation', 'Ubiquitination'],
        'output_genes': ['CTGF', 'CYR61', 'ANKRD1', 'AJUBA', 'AMOT'],
        'diseases': ['癌症', '发育异常'],
        'crosstalk': ['Wnt/beta-catenin', 'TGF-beta', 'Notch'],
        'key_sites': {
            'YAP1': {'S127': 'cytoplasmic_retention', 'S397': 'degradation'},
            'LATS1': {'T1079': 'activation'},
            'STK4': {'T183': 'activation'},
        },
    },

    'Notch': {
        'description': 'Notch信号通路 - 细胞命运决定',
        'key_kinases': ['ADAM10', 'ADAM17', 'PSEN1', 'PSEN2', 'NCSTN', 'APH1A'],
        'key_substrates': ['NOTCH1', 'NOTCH2', 'NOTCH3', 'NOTCH4', 'DLL1', 'DLL4', 'JAG1', 'JAG2'],
        'ptm_types': ['Phosphorylation', 'Ubiquitination', 'Cleavage'],
        'output_genes': ['HES1', 'HES5', 'HEY1', 'HEY2', 'MYC', 'CCND1'],
        'diseases': ['T-ALL', 'CADASIL', 'Alagille综合征'],
        'crosstalk': ['Wnt/beta-catenin', 'Hippo', 'TGF-beta'],
        'key_sites': {
            'NOTCH1': {'S2100': 'activation', 'T2121': 'PEST_degradation'},
            'NICD': {'S2121': 'nuclear_translocation'},
        },
    },

    'TGF-beta': {
        'description': 'TGF-beta信号通路 - 细胞增殖和分化',
        'key_kinases': ['TGFBR1', 'TGFBR2', 'ACVR1', 'ACVR2A', 'ACVR2B', 'BMPR1A', 'BMPR2'],
        'key_substrates': ['SMAD2', 'SMAD3', 'SMAD1', 'SMAD5', 'SMAD8', 'SMAD4'],
        'ptm_types': ['Phosphorylation', 'Ubiquitination', 'Sumoylation'],
        'output_genes': ['CDKN1A', 'CDKN2B', 'PAI1', 'COL1A1', 'COL3A1', 'FN1'],
        'diseases': ['癌症', '纤维化', 'Marfan综合征'],
        'crosstalk': ['MAPK/ERK', 'PI3K/AKT', 'Wnt/beta-catenin'],
        'key_sites': {
            'SMAD2': {'S465': 'activation', 'S467': 'activation'},
            'SMAD3': {'S423': 'activation', 'S425': 'activation'},
            'TGFBR1': {'S165': 'activation'},
        },
    },

    'Hedgehog': {
        'description': 'Hedgehog信号通路 - 发育模式',
        'key_kinases': ['CSNK1A1', 'CSNK1G2', 'PRKACA', 'GRK2'],
        'key_substrates': ['SMO', 'GLI1', 'GLI2', 'GLI3', 'PTCH1', 'PTCH2', 'SUFU'],
        'ptm_types': ['Phosphorylation', 'Ubiquitination', 'Cleavage'],
        'output_genes': ['GLI1', 'PTCH1', 'HHIP', 'BCL2', 'CCND1', 'MYCN'],
        'diseases': ['基底细胞癌', '髓母细胞瘤'],
        'crosstalk': ['Wnt/beta-catenin', 'PI3K/AKT', 'Notch'],
        'key_sites': {
            'SMO': {'S560': 'activation', 'S568': 'inhibition'},
            'GLI2': {'S230': 'activation'},
            'GLI3': {'S852': 'repressor_formation'},
        },
    },

    'AMPK': {
        'description': 'AMPK信号通路 - 能量感应',
        'key_kinases': ['PRKAA1', 'PRKAA2', 'PRKAB1', 'PRKAB2', 'PRKAG1', 'PRKAG2', 'PRKAG3'],
        'key_substrates': ['ACC1', 'ACC2', 'Raptor', 'ULK1', 'TSC2', 'SREBF1'],
        'ptm_types': ['Phosphorylation', 'Ubiquitination', 'Acetylation'],
        'output_genes': ['CPT1A', 'PGC1A', 'GLUT4', 'SREBF1', 'HMGCR'],
        'diseases': ['糖尿病', '肥胖', '癌症'],
        'crosstalk': ['mTOR', 'PI3K/AKT', 'Autophagy'],
        'key_sites': {
            'PRKAA1': {'T172': 'activation'},
            'ACC1': {'S79': 'inhibition'},
            'Raptor': {'S792': 'mTORC1_inhibition'},
        },
    },

    'Autophagy': {
        'description': '自噬通路 - 细胞自我消化',
        'key_kinases': ['ULK1', 'ULK2', 'ATG1', 'ATG13', 'VPS34', 'BECN1', 'PIK3C3'],
        'key_substrates': ['LC3', 'SQSTM1', 'ATG5', 'ATG7', 'ATG12', 'ATG16L1'],
        'ptm_types': ['Phosphorylation', 'Ubiquitination', 'Lipidation'],
        'output_genes': ['LC3B', 'SQSTM1', 'LAMP2', 'TFEB', 'CTSB', 'CTSD'],
        'diseases': ['神经退行性疾病', '癌症', '感染'],
        'crosstalk': ['mTOR', 'AMPK', 'Apoptosis'],
        'key_sites': {
            'ULK1': {'S757': 'mTOR_inhibition', 'S317': 'AMPK_activation', 'S777': 'activation'},
            'BECN1': {'S14': 'activation', 'K437': 'ubiquitination'},
            'LC3': {'K51': 'lipidation'},
        },
    },

    'p53': {
        'description': 'p53肿瘤抑制通路',
        'key_kinases': ['TP53', 'MDM2', 'ATM', 'ATR', 'CHEK1', 'CHEK2', 'CDKN2A'],
        'key_substrates': ['TP53', 'MDM2', 'BAX', 'PUMA', 'NOXA', 'CDKN1A', 'GADD45'],
        'ptm_types': ['Phosphorylation', 'Ubiquitination', 'Acetylation', 'Methylation', 'Sumoylation'],
        'output_genes': ['CDKN1A', 'BAX', 'PUMA', 'NOXA', 'GADD45A', 'MDM2', 'TP53I3'],
        'diseases': ['Li-Fraumeni综合征', '癌症'],
        'crosstalk': ['DNA Damage', 'Apoptosis', 'Cell Cycle'],
        'key_sites': {
            'TP53': {
                'S15': 'ATM_activation', 'S20': 'CHEK2_stabilization',
                'K382': 'acetylation', 'K370': 'methylation', 'K386': 'sumoylation'
            },
            'MDM2': {'S166': 'activation', 'K446': 'auto_ubiquitination'},
        },
    },

    'EGFR': {
        'description': 'EGFR信号通路 - 生长因子信号',
        'key_kinases': ['EGFR', 'ERBB2', 'ERBB3', 'ERBB4', 'SRC', 'ABL1'],
        'key_substrates': ['GRB2', 'SHC1', 'GAB1', 'PLCg1', 'STAT3', 'PIK3CA'],
        'ptm_types': ['Phosphorylation', 'Ubiquitination'],
        'output_genes': ['MYC', 'FOS', 'JUN', 'CCND1', 'MCL1', 'BCL2L1'],
        'diseases': ['肺癌', '乳腺癌', '结直肠癌'],
        'crosstalk': ['MAPK/ERK', 'PI3K/AKT', 'JAK/STAT'],
        'key_sites': {
            'EGFR': {'Y1068': 'activation', 'Y1173': 'activation', 'K721': 'ATP_binding'},
            'ERBB2': {'Y1248': 'activation', 'Y877': 'activation'},
        },
    },
}

# ============================================================
# 激酶-底物关系数据库
# ============================================================

KINASE_SUBSTRATE_DATABASE = {
    # 激酶特异性底物位点
    'CDK1': {
        'substrates': ['RB1', 'TP53', 'CDC25C', 'BCL2', 'LAMIN A/C'],
        'consensus': '[ST]PX[KR]',
        'key_sites': {
            'RB1': ['S780', 'S795', 'S807'],
            'TP53': ['S315'],
            'CDC25C': ['S216'],
        },
    },

    'CDK2': {
        'substrates': ['RB1', 'TP53', 'CDC25A', 'CDC6', 'NPAT'],
        'consensus': '[ST]PX[KR]',
        'key_sites': {
            'RB1': ['S567', 'S608', 'S612'],
            'TP53': ['S315'],
        },
    },

    'CDK4/6': {
        'substrates': ['RB1', 'SMAD3', 'FOXO1', 'MTOR'],
        'consensus': '[ST]PX[KR]',
        'key_sites': {
            'RB1': ['S780', 'S795', 'S807', 'S811'],
            'SMAD3': ['S8'],
        },
    },

    'MAPK1/3': {
        'substrates': ['ELK1', 'RSK', 'MNK', 'MSK1', 'C-MYC', 'STAT1', 'STAT3'],
        'consensus': 'PX[ST]P',
        'key_sites': {
            'ELK1': ['S383', 'S389'],
            'RSK': ['T360', 'S363'],
            'STAT3': ['S727'],
        },
    },

    'AKT1': {
        'substrates': ['GSK3B', 'FOXO1', 'FOXO3', 'TSC2', 'BAD', 'CASP9', 'MDM2', 'mTOR'],
        'consensus': 'RXRXX[ST]',
        'key_sites': {
            'GSK3B': ['S9'],
            'FOXO1': ['T24', 'S256', 'S319'],
            'TSC2': ['S939', 'S981', 'T1462'],
            'BAD': ['S136', 'S112'],
        },
    },

    'GSK3B': {
        'substrates': ['CTNNB1', 'Glycogen Synthase', 'TP53', 'MYC', 'NOTCH1', 'SNAIL'],
        'consensus': '[ST]XXX[ST]P',
        'key_sites': {
            'CTNNB1': ['S33', 'S37', 'T41', 'S45'],
            'TP53': ['S33', 'S37'],
            'MYC': ['T58'],
            'NOTCH1': ['S2100', 'T2121'],
        },
    },

    'ATM': {
        'substrates': ['TP53', 'CHEK2', 'H2AX', 'NBN', 'BRCA1', 'ATR'],
        'consensus': 'SQ/TQ',
        'key_sites': {
            'TP53': ['S15'],
            'CHEK2': ['T68'],
            'H2AX': ['S139'],
            'BRCA1': ['S1524'],
        },
    },

    'ATR': {
        'substrates': ['CHEK1', 'TP53', 'H2AX', 'BRCA1', 'SMARCAL1'],
        'consensus': 'SQ/TQ',
        'key_sites': {
            'CHEK1': ['S317', 'S345'],
            'TP53': ['S15', 'S37'],
            'H2AX': ['S139'],
        },
    },

    'mTOR': {
        'substrates': ['RPS6KB1', 'EIF4EBP1', 'ULK1', 'STAT3', 'HIF1A'],
        'consensus': '[ST]P',
        'key_sites': {
            'RPS6KB1': ['T389'],
            'EIF4EBP1': ['T37', 'T46', 'S65'],
            'ULK1': ['S757'],
            'STAT3': ['S727'],
        },
    },

    'JAK2': {
        'substrates': ['STAT3', 'STAT5', 'STAT1', 'STAT6'],
        'consensus': 'YXX[LIKV]',
        'key_sites': {
            'STAT3': ['Y705'],
            'STAT5': ['Y694'],
            'STAT1': ['Y701'],
        },
    },

    'IKK': {
        'substrates': ['NFKBIA', 'NFKBIB', 'RELA', 'IKBKB'],
        'consensus': 'DSGXXS',
        'key_sites': {
            'NFKBIA': ['S32', 'S36'],
            'NFKBIB': ['S19', 'S23'],
            'RELA': ['S536'],
        },
    },
}

# ============================================================
# PTM位点特异性功能注释
# ============================================================

PTM_SITE_ANNOTATIONS = {
    # 磷酸化位点
    'Phosphorylation': {
        'activating_sites': {
            'MAPK1_T185': '激活MAPK激酶活性',
            'MAPK1_Y187': '激活MAPK激酶活性',
            'MAPK3_T202': '激活ERK1激酶活性',
            'MAPK3_Y204': '激活ERK1激酶活性',
            'AKT1_T308': '激活AKT激酶活性',
            'AKT1_S473': 'AKT完全激活',
            'mTOR_S2448': '激活mTORC1',
            'STAT3_Y705': 'STAT3二聚化',
            'CHEK2_T68': '激活CHEK2',
            'ATM_S1981': '激活ATM',
        },
        'inhibitory_sites': {
            'GSK3B_S9': '抑制GSK3β活性',
            'BCL2_S70': '抑制BCL2抗凋亡功能',
            'CDC25C_S216': '抑制CDC25C活性',
            'YAP1_S127': 'YAP1滞留在细胞质',
            'ULK1_S757': '抑制自噬',
        },
        'degradation_sites': {
            'CTNNB1_S33': 'β-catenin降解',
            'CTNNB1_S37': 'β-catenin降解',
            'CTNNB1_T41': 'β-catenin降解',
            'NFKBIA_S32': 'IκBα降解',
            'NFKBIA_S36': 'IκBα降解',
            'MYC_T58': 'c-Myc降解',
        },
    },

    # 泛素化位点
    'Ubiquitination': {
        'degradation_sites': {
            'TP53_K370': 'p53降解',
            'TP53_K372': 'p53降解',
            'TP53_K373': 'p53降解',
            'CTNNB1_K19': 'β-catenin降解',
            'NOTCH1_K1855': 'Notch降解',
            'MYC_K147': 'c-Myc降解',
        },
        'signaling_sites': {
            'RIPK1_K63': 'NF-κB激活',
            'TRAF6_K63': 'NF-κB激活',
            'IKBKA_K285': 'IKK激活',
        },
    },

    # 乙酰化位点
    'Acetylation': {
        'activation_sites': {
            'TP53_K382': 'p53转录激活',
            'TP53_K370': 'p53 DNA结合',
            'H3K27': '基因激活标记',
            'H3K9': '基因激活标记',
        },
        'regulation_sites': {
            'CTNNB1_K345': 'β-catenin核转位',
            'RELA_K310': 'NF-κB转录活性',
            'STAT3_K685': 'STAT3二聚化',
        },
    },

    # 甲基化位点
    'Methylation': {
        'activation_sites': {
            'H3K4': '基因激活标记',
            'H3K36': '转录延伸',
            'H3K79': '基因激活',
        },
        'repression_sites': {
            'H3K9': '异染色质',
            'H3K27': 'Polycomb抑制',
        },
        'protein_sites': {
            'TP53_K370': 'p53抑制',
            'TP53_K382': 'p53激活',
            'RB1_K810': 'RB激活',
        },
    },

    # SUMO化位点
    'Sumoylation': {
        'nuclear_sites': {
            'TP53_K386': 'p53核定位',
            'RELA_K310': 'NF-κB核定位',
            'CTNNB1_K435': 'β-catenin核定位',
        },
        'regulation_sites': {
            'FOXO1_K262': 'FOXO转录抑制',
            'HIF1A_K391': 'HIF-1α稳定性',
        },
    },

    # 琥珀酰化位点
    'Succinylation': {
        'metabolic_sites': {
            'GAPDH_K184': '糖酵解调节',
            'IDH2_K180': 'TCA循环',
            'SDHA_K547': '电子传递链',
            'SOD1_K122': '氧化应激',
        },
    },
}

# ============================================================
# 通路之间的Cross-talk关系
# ============================================================

PATHWAY_CROSSTALK = {
    ('MAPK/ERK', 'PI3K/AKT'): {
        'type': 'synergistic',
        'mechanism': 'RAS同时激活MAPK和PI3K通路',
        'key_nodes': ['RAS', 'EGFR', 'SRC'],
        'functional_outcome': '细胞增殖和存活协同',
    },

    ('PI3K/AKT', 'mTOR'): {
        'type': 'cascade',
        'mechanism': 'AKT磷酸化TSC2激活mTOR',
        'key_nodes': ['AKT', 'TSC2', 'mTOR'],
        'functional_outcome': '细胞生长和代谢',
    },

    ('mTOR', 'Autophagy'): {
        'type': 'inhibitory',
        'mechanism': 'mTOR磷酸化ULK1抑制自噬',
        'key_nodes': ['mTOR', 'ULK1'],
        'functional_outcome': '细胞应激响应',
    },

    ('AMPK', 'mTOR'): {
        'type': 'inhibitory',
        'mechanism': 'AMPK磷酸化Raptor抑制mTORC1',
        'key_nodes': ['AMPK', 'Raptor', 'mTOR'],
        'functional_outcome': '能量应激下抑制生长',
    },

    ('NF-kB', 'Apoptosis'): {
        'type': 'antagonistic',
        'mechanism': 'NF-κB上调BCL2抑制凋亡',
        'key_nodes': ['RELA', 'BCL2', 'BAX'],
        'functional_outcome': '炎症抑制凋亡',
    },

    ('DNA Damage', 'Apoptosis'): {
        'type': 'cascade',
        'mechanism': 'DNA损伤激活p53诱导凋亡',
        'key_nodes': ['ATM', 'TP53', 'BAX', 'PUMA'],
        'functional_outcome': '损伤无法修复时细胞死亡',
    },

    ('Wnt/beta-catenin', 'Hippo'): {
        'type': 'synergistic',
        'mechanism': 'YAP和β-catenin协同调控下游基因',
        'key_nodes': ['YAP1', 'CTNNB1', 'TEAD'],
        'functional_outcome': '细胞增殖和器官大小',
    },

    ('Notch', 'Wnt/beta-catenin'): {
        'type': 'bidirectional',
        'mechanism': 'Notch和Wnt互相调控',
        'key_nodes': ['NOTCH1', 'CTNNB1', 'GSK3B'],
        'functional_outcome': '干细胞命运决定',
    },

    ('TGF-beta', 'MAPK/ERK'): {
        'type': 'context_dependent',
        'mechanism': 'TGF-β可激活或抑制MAPK',
        'key_nodes': ['TGFBR1', 'SMAD', 'RAS'],
        'functional_outcome': '上皮-间质转化',
    },

    ('p53', 'Cell Cycle'): {
        'type': 'inhibitory',
        'mechanism': 'p53上调p21抑制CDK',
        'key_nodes': ['TP53', 'CDKN1A', 'CDK'],
        'functional_outcome': '细胞周期停滞',
    },
}

# ============================================================
# 癌症相关的通路变异
# ============================================================

CANCER_PATHWAY_VARIANTS = {
    'melanoma': {
        'common_mutations': ['BRAF V600E', 'NRAS Q61', 'KIT D816'],
        'affected_pathways': ['MAPK/ERK', 'PI3K/AKT', 'KIT'],
        'therapeutic_targets': ['BRAF', 'MEK', 'PD-1'],
    },

    'lung_adenocarcinoma': {
        'common_mutations': ['EGFR L858R', 'KRAS G12', 'ALK fusion'],
        'affected_pathways': ['EGFR', 'MAPK/ERK', 'PI3K/AKT'],
        'therapeutic_targets': ['EGFR', 'ALK', 'MEK'],
    },

    'colorectal_cancer': {
        'common_mutations': ['APC truncation', 'KRAS G12', 'BRAF V600E', 'PIK3CA H1047'],
        'affected_pathways': ['Wnt/beta-catenin', 'MAPK/ERK', 'PI3K/AKT'],
        'therapeutic_targets': ['EGFR', 'VEGFR', 'BRAF'],
    },

    'breast_cancer': {
        'common_mutations': ['PIK3CA H1047', 'TP53 R175', 'ERBB2 amplification'],
        'affected_pathways': ['PI3K/AKT', 'p53', 'EGFR'],
        'therapeutic_targets': ['ER', 'HER2', 'CDK4/6', 'PI3K'],
    },

    'glioblastoma': {
        'common_mutations': ['EGFRvIII', 'PTEN loss', 'IDH1 R132'],
        'affected_pathways': ['EGFR', 'PI3K/AKT', 'DNA Damage'],
        'therapeutic_targets': ['EGFR', 'VEGF', 'mTOR'],
    },
}
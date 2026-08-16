"""
氨基酸编码常量 — 单一定义源 (Single Source of Truth)

所有氨基酸字母表、索引映射、非标准字符映射统一在此定义。
其他模块必须从此处导入，禁止重复定义。

约定:
  - 序列索引从1开始，0保留给padding (1-indexed, PAD_IDX=0)
  - AMINO_ACIDS 为 tuple（不可变），防止调用方意外修改
"""

from typing import Dict, Set

# 20种标准氨基酸字母表（不可变tuple）
AMINO_ACIDS: tuple = ("A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
                       "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y")

# 标准氨基酸字符串（用于需要str的场景，如set()或遍历）
AMINO_ACIDS_STR: str = "ACDEFGHIKLMNPQRSTVWY"

# 氨基酸→索引映射（1-indexed，0保留给padding）
AA_TO_IDX: dict = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}

# Padding
PAD_IDX: int = 0
AA_PAD_CHAR: str = "-"

# 支持的PTM修饰类型（扩展列表）
SUPPORTED_PTM_TYPES: tuple = (
    "O-GlcNAcylation",
    "S-nitrosylation",
    "S-palmitoylation",
    "N-myristoylation",
    "S-prenylation",
    "Disulfide bond",
    "Citrullination",
    "ADP-ribosylation",
    "Lactylation",
    "Crotonylation",
    "Propionylation",
    "Butyrylation",
    "Formylation",
    "Sulfation",
    "Carbonylation",
)

# Canonical (lowercase, no-prefix, no-hyphen) PTM type names for consistent lookup
# Maps from any variant to canonical form for normalization
PTM_TYPE_ALIASES: Dict[str, str] = {
    # aa_constants style → ptm_direction_mapper style
    "o-glcnacylation": "oglcnacylation",
    "s-nitrosylation": "nitrosylation",
    "nitrosylation": "nitrosylation",
    "s-palmitoylation": "palmitoylation",
    "palmitoylation": "palmitoylation",
    "n-myristoylation": "myristoylation",
    "myristoylation": "myristoylation",
    "s-prenylation": "prenylation",
    "prenylation": "prenylation",
    "disulfide bond": "disulfidebond",
    "disulfidebond": "disulfidebond",
    "adp-ribosylation": "adpribosylation",
    "adpribosylation": "adpribosylation",
    # PTM direction mapper already has lowercase entries for all common types
    "phosphorylation": "phosphorylation",
    "acetylation": "acetylation",
    "methylation": "methylation",
    "ubiquitination": "ubiquitination",
    "sumoylation": "sumoylation",
    "neddylation": "neddylation",
    "succinylation": "succinylation",
    "malonylation": "malonylation",
    "glutarylation": "glutarylation",
    "glycosylation": "glycosylation",
    "hydroxylation": "hydroxylation",
    "oxidation": "oxidation",
    "deamidation": "deamidation",
    "citrullination": "citrullination",
    "lactylation": "lactylation",
    "crotonylation": "crotonylation",
    "propionylation": "propionylation",
    "butyrylation": "butyrylation",
    "formylation": "formylation",
    "carbonylation": "carbonylation",
    "sulfation": "sulfation",
    "amidation": "amidation",
}

# All canonical PTM type names (lowercase, no prefix)
ALL_PTM_TYPES: Set[str] = set(PTM_TYPE_ALIASES.values())

def normalize_ptm_type(ptm_type: str) -> str:
    """Normalize a PTM type string to its canonical lowercase form.

    Handles:
    - Case differences (Phosphorylation → phosphorylation)
    - Prefix removal (N-myristoylation → myristoylation)
    - Hyphen/spacing differences (ADP-ribosylation → adpribosylation)
    - Direct lookup fallback

    Returns the canonical form, or the input lowercased if unknown.
    """
    normalized = ptm_type.strip().lower()
    # Try exact alias match first
    if normalized in PTM_TYPE_ALIASES:
        return PTM_TYPE_ALIASES[normalized]
    # Try removing common prefixes (N-, S-, O-)
    if len(normalized) > 2 and normalized[1] == '-':
        stripped = normalized[2:]
        if stripped in PTM_TYPE_ALIASES:
            return PTM_TYPE_ALIASES[stripped]
    # Try replacing hyphens with nothing
    no_hyphen = normalized.replace('-', '')
    if no_hyphen in PTM_TYPE_ALIASES:
        return PTM_TYPE_ALIASES[no_hyphen]
    # Try replacing spaces with nothing
    no_space = normalized.replace(' ', '')
    if no_space in PTM_TYPE_ALIASES:
        return PTM_TYPE_ALIASES[no_space]
    return normalized

# 非标准氨基酸字符映射表
# U (selenocysteine) → C (cysteine, 生化性质最接近)
# X (unknown) → A (alanine, 最常见氨基酸)
# J (leucine/isoleucine ambiguous) → L (leucine)
# B (asx: aspartic acid/asparagine) → D (aspartic acid)
# Z (glx: glutamic acid/glutamine) → E (glutamic acid)
# O (pyrrolysine) → K (lysine)
NON_STANDARD_AA_MAP: dict = {
    'U': 'C', 'X': 'A', 'J': 'L', 'B': 'D', 'Z': 'E', 'O': 'K',
}


# PTM 位点序列窗口默认宽度（N05, 2026-08-16）
# 训练/推理/报告三处窗口逻辑共用此常量，禁止在调用点重写魔法数 31/15/16。
# 语义: 窗口以修饰位点为中心，位点处于窗口正中 → 宽度必须为奇数。
DEFAULT_PTM_WINDOW_SIZE: int = 31
DEFAULT_PTM_HALF_WINDOW: int = 15

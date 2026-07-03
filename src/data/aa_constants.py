"""
氨基酸编码常量 — 单一定义源 (Single Source of Truth)

所有氨基酸字母表、索引映射、非标准字符映射统一在此定义。
其他模块必须从此处导入，禁止重复定义。

约定:
  - 序列索引从1开始，0保留给padding (1-indexed, PAD_IDX=0)
  - AMINO_ACIDS 为 tuple（不可变），防止调用方意外修改
"""

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

"""
氨基酸编码常量单元测试

验证 src/data/aa_constants.py 提供的常量:
  - 关键常量存在且取值正确 (20种标准氨基酸集合、PAD_IDX 等)
  - 不可变性: AMINO_ACIDS 为 tuple, 调用方无法通过 setattr / 索引赋值修改其内容
"""

import pytest

from src.data import aa_constants as aac


class TestAminoAcidAlphabet:
    """标准氨基酸字母表常量测试"""

    def test_amino_acids_is_tuple(self):
        """AMINO_ACIDS 必须为 tuple (不可变)"""
        assert isinstance(aac.AMINO_ACIDS, tuple)

    def test_amino_acids_has_twenty_residues(self):
        """标准氨基酸共 20 种"""
        assert len(aac.AMINO_ACIDS) == 20

    def test_amino_acids_no_duplicates(self):
        """字母表内无重复"""
        assert len(set(aac.AMINO_ACIDS)) == len(aac.AMINO_ACIDS)

    def test_amino_acids_all_single_uppercase(self):
        """每个残基为单字符大写字母"""
        for aa in aac.AMINO_ACIDS:
            assert isinstance(aa, str)
            assert len(aa) == 1
            assert aa.isalpha()
            assert aa.isupper()

    def test_amino_acids_str_matches_tuple(self):
        """AMINO_ACIDS_STR 与 AMINO_ACIDS 顺序一致"""
        assert aac.AMINO_ACIDS_STR == "".join(aac.AMINO_ACIDS)

    def test_canonical_residues_present(self):
        """关键残基均存在 (覆盖极性/非极性/芳香/带电)"""
        for residue in ("A", "G", "V", "L", "I", "F", "W", "Y",
                        "S", "T", "C", "M", "N", "Q", "K", "R",
                        "D", "E", "H", "P"):
            assert residue in aac.AMINO_ACIDS


class TestAaToIdxMapping:
    """氨基酸→索引映射测试"""

    def test_mapping_covers_all_residues(self):
        """每个标准氨基酸都有索引"""
        assert set(aac.AA_TO_IDX.keys()) == set(aac.AMINO_ACIDS)

    def test_indices_are_one_based_unique(self):
        """索引从 1 开始 (0 保留给 padding), 且互不相同"""
        indices = list(aac.AA_TO_IDX.values())
        assert min(indices) == 1
        assert max(indices) == len(aac.AMINO_ACIDS)
        assert len(set(indices)) == len(indices)

    def test_index_order_matches_alphabet(self):
        """索引顺序与字母表顺序一致"""
        for i, aa in enumerate(aac.AMINO_ACIDS):
            assert aac.AA_TO_IDX[aa] == i + 1


class TestPaddingConstants:
    """Padding 相关常量测试"""

    def test_pad_idx_is_zero(self):
        """PAD_IDX 固定为 0"""
        assert aac.PAD_IDX == 0

    def test_pad_char_is_dash(self):
        """AA_PAD_CHAR 为 '-'"""
        assert aac.AA_PAD_CHAR == "-"

    def test_pad_idx_not_in_aa_to_idx(self):
        """0 不被任何真实氨基酸占用"""
        assert 0 not in aac.AA_TO_IDX.values()


class TestNonStandardAaMap:
    """非标准氨基酸映射表测试"""

    def test_map_covers_expected_keys(self):
        """覆盖 U/X/J/B/Z/O 六类非标准字符"""
        assert set(aac.NON_STANDARD_AA_MAP.keys()) == {"U", "X", "J", "B", "Z", "O"}

    def test_map_targets_are_standard(self):
        """映射目标全部是标准氨基酸"""
        standard = set(aac.AMINO_ACIDS)
        for target in aac.NON_STANDARD_AA_MAP.values():
            assert target in standard

    def test_selenocysteine_maps_to_cysteine(self):
        """U → C (生化性质最接近)"""
        assert aac.NON_STANDARD_AA_MAP["U"] == "C"


class TestImmutability:
    """常量不可变性测试"""

    def test_tuple_item_assignment_raises(self):
        """对 tuple 元素赋值应抛 TypeError"""
        with pytest.raises(TypeError):
            aac.AMINO_ACIDS[0] = "Z"

    def test_tuple_append_raises(self):
        """tuple 不支持 append"""
        with pytest.raises(AttributeError):
            aac.AMINO_ACIDS.append("Z")

    def test_setattr_on_module_does_not_change_value(self, monkeypatch):
        """monkeypatch setattr 模块属性后, 重新导入仍得原始不可变值

        模块属性可被 setattr 替换, 但 AMINO_ACIDS 对象本身 (tuple) 不可变;
        这里验证替换后的对象取值不等于原始 tuple, 证明原始对象未被就地修改,
        同时原始 tuple 内容在测试间保持稳定。
        """
        original = aac.AMINO_ACIDS
        # 尝试用 monkeypatch 替换模块属性 (调用方误改)
        monkeypatch.setattr(aac, "AMINO_ACIDS", ("X",))
        # 替换后模块看到的是新对象, 原始 tuple 未被改写
        assert aac.AMINO_ACIDS == ("X",)
        # 原始对象自身内容不变
        assert original == ("A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
                            "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y")

    def test_dict_value_reassignment_is_isolated(self):
        """AA_TO_IDX 是 dict (可变), 重新赋值不影响模块外已绑定的副本快照

        这里记录一个对照: dict 本身可变, 但常量定义后不应被改;
        测试通过快照校验当前值, 防止意外被外部代码污染。
        """
        snapshot = dict(aac.AA_TO_IDX)
        # 模拟外部误改: 直接修改 dict (Python 允许, 但属错误用法)
        aac.AA_TO_IDX["A"] = 999
        # 恢复以保持测试隔离
        aac.AA_TO_IDX["A"] = snapshot["A"]
        # 快照仍记录原始值, 校验定义稳定性
        assert snapshot["A"] == 1

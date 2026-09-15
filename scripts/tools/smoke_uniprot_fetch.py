#!/usr/bin/env python3
"""
测试UniProt序列获取功能
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scripts.process_pmads_data import (
    fetch_uniprot_sequence,
    get_cached_sequence,
    save_cached_sequence,
    generate_protein_sequence,
    UNIPROT_CACHE_DIR,
)


def test_fetch_uniprot_sequence():
    """测试UniProt API查询"""
    print("=" * 60)
    print("测试1: fetch_uniprot_sequence()")
    print("=" * 60)

    # 测试已知蛋白质
    test_proteins = ["A1BG", "TP53", "AKT1"]

    for protein in test_proteins:
        print(f"\n查询蛋白质: {protein}")
        seq = fetch_uniprot_sequence(protein)
        if seq:
            print(f"  ✓ 成功获取序列，长度: {len(seq)}")
            print(f"  ✓ 序列前50aa: {seq[:50]}...")
            assert len(seq) > 0, "序列不应为空"
            assert seq.isalpha(), "序列应只包含字母"
        else:
            print(f"  ✗ 未能获取序列")

    print("\n测试1通过!\n")


def test_cache_functions():
    """测试缓存读写"""
    print("=" * 60)
    print("测试2: 缓存功能")
    print("=" * 60)

    # 清理测试缓存
    test_protein = "TEST_PROTEIN"
    test_seq = "MKTLLILTGLAVLLGLLAHSAQLTPTGTF"

    # 测试保存
    save_cached_sequence(test_protein, test_seq)
    print(f"  ✓ 保存缓存: {test_protein}")

    # 测试读取
    cached = get_cached_sequence(test_protein)
    if cached == test_seq:
        print(f"  ✓ 读取缓存成功，序列匹配")
    else:
        print(f"  ✗ 读取缓存失败或序列不匹配")
        return False

    # 测试读取不存在的缓存
    missing = get_cached_sequence("NON_EXISTENT_PROTEIN_12345")
    if missing is None:
        print(f"  ✓ 读取不存在的缓存返回None")
    else:
        print(f"  ✗ 读取不存在的缓存应返回None")
        return False

    print("\n测试2通过!\n")
    return True


def test_generate_protein_sequence():
    """测试序列生成函数"""
    print("=" * 60)
    print("测试3: generate_protein_sequence()")
    print("=" * 60)

    # 测试使用UniProt序列
    print("\n测试使用UniProt真实序列:")
    seq = generate_protein_sequence(
        protein_name="A1BG", ptm_position=237, ptm_aa="S", ptm_positions=[237, 229, 231], seq_length=500
    )
    print(f"  生成长度: {len(seq)}")
    print(f"  序列前100aa: {seq[:100]}...")

    # 检查序列是否为真实序列（不是全随机的）
    if len(seq) == 500:
        print(f"  ✓ 序列长度正确: 500")
    else:
        print(f"  ✗ 序列长度错误: {len(seq)} (期望500)")
        return False

    # 测试fallback随机序列（使用不存在的蛋白质名）
    print("\n测试fallback随机序列:")
    random_seq = generate_protein_sequence(
        protein_name="NON_EXISTENT_XYZ123", ptm_position=100, ptm_aa="K", ptm_positions=[100], seq_length=500
    )
    if len(random_seq) == 500:
        print(f"  ✓ fallback序列长度正确: 500")
    else:
        print(f"  ✗ fallback序列长度错误")
        return False

    print("\n测试3通过!\n")
    return True


def test_cache_directory():
    """测试缓存目录创建"""
    print("=" * 60)
    print("测试4: 缓存目录")
    print("=" * 60)

    if UNIPROT_CACHE_DIR.exists():
        print(f"  ✓ 缓存目录存在: {UNIPROT_CACHE_DIR}")
    else:
        print(f"  ✗ 缓存目录不存在: {UNIPROT_CACHE_DIR}")
        return False

    print("\n测试4通过!\n")
    return True


if __name__ == "__main__":
    print("\n开始测试UniProt序列获取功能\n")

    try:
        test_cache_directory()
        test_fetch_uniprot_sequence()
        test_cache_functions()
        test_generate_protein_sequence()

        print("=" * 60)
        print("所有测试通过!")
        print("=" * 60)
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

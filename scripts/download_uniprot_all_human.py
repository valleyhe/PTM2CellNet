#!/usr/bin/env python3
"""
从 UniProt 批量下载人类蛋白质序列
策略: 使用 stream 端点一次性获取所有结果, 无需分页

访问策略:
  - UniProt REST API 无明确的 rate limit header
  - stream 端点返回所有结果, 单次请求即可
  - 实测: 20K reviewed 蛋白 ~33s, 11MB
  - 保守设置: 每次请求间隔 5s, 超时 300s
"""

import argparse
import csv
import os
import time
import urllib.request
import urllib.parse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "uniprot")


def stream_query(query, fields="accession,sequence", fmt="tsv", sleep_after=5):
    """
    使用 UniProt stream 端点查询
    stream 端点返回所有匹配结果, 无需分页

    Args:
        query: UniProt 查询语句
        fields: 返回字段
        fmt: 格式 (tsv/json)
        sleep_after: 请求后等待秒数 (rate limiting)

    Returns:
        (data_string, total_count) 或 None
    """
    url_base = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "query": query,
        "format": fmt,
        "fields": fields,
    }
    url = f"{url_base}?{urllib.parse.urlencode(params)}"

    print(f"  📡 请求: {query[:80]}...")
    t0 = time.time()

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "PTM2CellNet/1.0 (data integration)")

        with urllib.request.urlopen(req, timeout=600) as resp:
            data = resp.read().decode("utf-8")
            elapsed = time.time() - t0

            # 检查 total
            total_header = resp.headers.get("X-Total-Results", "?")

            lines = data.strip().split("\n") if fmt == "tsv" else None
            count = len(lines) - 1 if lines and len(lines) > 1 else 0

            print(f"  ✅ 获取 {count:,} 条序列 (总计 {total_header}) 耗时 {elapsed:.1f}s")

            if sleep_after > 0:
                time.sleep(sleep_after)

            return data, count
    except urllib.error.HTTPError as e:
        print(f"  ❌ HTTP {e.code}: {e.reason}")
        if e.code == 429:
            retry_after = e.headers.get("Retry-After", "60")
            print(f"  ⏳ 被限流, 等待 {retry_after}s")
            time.sleep(int(retry_after))
        return None, 0
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return None, 0


def parse_tsv_sequences(tsv_data):
    """解析 TSV 数据为 {accession: sequence} 字典"""
    sequences = {}
    if not tsv_data or not tsv_data.strip():
        return sequences

    lines = tsv_data.strip().split("\n")
    if len(lines) < 2:
        return sequences

    reader = csv.reader(lines, delimiter="\t")
    header = next(reader, None)

    if not header:
        return sequences

    acc_idx = header.index("Entry") if "Entry" in header else 0
    seq_idx = header.index("Sequence") if "Sequence" in header else 1

    for row in reader:
        if len(row) > max(acc_idx, seq_idx):
            acc = row[acc_idx].strip()
            seq = row[seq_idx].strip()
            if seq and len(seq) > 10:
                sequences[acc] = seq

    return sequences


def main():
    parser = argparse.ArgumentParser(description="UniProt 人类蛋白质序列批量下载")
    parser.add_argument(
        "--output", type=str, default=None, help="输出文件路径 (默认: data/raw/uniprot/human_all_sequences.fasta)"
    )
    parser.add_argument("--no-reviewed", action="store_true", help="跳过 reviewed (Swiss-Prot) 蛋白")
    parser.add_argument("--no-unreviewed", action="store_true", help="跳过 unreviewed (TrEMBL) 蛋白")
    parser.add_argument("--dry-run", action="store_true", help="仅显示计划, 不实际下载")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_fasta = args.output or os.path.join(OUTPUT_DIR, "human_all_sequences.fasta")

    print("=" * 60)
    print("  UniProt 人类蛋白质序列批量下载")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_sequences = {}

    # ============================================================
    # Step 1: 下载所有 reviewed (Swiss-Prot) 人类蛋白
    # ============================================================
    if not args.no_reviewed:
        print("\n[1/2] 下载 reviewed (Swiss-Prot) 人类蛋白质...")
        print("  查询: reviewed:true AND organism_id:9606")

        if args.dry_run:
            print("  (dry-run: 跳过)")
        else:
            result, count = stream_query(
                "reviewed:true AND organism_id:9606",
                fields="accession,sequence",
                fmt="tsv",
                sleep_after=5,
            )
            if result:
                seqs = parse_tsv_sequences(result)
                all_sequences.update(seqs)
                print(f"  ✅ reviewed: {len(seqs):,} 条序列")

                # 保存为 FASTA
                reviewed_fasta = os.path.join(OUTPUT_DIR, "human_reviewed.fasta")
                with open(reviewed_fasta, "w") as f:
                    for acc, seq in seqs.items():
                        f.write(f">{acc}\n{seq}\n")
                print(f"  💾 保存: {reviewed_fasta} ({os.path.getsize(reviewed_fasta) / 1024 / 1024:.1f}MB)")

    # ============================================================
    # Step 2: 下载所有 unreviewed (TrEMBL) 人类蛋白
    # ============================================================
    if not args.no_unreviewed:
        print("\n[2/2] 下载 unreviewed (TrEMBL) 人类蛋白质...")
        print("  查询: reviewed:false AND organism_id:9606")

        if args.dry_run:
            print("  (dry-run: 跳过)")
        else:
            result, count = stream_query(
                "reviewed:false AND organism_id:9606",
                fields="accession,sequence",
                fmt="tsv",
                sleep_after=5,
            )
            if result:
                seqs = parse_tsv_sequences(result)
                all_sequences.update(seqs)
                print(f"  ✅ unreviewed: {len(seqs):,} 条序列")

    # ============================================================
    # 汇总 & 保存
    # ============================================================
    print(f"\n{'=' * 60}")
    print(f"  下载汇总")
    print(f"{'=' * 60}")
    print(f"  总序列数: {len(all_sequences):,}")

    if all_sequences and not args.dry_run:
        # 保存为 FASTA
        with open(output_fasta, "w") as f:
            for acc, seq in all_sequences.items():
                f.write(f">{acc}\n{seq}\n")

        size_mb = os.path.getsize(output_fasta) / 1024 / 1024
        print(f"  文件大小: {size_mb:.1f} MB")
        print(f"  保存路径: {output_fasta}")

        # 统计长度分布
        lengths = [len(s) for s in all_sequences.values()]
        print(f"  平均长度: {sum(lengths) // len(lengths)} aa")
        print(f"  最短: {min(lengths)} aa")
        print(f"  最长: {max(lengths)} aa")

    return all_sequences


if __name__ == "__main__":
    main()

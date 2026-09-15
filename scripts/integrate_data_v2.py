"""
PTM2CellNet 数据整合脚本 v2
- 仅整合人类(Homo sapiens)PTM数据
- 从 UniProt API 批量获取缺失的人类蛋白序列
- 整合已标记的 cell_state 数据

数据源:
  1. EPSD 2.0 - 人类磷酸化位点
  2. CPLM 4.0 - 人类赖氨酸修饰
  3. dbPTM 2025 - 人类PTM (过滤 _HUMAN 条目)
  4. UniProt API - 批量获取缺失序列
  5. 已有本地数据 - 带cell_state标签
"""

import argparse
import csv
import json
import os
import time
import urllib.request
import urllib.parse
from collections import defaultdict

import pandas as pd

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

PTM_TYPE_MAP = {
    "Phosphorylation": "phosphorylation",
    "phosphorylation": "phosphorylation",
    "Acetylation": "acetylation",
    "acetylation": "acetylation",
    "Methylation": "methylation",
    "methylation": "methylation",
    "Ubiquitination": "ubiquitination",
    "ubiquitination": "ubiquitination",
    "Sumoylation": "sumoylation",
    "sumoylation": "sumoylation",
    "Crotonylation": "crotonylation",
    "2-Hydroxyisobutyrylation": "2-hydroxyisobutyrylation",
    "β-Hydroxybutyrylation": "beta-hydroxybutyrylation",
    "Butyrylation": "butyrylation",
    "Malonylation": "malonylation",
    "Succinylation": "succinylation",
    "Glutarylation": "glutarylation",
    "Glycation": "glycation",
    "Lactylation": "lactylation",
    "Hydroxylation": "hydroxylation",
    "Phosphoglycerylation": "phosphoglycerylation",
    "Pupylation": "pupylation",
    "Neddylation": "neddylation",
    "Propionylation": "propionylation",
    "Benzoylation": "benzoylation",
    "Lipoylation": "lipoylation",
    "Biotinylation": "biotinylation",
    "Carboxymethylation": "carboxymethylation",
    "Carboxylation": "carboxylation",
    "Carboxyethylation": "carboxyethylation",
    "dietylphosphorylation": "dietylphosphorylation",
}


# ============================================================
# 1. 加载序列
# ============================================================


def _load_gene_symbol_map():
    """从本地 UniProt idmapping 构建 accession → gene_symbol 映射 (P2-1)。

    读取 data/raw/uniprot/human_idmapping.gz（UniProt 官方三列格式：
    accession<TAB>database<TAB>id），仅取 Gene_Name 行。文件缺失或解析
    失败时返回空映射并打印提示——纯离线路径，不引入网络依赖。

    返回:
        Dict[str, str]: UniProt accession → 标准基因名。
    """
    import gzip

    idmapping_path = os.path.join(RAW_DIR, "uniprot", "human_idmapping.gz")
    if not os.path.exists(idmapping_path):
        print(
            "  ⚠️ 未找到本地 idmapping (data/raw/uniprot/human_idmapping.gz)，"
            "gene_symbol 列将留空（不影响训练，但影响泄漏审计粒度）"
        )
        return {}

    gene_map = {}
    try:
        with gzip.open(idmapping_path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3 and parts[1] == "Gene_Name":
                    gene_map[parts[0]] = parts[2]
    except OSError as exc:
        print(f"  ⚠️ 解析 idmapping 失败: {exc}，gene_symbol 列将留空")
        return {}
    print(f"  ✅ 基因名映射加载: {len(gene_map):,} 个 accession")
    return gene_map


def load_local_sequences():
    """从本地文件加载序列 (UniProt 完整版 + CPLM)"""
    sequences = {}

    # UniProt 全量人类序列 (reviewed + unreviewed, 204K+)
    fasta_path = os.path.join(RAW_DIR, "uniprot", "human_all_sequences.fasta")
    if os.path.exists(fasta_path):
        with open(fasta_path, "r") as f:
            current_id = None
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id and current_seq:
                        sequences[current_id] = "".join(current_seq)
                    parts = line.split("|")
                    current_id = parts[1] if len(parts) >= 2 else line.split()[0][1:]
                    current_seq = []
                else:
                    if current_id:
                        current_seq.append(line)
            if current_id and current_seq:
                sequences[current_id] = "".join(current_seq)
        print(f"  ✅ 从 human_all_sequences.fasta 加载 {len(sequences):,} 条序列")
    else:
        # Fallback to old reference proteome
        fasta_path = os.path.join(RAW_DIR, "uniprot", "human_proteome.fasta")
        if os.path.exists(fasta_path):
            with open(fasta_path, "r") as f:
                current_id = None
                current_seq = []
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if current_id and current_seq:
                            sequences[current_id] = "".join(current_seq)
                        parts = line.split("|")
                        current_id = parts[1] if len(parts) >= 2 else line.split()[0][1:]
                        current_seq = []
                    else:
                        if current_id:
                            current_seq.append(line)
                if current_id and current_seq:
                    sequences[current_id] = "".join(current_seq)
            print(f"  ✅ 从 human_proteome.fasta 加载 {len(sequences):,} 条序列 (旧版)")

    # CPLM (自带完整序列)
    cplm_path = os.path.join(RAW_DIR, "cplm", "Homo sapiens.txt")
    if os.path.exists(cplm_path):
        cplm_count = 0
        with open(cplm_path, "r", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) >= 7:
                    uid = row[1].strip()
                    seq = row[6].strip()
                    if uid and seq and len(seq) > 50 and uid not in sequences:
                        sequences[uid] = seq
                        cplm_count += 1
        print(f"  ✅ 从 CPLM 额外加载 {cplm_count:,} 条序列")

    print(f"  ✅ 总计 {len(sequences):,} 条唯一序列")
    return sequences


def fetch_uniprot_batch(uniprot_ids, batch_size=500):
    """从 UniProt REST API 批量获取序列

    使用 UniProt 的 ID mapping 端点进行批量获取
    """
    sequences = {}
    total = len(uniprot_ids)

    print(f"  📡 从 UniProt API 获取 {total:,} 条序列...")

    # 使用 UniProt 的 ID list 下载方式
    url_base = "https://rest.uniprot.org/uniprotkb/stream"

    for i in range(0, total, batch_size):
        batch = uniprot_ids[i : i + batch_size]
        # 构建查询: accession:(ID1 OR ID2 OR ...)
        query_parts = " OR ".join(f"accession:{uid}" for uid in batch)

        params = {
            "query": query_parts,
            "format": "tsv",
            "fields": "accession,sequence",
            "size": len(batch),
        }

        try:
            url = f"{url_base}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "PTM2CellNet/1.0")
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read().decode("utf-8").strip()
                if data:
                    lines = data.split("\n")
                    if len(lines) > 1:
                        reader = csv.reader(lines, delimiter="\t")
                        header = next(reader, None)
                        if header:
                            acc_idx = header.index("Entry") if "Entry" in header else 0
                            seq_idx = header.index("Sequence") if "Sequence" in header else 1
                            for row in reader:
                                if len(row) > max(acc_idx, seq_idx):
                                    acc = row[acc_idx].strip()
                                    seq = row[seq_idx].strip()
                                    if seq and len(seq) > 10:
                                        sequences[acc] = seq
        except Exception as e:
            # 如果批量失败，尝试单个获取
            if len(batch) > 10:
                # 缩小批次重试
                for uid in batch[:5]:
                    try:
                        uid_url = f"https://rest.uniprot.org/uniprotkb/{uid}.tsv?fields=accession,sequence"
                        req = urllib.request.Request(uid_url)
                        req.add_header("User-Agent", "PTM2CellNet/1.0")
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            data = resp.read().decode("utf-8").strip()
                            lines = data.split("\n")
                            if len(lines) >= 2:
                                reader = csv.reader(lines, delimiter="\t")
                                header = next(reader, None)
                                row = next(reader, None)
                                if header and row:
                                    acc_idx = header.index("Entry") if "Entry" in header else 0
                                    seq_idx = header.index("Sequence") if "Sequence" in header else 1
                                    if len(row) > max(acc_idx, seq_idx):
                                        acc = row[acc_idx].strip()
                                        seq = row[seq_idx].strip()
                                        if seq and len(seq) > 10:
                                            sequences[acc] = seq
                    except Exception:
                        pass

        fetched = min(i + batch_size, total)
        if (i // batch_size + 1) % 20 == 0 or fetched >= total:
            pct = fetched / total * 100
            print(f"    进度: {fetched:,}/{total:,} ({pct:.0f}%), 获取 {len(sequences):,} 条")
        time.sleep(0.5)

    print(f"  ✅ API 获取 {len(sequences):,}/{total:,} 条序列")
    return sequences


# ============================================================
# 2. 加载 PTM 位点 (仅人类)
# ============================================================
def load_epsd_sites():
    """EPSD: 已下载 Homo sapiens，直接使用"""
    epsd_path = os.path.join(RAW_DIR, "epsd", "Homo sapiens.txt")
    if not os.path.exists(epsd_path):
        return defaultdict(list)

    sites = defaultdict(list)
    count = 0
    with open(epsd_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)  # 跳过表头
        for row in reader:
            if len(row) >= 4:
                uid = row[1].strip()
                aa = row[2].strip()
                try:
                    pos = int(row[3].strip())
                except ValueError:
                    continue
                if uid and pos > 0:
                    sites[uid].append({"position": pos, "type": "phosphorylation", "amino_acid": aa})
                    count += 1

    print(f"  ✅ EPSD: {count:,} 磷酸化位点 ({len(sites):,} 人类蛋白)")
    return sites


def load_cplm_sites():
    """CPLM: 已下载 Homo sapiens，直接使用"""
    cplm_path = os.path.join(RAW_DIR, "cplm", "Homo sapiens.txt")
    if not os.path.exists(cplm_path):
        return defaultdict(list)

    sites = defaultdict(list)
    count = 0
    with open(cplm_path, "r", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 4:
                uid = row[1].strip()
                pos = int(row[2].strip())
                ptm_type = row[3].strip()
                if uid and pos > 0:
                    sites[uid].append(
                        {
                            "position": pos,
                            "type": PTM_TYPE_MAP.get(ptm_type, ptm_type.lower()),
                            "amino_acid": None,
                        }
                    )
                    count += 1

    print(f"  ✅ CPLM: {count:,} 赖氨酸修饰位点 ({len(sites):,} 人类蛋白)")
    return sites


def load_dbptm_human_sites(ptm_type_name):
    """dbPTM: 仅加载 _HUMAN 条目"""
    ptm_file = os.path.join(RAW_DIR, "dbptm", ptm_type_name)
    if not os.path.exists(ptm_file):
        return defaultdict(list)

    sites = defaultdict(list)
    count = 0
    skipped_non_human = 0
    skipped_parse = 0

    with open(ptm_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                skipped_parse += 1
                continue

            # 检查是否为人类蛋白
            entry = parts[0].strip()
            if not entry.endswith("_HUMAN"):
                skipped_non_human += 1
                continue

            # 解析 (6列格式)
            try:
                pos = int(parts[2])
                uid = parts[1].strip()
                raw_ptm_type = parts[3].strip()
            except (ValueError, IndexError):
                # 尝试5列格式
                try:
                    pos = int(parts[1])
                    uid = parts[0].strip()
                    raw_ptm_type = parts[2].strip()
                except (ValueError, IndexError):
                    skipped_parse += 1
                    continue

            if uid and pos > 0:
                sites[uid].append(
                    {
                        "position": pos,
                        "type": PTM_TYPE_MAP.get(raw_ptm_type, raw_ptm_type.lower()),
                        "amino_acid": None,
                    }
                )
                count += 1

    print(f"  ✅ dbPTM {ptm_type_name}: {count:,} 位点 ({len(sites):,} 人类蛋白, 跳过{skipped_non_human:,}非人类)")
    return sites


# ============================================================
# 3. 合并 & 构建数据集
# ============================================================
def merge_sites(all_sites_dicts):
    """合并并去重 PTM 位点"""
    merged = defaultdict(list)
    total, dedup = 0, 0

    for sites_dict in all_sites_dicts:
        for uid, sites in sites_dict.items():
            for site in sites:
                pos, ptm_type = site["position"], site["type"]
                found = any(s["position"] == pos and s["type"] == ptm_type for s in merged[uid])
                if found:
                    dedup += 1
                else:
                    merged[uid].append(site)
                    total += 1

    print(f"\n  合并: {total:,} 唯一位点 (去重 {dedup:,})")
    return merged


def build_dataset(sequences, ptm_sites, max_seq_len=1000, gene_symbol_map=None):
    """构建 (sequence, ptm_sites) 数据集, 支持异构体ID解析

    P2-1: 输出推荐 provenance 列 (protein_accession / gene_symbol /
    source_db / evidence_level)，供数据契约与泄漏审计使用。
    """
    records = []
    no_seq = 0
    too_short = 0
    isoform_resolved = 0
    gene_map = gene_symbol_map or {}

    for uid, sites in ptm_sites.items():
        seq = sequences.get(uid)

        # 如果直接找不到, 尝试解析异构体ID (e.g., Q9Y483-3 → Q9Y483)
        if not seq and "-" in uid:
            canonical_id = uid.split("-")[0]
            seq = sequences.get(canonical_id)
            if seq:
                isoform_resolved += 1
                uid = canonical_id  # 使用canonical ID

        if not seq:
            no_seq += 1
            continue

        seq_len = len(seq)
        if seq_len < 20:
            too_short += 1
            continue

        if seq_len > max_seq_len:
            seq = seq[:max_seq_len]
            seq_len = max_seq_len
            sites = [s for s in sites if s["position"] <= max_seq_len]

        if not sites:
            continue

        sites = sorted(sites, key=lambda x: x["position"])
        ptm_json = json.dumps(sites, ensure_ascii=False)

        records.append(
            {
                "id": f"PTM_{uid}",
                "sequence": seq,
                "ptm_sites": ptm_json,
                "uniprot_id": uid,
                "protein_accession": uid,
                "gene_symbol": gene_map.get(uid, ""),
                "source_db": "epsd+cplm+dbptm",
                "evidence_level": "database",
                "seq_length": seq_len,
                "num_ptm_sites": len(sites),
                "cell_state": "unknown",
            }
        )

    print(f"\n  数据集: {len(records):,} 条记录")
    print(f"    跳过(无序列): {no_seq:,}")
    print(f"    跳过(太短): {too_short:,}")
    print(f"    异构体→Canonical: {isoform_resolved:,}")
    return pd.DataFrame(records)


# ============================================================
# 4. 整合已标记数据
# ============================================================
def integrate_labels(integrated_df, gene_symbol_map=None):
    """
    尝试将现有带标签数据整合到整合数据集中

    PMADS 数据已有 cell_state 标签
    """
    pmads_path = os.path.join(PROCESSED_DIR, "pmads_combined.csv")
    if not os.path.exists(pmads_path):
        return integrated_df

    print(f"\n  整合 PMADS 标签数据...")
    pmads = pd.read_csv(pmads_path)
    print(f"  PMADS 记录: {len(pmads):,}")

    # PMADS 有完整 sequence + cell_state + ptm_sites
    # 直接保留为独立的已标记数据集
    pmads_clean = pmads[["id", "sequence", "ptm_sites", "cell_state"]].copy()
    pmads_clean["uniprot_id"] = pmads.get("protein", "unknown")
    # P2-1: PMADS 的 protein 列即 UniProt accession；source 列即来源数据库。
    pmads_clean["protein_accession"] = pmads.get("protein", "unknown").astype(str)
    gene_map = gene_symbol_map or {}
    pmads_clean["gene_symbol"] = pmads_clean["protein_accession"].map(lambda acc: gene_map.get(acc, ""))
    pmads_clean["source_db"] = pmads.get("source", "pmads")
    pmads_clean["evidence_level"] = "database"
    pmads_clean["seq_length"] = pmads_clean["sequence"].str.len()
    pmads_clean["num_ptm_sites"] = 1  # PMADS 每行一个PTM位点

    print(f"  PMADS cell_state 分布:")
    cs_dist = pmads_clean["cell_state"].value_counts()
    for cs, cnt in cs_dist.items():
        print(f"    {str(cs)[:40]}: {cnt:,}")

    return pmads_clean


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="PTM2CellNet 数据整合 v2 (人类专用)")
    parser.add_argument("--max-seq-len", type=int, default=1000)
    parser.add_argument("--output", type=str, default="data/processed/ptm_integrated_human.csv")
    parser.add_argument("--fetch-api", action="store_true", help="从 UniProt API 获取缺失序列")
    parser.add_argument("--api-batch", type=int, default=500, help="API 批次大小")
    args = parser.parse_args()

    output_path = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=" * 60)
    print("  PTM2CellNet 数据整合 v2 (人类专用)")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 加载本地序列
    print("\n[1/7] 加载本地蛋白质序列...")
    sequences = load_local_sequences()

    # 2. 收集所有人类 UniProt ID
    print("\n[2/7] 收集人类 PTM UniProt ID...")
    all_human_ids = set()

    # EPSD
    epsd_ids = set()
    epsd_path = os.path.join(RAW_DIR, "epsd", "Homo sapiens.txt")
    if os.path.exists(epsd_path):
        with open(epsd_path, "r") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) >= 2:
                    epsd_ids.add(row[1].strip())
    all_human_ids.update(epsd_ids)

    # dbPTM
    for ptm_file in ["Phosphorylation", "Acetylation", "Methylation", "Ubiquitination", "Sumoylation"]:
        pfile = os.path.join(RAW_DIR, "dbptm", ptm_file)
        if os.path.exists(pfile):
            with open(pfile, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 5 and parts[0].endswith("_HUMAN"):
                        try:
                            int(parts[2])
                            all_human_ids.add(parts[1].strip())
                        except ValueError:
                            pass

    # CPLM
    cplm_ids = set()
    cplm_path = os.path.join(RAW_DIR, "cplm", "Homo sapiens.txt")
    if os.path.exists(cplm_path):
        with open(cplm_path, "r") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) >= 2:
                    cplm_ids.add(row[1].strip())
    all_human_ids.update(cplm_ids)

    # CPLM already has sequences
    cplm_with_seq = all_human_ids & set(sequences.keys())
    need_fetch = all_human_ids - set(sequences.keys())
    print(f"  总人类 UniProt ID: {len(all_human_ids):,}")
    print(f"  已有序列: {len(cplm_with_seq):,}")
    print(f"  需获取: {len(need_fetch):,}")

    # 3. 从 UniProt API 获取缺失序列
    if args.fetch_api and need_fetch:
        print(f"\n[3/7] 从 UniProt API 获取 {len(need_fetch):,} 条序列...")
        api_seqs = fetch_uniprot_batch(sorted(need_fetch), args.api_batch)
        sequences.update(api_seqs)
    else:
        print(f"\n[3/7] 跳过 API 获取 (使用 --fetch-api 启用)")

    print(f"  ✅ 最终可用序列: {len(sequences):,}")

    # 4. 加载 PTM 位点
    print(f"\n[4/7] 加载 PTM 位点...")
    epsd_sites = load_epsd_sites()
    cplm_sites = load_cplm_sites()
    dbptm_sites_list = []
    for ptm_type in ["Phosphorylation", "Acetylation", "Methylation", "Ubiquitination", "Sumoylation"]:
        dbptm_sites_list.append(load_dbptm_human_sites(ptm_type))

    # 5. 合并位点
    print(f"\n[5/7] 合并 PTM 位点...")
    merged_sites = merge_sites([epsd_sites, cplm_sites] + dbptm_sites_list)

    # 6. 构建数据集
    print(f"\n[6/7] 构建训练数据集...")
    # P2-1: 加载本地基因名映射（离线），产出 provenance 列。
    gene_symbol_map = _load_gene_symbol_map()
    df = build_dataset(sequences, merged_sites, args.max_seq_len, gene_symbol_map)

    # 7. 保存
    print(f"\n[7/7] 保存数据...")
    df.to_csv(output_path, index=False, encoding="utf-8")
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  ✅ 已保存: {output_path} ({size_mb:.1f} MB)")

    # 统计报告
    print(f"\n{'=' * 60}")
    print(f"  数据质量报告")
    print(f"{'=' * 60}")
    print(f"  总记录数: {len(df):,}")
    if len(df) > 0:
        print(f"  平均序列长度: {df['seq_length'].mean():.0f}")
        print(f"  中位序列长度: {df['seq_length'].median():.0f}")
        print(f"  平均PTM位点数: {df['num_ptm_sites'].mean():.1f}")
        print(f"  中位PTM位点数: {df['num_ptm_sites'].median():.0f}")

        all_types = []
        for ptm_json_str in df["ptm_sites"]:
            try:
                for s in json.loads(ptm_json_str):
                    all_types.append(s["type"])
            except (json.JSONDecodeError, TypeError):
                pass
        if all_types:
            type_counts = pd.Series(all_types).value_counts()
            print(f"\n  PTM 类型分布:")
            for ptm_type, cnt in type_counts.items():
                print(f"    {ptm_type}: {cnt:,}")

    # 整合标签数据
    print(f"\n{'=' * 60}")
    print(f"  整合已标记数据...")
    labeled_df = integrate_labels(df, gene_symbol_map)
    if labeled_df is not None and len(labeled_df) > 0:
        labeled_path = output_path.replace(".csv", "_labeled.csv")
        labeled_df.to_csv(labeled_path, index=False, encoding="utf-8")
        print(f"  ✅ 已标记数据保存: {labeled_path}")

    print(f"\n{'=' * 60}")
    print(f"  ✅ 数据整合完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

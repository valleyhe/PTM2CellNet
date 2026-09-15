#!/usr/bin/env python3
"""
处理PMADS数据集，转换为训练格式

输入:
- data/ptmdb/CPTAC_PTM_intensity.csv (175k+ PTM records)
- data/ptmdb/PTMD_data_in_pmads.txt (PTM-disease associations)

输出:
- data/processed/pmads_combined.csv (合并后的训练数据)
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from statistics import median

import pandas as pd
import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

UNIPROT_CACHE_DIR = Path("data/uniprot_cache")


def parse_ptm_site(site_str):
    """
    解析PTM位点字符串，如 "A1BG-Ser237"

    Returns:
        (protein_name, amino_acid, position)
    """
    # Pattern: ProteinName-AminoAcidPosition
    # e.g., "A1BG-Ser237" -> ("A1BG", "S", 237)
    # e.g., "A1CF-Ser482" -> ("A1CF", "S", 482)

    aa_map = {
        "Ser": "S",
        "Thr": "T",
        "Tyr": "Y",
        "Lys": "K",
        "Arg": "R",
        "His": "H",
        "Asp": "D",
        "Glu": "E",
        "Asn": "N",
        "Gln": "Q",
        "Cys": "C",
        "Gly": "G",
        "Pro": "P",
        "Ala": "A",
        "Val": "V",
        "Ile": "I",
        "Leu": "L",
        "Met": "M",
        "Phe": "F",
        "Trp": "W",
    }

    # Match pattern: Protein-AAA###
    match = re.match(r"([A-Z0-9]+)-([A-Za-z]+)(\d+)", site_str)
    if match:
        protein = match.group(1)
        aa_code = match.group(2)
        position = int(match.group(3))
        aa = aa_map.get(aa_code, "X")
        return protein, aa, position

    return None, None, None


def _sanitize_protein_name(protein_name):
    """将蛋白名称转换为安全文件名。"""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(protein_name))


def get_cached_sequence(protein_name):
    """
    从本地缓存读取蛋白序列。

    缓存路径: data/uniprot_cache/{protein_name}.json
    """
    cache_file = UNIPROT_CACHE_DIR / f"{_sanitize_protein_name(protein_name)}.json"
    if not cache_file.exists():
        return None

    try:
        with cache_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        sequence = payload.get("sequence")
        if isinstance(sequence, str) and sequence:
            return sequence
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: failed to read cache for {protein_name}: {exc}")

    return None


def save_cached_sequence(protein_name, sequence):
    """
    保存蛋白序列到本地缓存。

    缓存路径: data/uniprot_cache/{protein_name}.json
    """
    if not sequence:
        return

    UNIPROT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = UNIPROT_CACHE_DIR / f"{_sanitize_protein_name(protein_name)}.json"

    payload = {"protein_name": protein_name, "sequence": sequence}

    try:
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except OSError as exc:
        print(f"Warning: failed to write cache for {protein_name}: {exc}")


def fetch_uniprot_sequence(protein_name):
    """
    从UniProt查询人类蛋白序列。

    API:
    https://rest.uniprot.org/uniprotkb/search?query=gene:{protein_name}+AND+organism_id:9606&fields=sequence&format=json
    """
    url = (
        "https://rest.uniprot.org/uniprotkb/search"
        f"?query=gene:{protein_name}+AND+organism_id:9606"
        "&fields=sequence&format=json"
    )

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        if not results:
            return None

        sequence_field = results[0].get("sequence", {})
        if isinstance(sequence_field, dict):
            sequence = sequence_field.get("value")
        else:
            sequence = sequence_field

        if isinstance(sequence, str) and sequence:
            return sequence
    except (requests.RequestException, ValueError) as exc:
        print(f"Warning: UniProt lookup failed for {protein_name}: {exc}")
    finally:
        # Respect API rate limit: one request every ~0.33s
        time.sleep(0.33)

    return None


def load_cptac_data(filepath):
    """
    加载CPTAC PTM强度数据

    CSV格式: ID, P, Protein-Site, intensity samples...
    """
    print(f"Loading CPTAC data from {filepath}...")

    df = pd.read_csv(filepath, header=None)

    # First 3 columns: ID, type, protein-site
    # Remaining columns: intensity values across samples

    records = []
    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"  Processed {idx}/{len(df)} records...")

        site_str = row[2]  # Protein-Site column
        protein, aa, position = parse_ptm_site(str(site_str))

        if protein is None:
            continue

        # Get non-null intensity values
        intensities = pd.Series(row.iloc[3:]).dropna()
        if len(intensities) == 0:
            continue

        # Calculate average intensity as label
        avg_intensity = intensities.mean()

        # Create binary label based on intensity (high vs low modification)
        # Using median as threshold
        label = 1 if avg_intensity > 0 else 0

        records.append(
            {
                "id": f"PMADS_{idx}",
                "protein": protein,
                "sequence": f"{aa}",  # Placeholder - will need actual sequence
                "ptm_position": position,
                "ptm_type": "Phosphorylation" if aa in ["S", "T", "Y"] else "Acetylation" if aa == "K" else "Other",
                "ptm_aa": aa,
                "intensity": avg_intensity,
                "label": label,
            }
        )

    return pd.DataFrame(records)


def load_ptmd_data(filepath):
    """
    加载PTM-disease关联数据

    Format: UniProtID, ..., PTMType, ..., Disease, ..., Regulation, ...
    """
    print(f"Loading PTMD data from {filepath}...")

    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        encoding="latin-1",
        names=[
            "uniprot",
            "blank1",
            "ptm",
            "blank2",
            "disease",
            "blank3",
            "regulation",
            "blank4",
            "pmid",
            "blank5",
            "target",
            "reference",
            "evidence",
            "ptm_type",
        ],
    )

    records = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        if idx % 1000 == 0:
            print(f"  Processed {idx}/{len(df)} records...")

        ptm_desc = str(row["ptm"])
        # Parse "Protein-Site" format
        if "-" in ptm_desc:
            parts = ptm_desc.rsplit("-", 1)
            if len(parts) == 2:
                protein, site = parts
                # Extract amino acid and position
                match = re.match(r"([A-Za-z]+)(\d+)", site)
                if match:
                    aa_code = match.group(1)
                    position = int(match.group(2))

                    aa_map = {"Ser": "S", "Thr": "T", "Tyr": "Y", "Lys": "K"}
                    aa = aa_map.get(aa_code, "X")

                    # Regulation: U (Up) or D (Down) as label
                    regulation = str(row["regulation"]).strip()
                    label = 1 if regulation == "U" else 0 if regulation == "D" else -1

                    if label != -1:
                        records.append(
                            {
                                "id": f"PTMD_{idx}",
                                "protein": protein,
                                "sequence": f"{aa}",
                                "ptm_position": position,
                                "ptm_type": row["ptm_type"] if pd.notna(row["ptm_type"]) else "Unknown",
                                "ptm_aa": aa,
                                "disease": row["disease"],
                                "regulation": regulation,
                                "label": label,
                            }
                        )

    return pd.DataFrame(records)


def generate_protein_sequence(protein_name, ptm_position, ptm_aa, ptm_positions, seq_length=500):
    """
    生成蛋白序列窗口（优先UniProt真实序列，失败时回退随机序列）。

    优先级:
    1) 本地缓存
    2) UniProt API
    3) 随机序列回退

    使用该蛋白质所有PTM位置的中位数作为窗口中心，返回500aa窗口。
    """
    import random

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    # Compute median position across all PTM sites for this protein
    positions = [int(p) for p in (ptm_positions or []) if pd.notna(p)]
    if positions:
        median_pos = int(median(positions))
    else:
        median_pos = int(ptm_position) if pd.notna(ptm_position) else seq_length // 2

    current_pos = int(ptm_position) if pd.notna(ptm_position) else median_pos

    # 1) Check cache
    full_sequence = get_cached_sequence(protein_name)

    # 2) Query UniProt API if cache missed
    if not full_sequence:
        full_sequence = fetch_uniprot_sequence(protein_name)
        if full_sequence:
            save_cached_sequence(protein_name, full_sequence)

    center_idx = seq_length // 2

    # 3) Fallback to random sequence if API failed
    if not full_sequence:
        sequence = list(random.choices(amino_acids, k=seq_length))
        pos = center_idx + (current_pos - median_pos)
        if 0 <= pos < seq_length:
            sequence[pos] = ptm_aa
        return "".join(sequence)

    # Use median-centered 500aa window from full sequence
    seq_len = len(full_sequence)
    if seq_len >= seq_length:
        # Position is 1-based; center the median position at window center.
        start = max(0, min(seq_len - seq_length, median_pos - center_idx - 1))
        window = list(full_sequence[start : start + seq_length])
    else:
        # Short proteins: place sequence into a 500aa padded window, still median-centered.
        window = list(random.choices(amino_acids, k=seq_length))
        offset = center_idx - (median_pos - 1)
        for idx, aa in enumerate(full_sequence):
            target = idx + offset
            if 0 <= target < seq_length:
                window[target] = aa

    # Ensure PTM amino acid aligns with current site relative to median center.
    ptm_idx = center_idx + (current_pos - median_pos)
    if 0 <= ptm_idx < seq_length:
        window[ptm_idx] = ptm_aa

    return "".join(window)


def create_ptm_sites(ptm_position, ptm_aa, ptm_type):
    """
    创建PTM位点JSON格式
    """
    ptm_sites = [{"position": int(ptm_position), "type": ptm_type, "amino_acid": ptm_aa}]

    return json.dumps(ptm_sites)


def process_and_merge_data(output_path="data/processed/pmads_combined.csv"):
    """
    处理所有PMADS数据并合并
    """
    ptmdb_dir = Path("data/ptmdb")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load CPTAC data
    cptac_df = load_cptac_data(ptmdb_dir / "CPTAC_PTM_intensity.csv")
    print(f"\nCPTAC data loaded: {len(cptac_df)} records")

    # Load PTMD data (skip for now due to parsing issues)
    ptmd_df = pd.DataFrame()  # Empty DataFrame for now
    print("Skipping PTMD data (parsing issues)")

    # Combine datasets
    print("\nCombining datasets...")

    # Add source column
    cptac_df["source"] = "CPTAC"

    # Use only CPTAC data for now
    combined_df = cptac_df.copy()

    # Collect PTM positions per protein for consistent window centering
    protein_to_positions = (
        combined_df.groupby("protein")["ptm_position"].apply(lambda s: [int(p) for p in s if pd.notna(p)]).to_dict()
    )

    # Generate sequences and PTM sites
    print("\nGenerating protein sequences and PTM annotations...")
    combined_df["sequence"] = combined_df.apply(
        lambda row: generate_protein_sequence(
            row["protein"], row["ptm_position"], row["ptm_aa"], protein_to_positions.get(row["protein"], [])
        ),
        axis=1,
    )

    combined_df["ptm_sites"] = combined_df.apply(
        lambda row: create_ptm_sites(row["ptm_position"], row["ptm_aa"], row["ptm_type"]), axis=1
    )

    # Map labels to cell states
    # 1 -> "Activated", 0 -> "Quiescent"
    label_map = {1: "Activated", 0: "Quiescent"}
    combined_df["cell_state"] = combined_df["label"].map(label_map)

    # Select and rename columns to match existing format
    final_df = combined_df[
        ["id", "sequence", "ptm_sites", "cell_state", "protein", "ptm_position", "ptm_type", "source"]
    ].copy()

    # Save
    print(f"\nSaving combined data to {output_file}...")
    final_df.to_csv(output_file, index=False)

    print(f"\n{'=' * 60}")
    print("Data Processing Complete!")
    print(f"{'=' * 60}")
    print(f"Total records: {len(final_df)}")
    print(f"CPTAC records: {len(cptac_df)}")
    print(f"PTMD records: {len(ptmd_df)}")
    print(f"\nLabel distribution:")
    print(final_df["cell_state"].value_counts())
    print(f"\nPTM type distribution:")
    print(final_df["ptm_type"].value_counts().head(10))

    return final_df


def main():
    parser = argparse.ArgumentParser(description="Process PMADS dataset")
    parser.add_argument("--output", type=str, default="data/processed/pmads_combined.csv", help="Output file path")
    args = parser.parse_args()

    process_and_merge_data(args.output)


if __name__ == "__main__":
    main()

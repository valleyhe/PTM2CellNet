import json
import pickle
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def test_exporter_selects_explicit_source_rows_and_compacts_vocabulary(tmp_path):
    checkpoint = tmp_path / "encoder.pt"
    vocabulary = tmp_path / "token_dict.json"
    output = tmp_path / "asset"
    matrix = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    torch.save({"state_dict": {"token_embedding": {"weight": matrix}}}, checkpoint)
    vocabulary.write_text(json.dumps({"ENSG_B": 4, "ENSG_A": 1}), encoding="utf-8")
    script = Path(__file__).parents[3] / "scripts" / "export_perturbgen_gene_embeddings.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--tensor-key",
            "state_dict.token_embedding.weight",
            "--vocabulary",
            str(vocabulary),
            "--output-dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    exported = load_file(str(output / "gene_embeddings.safetensors"))["gene_embeddings"]
    assert torch.equal(exported, matrix[[1, 4]])
    assert json.loads((output / "vocabulary.json").read_text()) == {"ENSG_A": 0, "ENSG_B": 1}


def test_exporter_rejects_missing_explicit_tensor_key(tmp_path):
    checkpoint = tmp_path / "encoder.pt"
    vocabulary = tmp_path / "token_dict.json"
    torch.save({"weight": torch.ones(2, 2)}, checkpoint)
    vocabulary.write_text(json.dumps({"ENSG_A": 0}), encoding="utf-8")
    script = Path(__file__).parents[3] / "scripts" / "export_perturbgen_gene_embeddings.py"
    result = subprocess.run(
        [sys.executable, str(script), "--checkpoint", str(checkpoint), "--tensor-key", "guessed.weight",
         "--vocabulary", str(vocabulary), "--output-dir", str(tmp_path / "asset")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "explicit tensor key not found" in result.stderr


def test_exporter_accepts_flat_dotted_state_dict_key(tmp_path):
    checkpoint = tmp_path / "encoder.pt"
    vocabulary = tmp_path / "token_dict.json"
    output = tmp_path / "asset"
    matrix = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch.save({"transformer.encoder.embedding.weight": matrix}, checkpoint)
    vocabulary.write_text(json.dumps({"ENSG_A": 0, "ENSG_B": 1}), encoding="utf-8")
    script = Path(__file__).parents[3] / "scripts" / "export_perturbgen_gene_embeddings.py"
    result = subprocess.run(
        [sys.executable, str(script), "--checkpoint", str(checkpoint),
         "--tensor-key", "transformer.encoder.embedding.weight",
         "--vocabulary", str(vocabulary), "--output-dir", str(output)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_exporter_accepts_upstream_pickle_vocabulary(tmp_path):
    checkpoint = tmp_path / "encoder.ckpt"
    vocabulary = tmp_path / "token_dict.pkl"
    output = tmp_path / "asset"
    matrix = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.save(
        {"state_dict": {"transformer.token_embedding.weight": matrix}},
        checkpoint,
    )
    with vocabulary.open("wb") as handle:
        pickle.dump({"<pad>": 0, "ENSG000001": 2}, handle)
    script = Path(__file__).parents[3] / "scripts" / "export_perturbgen_gene_embeddings.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--tensor-key",
            "state_dict.transformer.token_embedding.weight",
            "--vocabulary",
            str(vocabulary),
            "--output-dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["vocabulary_format"] == "pkl"
    assert json.loads((output / "vocabulary.json").read_text()) == {
        "<pad>": 0,
        "ENSG000001": 1,
    }

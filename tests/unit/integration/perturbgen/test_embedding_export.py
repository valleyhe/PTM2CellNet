import json
import sys

import pytest

from src.integration.perturbgen.embedding_export import (
    EmbeddingExportCommand,
    write_gate0_failure_evidence,
)


def test_export_command_uses_argv_and_rejects_existing_output(tmp_path):
    root = tmp_path / "project"
    script = root / "scripts" / "export_perturbgen_gene_embeddings.py"
    script.parent.mkdir(parents=True)
    script.write_text("", encoding="utf-8")
    checkpoint = tmp_path / "encoder.ckpt"
    vocabulary = tmp_path / "vocabulary.json"
    checkpoint.write_bytes(b"checkpoint")
    vocabulary.write_text("{}", encoding="utf-8")
    output = tmp_path / "asset"
    command = EmbeddingExportCommand(
        python_executable=__import__("pathlib").Path(sys.executable),
        checkpoint=checkpoint,
        tensor_key="state_dict.encoder.embedding.weight",
        vocabulary=vocabulary,
        output_dir=output,
    )
    argv = command.build_argv(root)
    assert isinstance(argv, list)
    assert "state_dict.encoder.embedding.weight" in argv
    output.mkdir()
    with pytest.raises(FileExistsError):
        command.build_argv(root)


def test_gate0_failure_evidence_is_explicit(tmp_path):
    path = write_gate0_failure_evidence(
        tmp_path / "evidence.json",
        reason_codes=["encoder_weight_missing", "cohort_missing", "encoder_weight_missing"],
    )
    payload = json.loads(path.read_text())
    assert payload["status"] == "blocked"
    assert payload["reason_codes"] == ["cohort_missing", "encoder_weight_missing"]

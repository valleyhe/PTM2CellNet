from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.integration.perturbgen.dimensions import (
    PerturbGenDimensionError,
    PerturbGenDimensions,
    derive_perturbgen_dimensions,
    parse_dimension_probe_output,
)


def test_parse_dimension_probe_ignores_external_progress_lines():
    dimensions = parse_dimension_probe_output(
        "Loading dataset...\n"
        '{"tgt_vocab_size": 2005, "max_seq_length": 248}\n'
    )

    assert dimensions == PerturbGenDimensions(tgt_vocab_size=2005, max_seq_length=248)


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "{}",
        '{"tgt_vocab_size": true, "max_seq_length": 248}',
        '{"tgt_vocab_size": 2005, "max_seq_length": 0}',
        "not-json",
    ],
)
def test_parse_dimension_probe_rejects_invalid_output(stdout):
    with pytest.raises(PerturbGenDimensionError):
        parse_dimension_probe_output(stdout)


def test_derive_dimensions_runs_probe_in_external_environment(tmp_path, monkeypatch):
    external_python = tmp_path / "python"
    external_python.write_text("", encoding="utf-8")
    src_dataset = tmp_path / "src.dataset"
    src_dataset.mkdir()
    target_folder = tmp_path / "tgt"
    target_folder.mkdir()
    cwd = tmp_path / "repo"
    cwd.mkdir()

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout='progress\n{"tgt_vocab_size": 2005, "max_seq_length": 248}\n',
            stderr="",
        )

    monkeypatch.setattr(
        "src.integration.perturbgen.dimensions.subprocess.run",
        fake_run,
    )
    result = derive_perturbgen_dimensions(
        external_python,
        src_dataset=src_dataset,
        tgt_dataset_folder=target_folder,
        cwd=cwd,
    )

    assert result == PerturbGenDimensions(tgt_vocab_size=2005, max_seq_length=248)
    assert captured["command"][0] == str(external_python.resolve())
    assert captured["command"][1] == "-c"
    assert captured["command"][-2:] == [
        str(src_dataset.resolve()),
        str(target_folder.resolve()),
    ]
    assert captured["kwargs"]["cwd"] == str(cwd.resolve())
    assert captured["kwargs"]["check"] is False

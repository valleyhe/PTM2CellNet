"""Main-process contract for the external PerturbGen embedding exporter."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from src.models.perturbgen_embedding import (
    PerturbGenEmbeddingAsset,
    load_perturbgen_embedding_asset,
)


@dataclass(frozen=True)
class EmbeddingExportCommand:
    """Validated argv for running the custom exporter in an isolated Python."""

    python_executable: Path
    checkpoint: Path
    tensor_key: str
    vocabulary: Path
    output_dir: Path

    def build_argv(self, project_root: str | Path) -> list[str]:
        root = Path(project_root).resolve(strict=True)
        script = (root / "scripts" / "export_perturbgen_gene_embeddings.py").resolve(strict=True)
        python_path = self.python_executable.expanduser().resolve(strict=True)
        checkpoint = self.checkpoint.expanduser().resolve(strict=True)
        vocabulary = self.vocabulary.expanduser().resolve(strict=True)
        if not self.tensor_key.strip() or self.tensor_key.startswith(".") or self.tensor_key.endswith("."):
            raise ValueError("tensor_key must be an explicit non-empty dot-separated key")
        output = self.output_dir.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"embedding export output already exists: {output}")
        return [
            str(python_path),
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--tensor-key",
            self.tensor_key,
            "--vocabulary",
            str(vocabulary),
            "--output-dir",
            str(output),
        ]


def validate_embedding_export(
    output_dir: str | Path,
    *,
    expected_checkpoint_sha256: str,
    expected_tensor_key: str,
) -> PerturbGenEmbeddingAsset:
    """Validate exporter output and bind it to the requested source asset."""

    asset = load_perturbgen_embedding_asset(output_dir)
    source = asset.manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("embedding manifest source must be an object")
    if source.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("embedding manifest checkpoint hash does not match requested asset")
    if source.get("tensor_key") != expected_tensor_key:
        raise ValueError("embedding manifest tensor key does not match requested key")
    return asset


def write_gate0_failure_evidence(
    destination: str | Path,
    *,
    reason_codes: list[str],
) -> Path:
    """Write explicit Gate-0 evidence without claiming a successful spike."""

    if not reason_codes:
        raise ValueError("at least one Gate-0 reason code is required")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "gate": "Gate-0",
        "status": "blocked",
        "reason_codes": sorted(set(reason_codes)),
        "python": sys.version.split()[0],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path

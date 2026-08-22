"""PerturbGen gene vocabulary resolver.

Strict symbol/Ensembl/token resolution for DAVF × PerturbGen integration.
Unknown genes must fail fast; hash fallback is forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping

import pandas as pd

_ENSEMBL_RE = re.compile(r"^ENSG\d+$")
_ENSEMBL_VERSION_RE = re.compile(r"\.\d+$")


def normalize_ensembl_id(value: str) -> str:
    """Return canonical ENSG identifier without version suffix."""
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("ensembl_id must not be empty")
    normalized = _ENSEMBL_VERSION_RE.sub("", normalized)
    if not _ENSEMBL_RE.fullmatch(normalized):
        raise ValueError(f"invalid Ensembl gene id: {value!r}")
    return normalized


def normalize_gene_symbol(value: str) -> str:
    """Return canonical upper-case gene symbol."""
    normalized = str(value).strip().upper()
    if not normalized:
        raise ValueError("gene_symbol must not be empty")
    return normalized


@dataclass(frozen=True)
class GeneVocabularyEntry:
    """Single canonical gene vocabulary entry."""

    ensembl_id: str
    gene_symbol: str
    token_id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ensembl_id", normalize_ensembl_id(self.ensembl_id))
        object.__setattr__(self, "gene_symbol", normalize_gene_symbol(self.gene_symbol))
        if self.token_id is not None and self.token_id < 0:
            raise ValueError("token_id must be >= 0")


class GeneVocabularyResolver:
    """Exact resolver for symbol ↔ ENSG ↔ token ids.

    The mapping is intentionally strict:

    - unknown genes raise ``KeyError``;
    - ambiguous symbol/ENSG/token collisions raise ``ValueError`` at build time;
    - no hash, modulo, or random fallback is allowed.
    """

    def __init__(self, entries: Iterable[GeneVocabularyEntry]):
        canonical_entries = tuple(entries)
        if not canonical_entries:
            raise ValueError("gene vocabulary must not be empty")

        ensembl_to_symbol: dict[str, str] = {}
        symbol_to_ensembl: dict[str, str] = {}
        token_to_ensembl: dict[int, str] = {}
        ensembl_to_token: dict[str, int] = {}

        for entry in canonical_entries:
            existing_symbol = ensembl_to_symbol.get(entry.ensembl_id)
            if existing_symbol is not None and existing_symbol != entry.gene_symbol:
                raise ValueError(
                    "duplicate ensembl_id maps to different symbols: "
                    f"{entry.ensembl_id} -> {existing_symbol!r} / {entry.gene_symbol!r}"
                )

            existing_ensembl = symbol_to_ensembl.get(entry.gene_symbol)
            if existing_ensembl is not None and existing_ensembl != entry.ensembl_id:
                raise ValueError(
                    "duplicate gene_symbol maps to different ensembl ids: "
                    f"{entry.gene_symbol} -> {existing_ensembl!r} / {entry.ensembl_id!r}"
                )

            ensembl_to_symbol[entry.ensembl_id] = entry.gene_symbol
            symbol_to_ensembl[entry.gene_symbol] = entry.ensembl_id

            if entry.token_id is not None:
                existing_gene_token = ensembl_to_token.get(entry.ensembl_id)
                if existing_gene_token is not None and existing_gene_token != entry.token_id:
                    raise ValueError(
                        "duplicate ensembl_id maps to different token ids: "
                        f"{entry.ensembl_id} -> {existing_gene_token} / {entry.token_id}"
                    )
                existing_token_gene = token_to_ensembl.get(entry.token_id)
                if existing_token_gene is not None and existing_token_gene != entry.ensembl_id:
                    raise ValueError(
                        "duplicate token_id maps to different ensembl ids: "
                        f"{entry.token_id} -> {existing_token_gene!r} / {entry.ensembl_id!r}"
                    )
                token_to_ensembl[entry.token_id] = entry.ensembl_id
                ensembl_to_token[entry.ensembl_id] = entry.token_id

        self._entries = canonical_entries
        self._ensembl_to_symbol = ensembl_to_symbol
        self._symbol_to_ensembl = symbol_to_ensembl
        self._token_to_ensembl = token_to_ensembl
        self._ensembl_to_token = ensembl_to_token

    @classmethod
    def from_var(
        cls,
        var: pd.DataFrame,
        *,
        ensembl_col: str = "ensembl_id",
        symbol_col: str | None = None,
        token_map: Mapping[str, int] | None = None,
        token_col: str | None = None,
    ) -> "GeneVocabularyResolver":
        """Build a resolver from ``AnnData.var``-like table."""
        if ensembl_col not in var.columns:
            raise ValueError(f"var must contain column {ensembl_col!r}")
        if token_map is not None and token_col is not None:
            raise ValueError("token_map and token_col are mutually exclusive")

        entries: list[GeneVocabularyEntry] = []
        for row_index, (_, row) in enumerate(var.iterrows()):
            if symbol_col is None:
                gene_symbol = row.name
            else:
                if symbol_col not in var.columns:
                    raise ValueError(f"var must contain column {symbol_col!r}")
                gene_symbol = row[symbol_col]

            token_id: int | None = None
            canonical_ensembl = normalize_ensembl_id(row[ensembl_col])
            if token_map is not None:
                token_id = token_map.get(canonical_ensembl)
            elif token_col is not None:
                if token_col not in var.columns:
                    raise ValueError(f"var must contain column {token_col!r}")
                raw_token = row[token_col]
                if pd.isna(raw_token):
                    token_id = None
                else:
                    token_id = int(raw_token)

            try:
                entries.append(
                    GeneVocabularyEntry(
                        ensembl_id=canonical_ensembl,
                        gene_symbol=gene_symbol,
                        token_id=token_id,
                    )
                )
            except ValueError as exc:
                raise ValueError(f"invalid gene vocabulary row {row_index}: {exc}") from exc
        return cls(entries)

    @property
    def entries(self) -> tuple[GeneVocabularyEntry, ...]:
        return self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def has_symbol(self, gene_symbol: str) -> bool:
        return normalize_gene_symbol(gene_symbol) in self._symbol_to_ensembl

    def has_ensembl(self, ensembl_id: str) -> bool:
        return normalize_ensembl_id(ensembl_id) in self._ensembl_to_symbol

    def resolve_symbol(self, gene_symbol: str) -> str:
        normalized = normalize_gene_symbol(gene_symbol)
        try:
            return self._symbol_to_ensembl[normalized]
        except KeyError as exc:
            raise KeyError(
                f"gene symbol {gene_symbol!r} is not present in PerturbGen vocabulary; "
                "hash fallback is disabled"
            ) from exc

    def resolve_ensembl(self, ensembl_id: str) -> str:
        normalized = normalize_ensembl_id(ensembl_id)
        try:
            return self._ensembl_to_symbol[normalized]
        except KeyError as exc:
            raise KeyError(
                f"ensembl id {ensembl_id!r} is not present in PerturbGen vocabulary; "
                "hash fallback is disabled"
            ) from exc

    def resolve_token(
        self,
        *,
        gene_symbol: str | None = None,
        ensembl_id: str | None = None,
    ) -> int:
        if (gene_symbol is None) == (ensembl_id is None):
            raise ValueError("provide exactly one of gene_symbol or ensembl_id")

        canonical_ensembl = (
            self.resolve_symbol(gene_symbol)
            if gene_symbol is not None
            else normalize_ensembl_id(ensembl_id)
        )
        try:
            return self._ensembl_to_token[canonical_ensembl]
        except KeyError as exc:
            lookup_value = gene_symbol if gene_symbol is not None else ensembl_id
            raise KeyError(
                f"token id for {lookup_value!r} is not present in PerturbGen vocabulary; "
                "hash fallback is disabled"
            ) from exc

    def canonical_pair(
        self,
        *,
        gene_symbol: str | None = None,
        ensembl_id: str | None = None,
    ) -> tuple[str, str]:
        if (gene_symbol is None) == (ensembl_id is None):
            raise ValueError("provide exactly one of gene_symbol or ensembl_id")
        if gene_symbol is not None:
            canonical_ensembl = self.resolve_symbol(gene_symbol)
            return self.resolve_ensembl(canonical_ensembl), canonical_ensembl
        canonical_ensembl = normalize_ensembl_id(ensembl_id)
        return self.resolve_ensembl(canonical_ensembl), canonical_ensembl

import pandas as pd
import pytest

from src.models.gene_vocabulary import (
    GeneVocabularyEntry,
    GeneVocabularyResolver,
)


def test_resolver_normalizes_versions_and_resolves_bidirectionally():
    resolver = GeneVocabularyResolver(
        [
            GeneVocabularyEntry("ENSG00000141510.18", "tp53", token_id=7),
            GeneVocabularyEntry("ENSG00000146648", "EGFR", token_id=9),
        ]
    )

    assert resolver.resolve_symbol("TP53") == "ENSG00000141510"
    assert resolver.resolve_ensembl("ENSG00000141510.2") == "TP53"
    assert resolver.resolve_token(gene_symbol="tp53") == 7
    assert resolver.canonical_pair(ensembl_id="ENSG00000146648.7") == ("EGFR", "ENSG00000146648")


def test_resolver_rejects_collisions():
    with pytest.raises(ValueError, match="different ensembl ids"):
        GeneVocabularyResolver(
            [
                GeneVocabularyEntry("ENSG00000141510", "TP53", token_id=1),
                GeneVocabularyEntry("ENSG00000269305", "TP53", token_id=2),
            ]
        )

    with pytest.raises(ValueError, match="different token ids"):
        GeneVocabularyResolver(
            [
                GeneVocabularyEntry("ENSG00000141510", "TP53", token_id=1),
                GeneVocabularyEntry("ENSG00000141510", "TP53", token_id=2),
            ]
        )

    with pytest.raises(ValueError, match="different ensembl ids"):
        GeneVocabularyResolver.from_var(
            pd.DataFrame(
                {"ensembl_id": ["ENSG00000141510", "ENSG00000269305"]},
                index=["TP53", "TP53"],
            )
        )


def test_resolver_has_no_hash_fallback():
    resolver = GeneVocabularyResolver([GeneVocabularyEntry("ENSG00000141510", "TP53", token_id=1)])

    with pytest.raises(KeyError, match="hash fallback is disabled"):
        resolver.resolve_symbol("NOT_A_GENE")
    with pytest.raises(KeyError, match="hash fallback is disabled"):
        resolver.resolve_token(ensembl_id="ENSG99999999999")


def test_from_var_uses_index_as_symbol_and_token_map():
    var = pd.DataFrame(
        {"ensembl_id": ["ENSG00000141510.18", "ENSG00000146648"]},
        index=["tp53", "egfr"],
    )
    resolver = GeneVocabularyResolver.from_var(
        var,
        token_map={
            "ENSG00000141510": 3,
            "ENSG00000146648": 5,
        },
    )

    assert resolver.resolve_token(gene_symbol="EGFR") == 5

import json

import numpy as np

from src.data.features import FeatureExtractor


def test_extract_ptm_features_accepts_json_string_and_list():
    extractor = FeatureExtractor(
        {
            "data": {
                "max_sequence_length": 6,
                "ptm_types": ["phosphorylation", "acetylation"],
            }
        }
    )
    ptm_sites = [
        {"position": 2, "type": "phosphorylation"},
        {"position": 4, "type": "acetylation"},
    ]

    from_json = extractor.extract_ptm_features(json.dumps(ptm_sites), sequence_length=4)
    from_list = extractor.extract_ptm_features(ptm_sites, sequence_length=4)

    np.testing.assert_array_equal(
        from_json["position_features"],
        from_list["position_features"],
    )
    np.testing.assert_array_equal(
        from_json["count_features"],
        from_list["count_features"],
    )


def test_extract_ptm_features_array_matches_flat_contract():
    extractor = FeatureExtractor(
        {
            "data": {
                "max_sequence_length": 6,
                "ptm_types": ["phosphorylation", "acetylation"],
            }
        }
    )
    ptm_sites = [
        {"position": 2, "type": "phosphorylation"},
        {"position": 4, "type": "acetylation"},
    ]

    features = extractor.extract_ptm_features_array(ptm_sites, sequence_length=4)

    assert features.shape == (6 * 2 + 2,)
    assert features[2] == 1.0
    assert features[7] == 1.0
    np.testing.assert_array_equal(features[-2:], np.array([1.0, 1.0], dtype=np.float32))


def test_extract_structural_features_has_expected_shape():
    extractor = FeatureExtractor({"data": {"max_sequence_length": 5}})

    structural = extractor.extract_structural_features("ACD")

    assert structural.shape == (5, 3)
    assert np.all(np.isfinite(structural))
    assert np.allclose(structural[3:], 0.0)


def test_extract_sequence_features_includes_structural_features_when_enabled():
    base_config = {
        "data": {"max_sequence_length": 4},
        "features": {
            "sequence_encoding": "onehot",
            "include_physicochemical": False,
            "include_structural_features": False,
        },
    }
    structural_config = {
        "data": {"max_sequence_length": 4},
        "features": {
            "sequence_encoding": "onehot",
            "include_physicochemical": False,
            "include_structural_features": True,
        },
    }

    without_structural = FeatureExtractor(base_config).extract_sequence_features(["AC"])
    with_structural = FeatureExtractor(structural_config).extract_sequence_features(["AC"])

    assert with_structural.shape[0] == 1
    assert with_structural.shape[1] == without_structural.shape[1] + (4 * 3)


def test_use_feature_extractor_returns_combined_vector():
    config = {
        "data": {"max_sequence_length": 4},
        "features": {
            "sequence_encoding": "onehot",
            "include_physicochemical": False,
            "include_ptm_features": True,
            "use_feature_extractor": True,
        },
    }
    extractor = FeatureExtractor(config)

    features = extractor.extract_features_for_sample(
        "AC",
        [{"position": 2, "type": "phosphorylation"}],
    )

    expected_seq_len = 4 * (len("ACDEFGHIKLMNPQRSTVWY") + 1)
    expected_ptm_len = 4 * 5 + 5
    assert features.shape == (expected_seq_len + expected_ptm_len,)
    assert features.dtype == np.float32

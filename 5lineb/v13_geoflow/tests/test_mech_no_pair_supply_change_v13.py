import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from reconstruct_mech_no_pair_supply_v13 import (
    crossfit_typical_change,
    hospital_service_range_features,
    linear_channel_change,
    reference_vector,
    split_channels,
    type_scope_summary,
    type_typical_scope_features,
)
from plot_mech_no_pair_supply_change_abcd_v13 import holm, weighted_destination_summary


def test_external_channel_difference_is_additive():
    rng = np.random.default_rng(12)
    h = rng.normal(size=(7, 12))
    coef = rng.normal(size=(3, 12))
    h, width = split_channels(h)
    reference = reference_vector(h, rng.normal(size=(7, 8)), width)
    total, parts = linear_channel_change(coef, h, reference, width)
    expected = (h - reference) @ coef.T
    np.testing.assert_allclose(total, expected)
    np.testing.assert_allclose(parts.sum(axis=0), expected)
    np.testing.assert_array_equal(reference[:, :width], h[:, :width])


def test_typical_reference_and_decoder_are_cross_fitted():
    rng = np.random.default_rng(19)
    target = rng.normal(size=(18, 3))
    base = target @ rng.normal(size=(3, 4)) + rng.normal(scale=0.05, size=(18, 4))
    external = target @ rng.normal(size=(3, 8)) + rng.normal(scale=0.05, size=(18, 8))
    h = np.concatenate([base, external], axis=1)
    folds = np.repeat(np.arange(3), 6)
    result = crossfit_typical_change(h, target, folds, base, random_state=31)
    assert result["reference"].shape == h.shape
    assert result["delta"].shape == target.shape
    np.testing.assert_allclose(result["reference"][:, :4], base)
    np.testing.assert_allclose(result["parts"].sum(axis=0), result["delta"], rtol=1e-10, atol=1e-10)


def test_service_range_features_exclude_self_heldout_and_chc_context():
    h = np.arange(36, dtype=float).reshape(3, 12)
    target = np.arange(27, dtype=float).reshape(3, 9)
    columns = [
        "RA_TH", "SA_TH", "TC_TH", "RA_SH", "SA_SH", "TC_SH",
        "RA_CHC", "SA_CHC", "TC_CHC",
    ]
    W = np.array([[100.0, 4.0, 9.0], [2.0, 100.0, 7.0], [5.0, 6.0, 100.0]])
    heldout = np.zeros_like(W, dtype=bool)
    heldout[0, 2] = True
    features, names = hospital_service_range_features(
        h, target, W, heldout, columns, np.array([0, 1, 2])
    )
    assert features.shape == (3, 19)
    assert len(names) == features.shape[1]
    assert not any("CHC" in name for name in names)
    np.testing.assert_allclose(features[0, 9:15], target[1, :6])
    np.testing.assert_array_equal(features[:, 15:], np.eye(4)[:3])


def test_type_typical_scope_is_estimated_from_other_spatial_folds():
    features = np.zeros((8, 11), dtype=float)
    features[:, 2:7] = np.arange(40, dtype=float).reshape(8, 5)
    node_type = np.tile(np.arange(4), 2)
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    query = type_typical_scope_features(features, node_type, folds, width=2)
    np.testing.assert_allclose(query[:4, 2:7], features[4:, 2:7])
    np.testing.assert_allclose(query[4:, 2:7], features[:4, 2:7])
    summary = type_scope_summary(features, node_type, folds, width=2)
    assert len(summary) == 8
    first = summary.query("heldout_fold == 0 and baseline_configuration == 0").iloc[0]
    assert first.training_n == 1
    np.testing.assert_allclose(first.visible_external_flow_median, np.expm1(20.0))


def test_holm_adjustment_is_monotone_in_sorted_order():
    raw = np.array([0.03, 0.001, 0.02, 0.2])
    adjusted = holm(raw.tolist())
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all((adjusted >= raw) & (adjusted <= 1))


def test_destination_summary_uses_origin_shares_and_excludes_self_flows():
    change = np.array([1.0, 3.0, 8.0])
    node_type = np.array([0, 1, 0])
    cluster = np.array([1, 1, 2])
    W = np.array([[100.0, 4.0, 9.0], [2.0, 100.0, 7.0], [5.0, 6.0, 100.0]])
    table = weighted_destination_summary(change, node_type, cluster, W, cluster_ids=(1,))
    scarcity = table.loc[table.destination_type == 0].iloc[0]
    chc = table.loc[table.destination_type == 1].iloc[0]
    assert scarcity.weighted_change == 3.0
    assert chc.weighted_change == 1.0
    assert scarcity.origin_n == 1 and chc.origin_n == 1

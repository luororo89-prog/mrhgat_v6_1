import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from reconstruct_mech_no_pair_supply_v13 import linear_channel_change, local_reference, split_channels
from plot_mech_no_pair_supply_change_abcd_v13 import holm, weighted_destination_summary


def test_external_channel_difference_is_additive():
    rng = np.random.default_rng(12)
    h = rng.normal(size=(7, 12))
    coef = rng.normal(size=(3, 12))
    h, width = split_channels(h)
    local = local_reference(h, width)
    total, parts = linear_channel_change(coef, h, width)
    expected = (h - local) @ coef.T
    np.testing.assert_allclose(total, expected)
    np.testing.assert_allclose(parts.sum(axis=0), expected)
    np.testing.assert_array_equal(local[:, :width], h[:, :width])
    np.testing.assert_array_equal(local[:, width:], np.zeros_like(local[:, width:]))


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

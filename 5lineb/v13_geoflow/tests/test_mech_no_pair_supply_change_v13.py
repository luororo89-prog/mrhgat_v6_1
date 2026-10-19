import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from reconstruct_mech_no_pair_supply_v13 import (
    crossfit_typical_change,
    linear_channel_change,
    reference_vector,
    split_channels,
)
from plot_mech_no_pair_supply_change_abcd_v13 import (
    SUPPLY_BLOCKS,
    cluster_cumulative_summary,
    cluster_relation_summary,
    holm,
    weighted_destination_summary,
)


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
    result = crossfit_typical_change(h, target, folds)
    assert result["reference"].shape == h.shape
    assert result["delta"].shape == target.shape
    np.testing.assert_allclose(result["reference"][:, :4], base)
    np.testing.assert_allclose(result["parts"].sum(axis=0), result["delta"], rtol=1e-10, atol=1e-10)


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


def test_cluster_cumulative_shift_starts_at_baseline_and_ends_at_change_9():
    values = {
        column: np.array([0.0, 0.0])
        for _, columns in SUPPLY_BLOCKS
        for column in columns
    }
    values["delta_RA_TH"] = np.array([3.0, 1.0])
    values["delta_RA_SH"] = np.array([4.0, 2.0])
    values["delta_RA_CHC"] = np.array([12.0, 2.0])
    energy = sum(np.square(values[column]) for column in values)
    values["change_9"] = np.sqrt(energy / 9.0)
    table = cluster_cumulative_summary(pd.DataFrame(values), np.array([1, 1]), cluster_ids=(1,))
    trajectory = table.sort_values("stage").mean_cumulative_shift.to_numpy()
    assert trajectory[0] == 0.0
    assert np.all(np.diff(trajectory) >= 0.0)
    np.testing.assert_allclose(trajectory[-1], values["change_9"].mean())


def test_cluster_relation_uses_concentration_and_excludes_self_outflow():
    data = pd.DataFrame({"baseline_type": [0, 1, 0], "change_9": [1.0, 3.0, 8.0]})
    cluster = np.array([1, 1, 2])
    W = np.array([[100.0, 4.0, 0.0], [2.0, 100.0, 0.0], [0.0, 0.0, 100.0]])
    table = cluster_relation_summary(data, cluster, W, cluster_ids=(1,))
    row = table.iloc[0]
    assert row.outgoing_flow == 6.0
    np.testing.assert_allclose(row.configuration_concentration, (4.0 / 6.0) ** 2 + (2.0 / 6.0) ** 2)
    assert row.mean_change == 2.0

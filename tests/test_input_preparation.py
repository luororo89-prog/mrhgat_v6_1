"""Checks of preservation and preprocessing, not acceptance of scientific gates."""
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare", ROOT / "src/A_data/prepare_inputs.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class InputPreparationTests(unittest.TestCase):
    def test_duplicate_and_fractional_ids_rejected(self):
        for values in ([2, 2], [2, 3.5], [2, None]):
            with self.assertRaises(ValueError):
                prepare.integer_ids(pd.Series(values), unique=True)

    def test_row_order_is_explicit_not_assumed(self):
        flow = pd.DataFrame({"source": [20, 10], "target": [10, 20], "flow_count": [7, 3]})
        np.testing.assert_array_equal(prepare.reconstruct_flows(flow, [10, 20]), [[0, 3], [7, 0]])
        with self.assertRaises(ValueError):
            prepare.reconstruct_flows(flow, [10, 30])

    def test_legacy_values_and_classes_preserved(self):
        source = prepare.read_csv(prepare.BASE / prepare.PATHS["supply"])
        saved = pd.read_parquet(ROOT / "data/processed/legacy_supply_raw.parquet")
        pd.testing.assert_frame_equal(saved, source[["OBJECTID"] + prepare.FEATURES])
        classes = pd.read_parquet(ROOT / "data/processed/original_classes.parquet")
        pd.testing.assert_frame_equal(classes, source[["OBJECTID", "Cluster_Label"]])
        z = pd.read_parquet(ROOT / "data/processed/legacy_supply_zscore.parquet")
        self.assertNotIn("Cluster_Label", z.columns)
        np.testing.assert_allclose(z[prepare.FEATURES].mean(), 0, atol=1e-12)
        np.testing.assert_allclose(z[prepare.FEATURES].std(ddof=0), 1, atol=1e-12)

    def test_original_network_preserved(self):
        order = pd.read_parquet(ROOT / "data/processed/verified_matrix_node_order.parquet").OBJECTID
        flow = pd.read_parquet(ROOT / "data/processed/flows_community_aggregate.parquet")
        original = np.loadtxt(prepare.BASE / prepare.PATHS["matrix"], delimiter=",")
        np.testing.assert_array_equal(prepare.reconstruct_flows(flow, order), original)
        adj = pd.read_parquet(ROOT / "data/processed/adjacency.parquet")
        adj_flow = adj.rename(columns={"community_id": "source", "neighbor_id": "target"}).assign(flow_count=1)
        actual = prepare.reconstruct_flows(adj_flow, order)
        np.testing.assert_array_equal(actual, np.loadtxt(prepare.BASE / prepare.PATHS["queen"], delimiter=","))

    def test_time_conversion_is_unit_only(self):
        saved = pd.read_parquet(ROOT / "data/processed/travel_time_community_observed.parquet")
        source = pd.read_excel(prepare.BASE / prepare.PATHS["time"])
        pd.testing.assert_frame_equal(saved[source.columns], source)
        np.testing.assert_allclose(saved.time_min, source.time_s / 60)
        self.assertNotIn("t_peak", saved.columns)
        self.assertNotIn("t_free", saved.columns)

    def test_gate_and_source_integrity(self):
        status = json.loads((ROOT / "reports/A_readiness.json").read_text(encoding="utf-8"))
        self.assertFalse(status["ready_for_training"])
        self.assertTrue(status["blockers"])
        run = Path(status["run_directory"])
        original = json.loads((run / "source_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(original, prepare.source_manifest())
        if status.get('ready_for_synthetic_validation'):
            with np.load(ROOT / 'data/graph/graph_v6.npz') as graph:
                self.assertEqual(set(graph.files), {'x', 'F', 'K', 'ids', 'fold'})
                self.assertEqual(float(graph['F'].sum()), 330831)


if __name__ == "__main__":
    unittest.main(verbosity=2)

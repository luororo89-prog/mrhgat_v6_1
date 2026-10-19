"""Reconstruct relative supply shifts from ``mech_no_pair`` node vectors.

The saved ``arrays.npz`` file contains the final three-channel node
representation ``h = [h_base, h_comp, h_compete]``.  A Ridge decoder is fitted
from that representation to the nine frozen-ruler supply coordinates.  By
default, the typical external aggregation is the coordinate-wise median among
the 20 training-fold nodes of the same baseline configuration whose
standardized ``h_base`` vectors are closest to the evaluated node.  The
relative shift is therefore

    delta_x = D(h_base, h_ext) - D(h_base, median_20NN(h_ext | h_base, T)).

For each held-out spatial fold, neither the decoder nor the conditional
reference model is fitted on the evaluated nodes.  Common external mobility,
including routine movement within broad hospital catchments, can enter the
reference.  Because the aggregate OD does not identify facility level, the
residual is not uniquely attributable to community-health-centre movement and
is not an independently observed post-interaction supply outcome.

Example
-------
python 5lineb/v13_geoflow/reconstruct_mech_no_pair_supply_v13.py \
  --mechanism-results /path/to/outpatient_d0.25_od_mechanism_v13 \
  --baseline-table /path/to/node_results.csv \
  --folds /path/to/node_spatial_folds.csv \
  --ruler /path/to/ruler.json \
  --output /path/to/local_output
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
sys.path.insert(0, str(COMMON))
from decode import ALPHAS  # noqa: E402
from ruler import measure  # noqa: E402

VARIANT = "mech_no_pair"
CHANNELS = ("base", "comp", "compete")
CONFIGURATION_LABELS = ("scarcity", "chc_based", "sh_based", "abundance")
REFERENCE_TREES = 400
REFERENCE_MIN_SAMPLES_LEAF = 5
REFERENCE_MAX_FEATURES = 0.8
REFERENCE_RANDOM_STATE = 7100
KNN_NEIGHBORS = 20


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def frozen_ruler(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    frozen = report["main"]["frozen"]
    for key in ("U", "weights", "centers", "mean", "std"):
        if key in frozen and frozen[key] is not None:
            frozen[key] = np.asarray(frozen[key], dtype=float)
    return frozen


def ordered_baseline(path: Path, ids: np.ndarray, columns: list[str]) -> pd.DataFrame:
    table = pd.read_csv(path)
    needed = ["community_id", *[f"before_{column}" for column in columns]]
    missing = [column for column in needed if column not in table]
    if missing:
        raise ValueError(f"baseline table is missing columns: {missing}")
    candidates = [column for column in ("name", "baseline_type", "eligible") if column in table]
    source = table[needed + candidates].copy()
    if source.community_id.duplicated().any():
        numeric = [column for column in source if column.startswith("before_")]
        spread = source.groupby("community_id")[numeric].agg(lambda x: float(np.ptp(x.to_numpy(dtype=float))))
        if spread.to_numpy().max(initial=0.0) > 1e-12:
            raise ValueError("baseline coordinates differ across duplicate community rows")
        source = source.drop_duplicates("community_id")
    source = source.set_index("community_id").reindex(ids)
    if source.index.has_duplicates or source.isna().any().any():
        raise ValueError("baseline table does not match the fixed model node order")
    return source.reset_index()


def ordered_folds(path: Path, ids: np.ndarray) -> np.ndarray:
    table = pd.read_csv(path)
    if not {"community_id", "fold"} <= set(table):
        raise ValueError("fold table must contain community_id and fold")
    folds = table.drop_duplicates("community_id").set_index("community_id").reindex(ids)["fold"]
    if folds.isna().any() or folds.nunique() < 2:
        raise ValueError("fold table does not cover the fixed model node order")
    return folds.to_numpy()


def split_channels(h: np.ndarray) -> tuple[np.ndarray, int]:
    h = np.asarray(h, dtype=float)
    if h.ndim != 2 or h.shape[1] % len(CHANNELS):
        raise ValueError("aggregated representation must contain three equal channel blocks")
    return h, h.shape[1] // len(CHANNELS)


def reference_vector(h: np.ndarray, typical_external: np.ndarray, width: int) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    typical_external = np.asarray(typical_external, dtype=float)
    if typical_external.shape != (h.shape[0], h.shape[1] - width):
        raise ValueError("typical external aggregation has incompatible shape")
    return np.concatenate([h[:, :width], typical_external], axis=1)


def hospital_service_range_features(
    h: np.ndarray,
    target: np.ndarray,
    W: np.ndarray,
    heldout: np.ndarray,
    columns: list[str],
    node_type: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build reference covariates without using held-out OD edges or CHC context."""
    h, width = split_channels(h)
    W = np.asarray(W, dtype=float)
    heldout = np.asarray(heldout, dtype=bool)
    if W.shape != (len(h), len(h)) or heldout.shape != W.shape:
        raise ValueError("W and heldout must be square matrices aligned with h")
    if target.shape != (len(h), len(columns)):
        raise ValueError("target and columns must be aligned")
    node_type = np.asarray(node_type, dtype=int)
    if node_type.shape != (len(h),) or not np.isin(node_type, np.arange(4)).all():
        raise ValueError("node_type must contain the four baseline configurations 0--3")

    off_diagonal = ~np.eye(len(h), dtype=bool)
    visible = (W > 0) & off_diagonal & ~heldout
    context = W * visible
    external_flow = context.sum(axis=1)
    destination_count = visible.sum(axis=1)
    share = np.divide(
        context,
        external_flow[:, None],
        out=np.zeros_like(context),
        where=external_flow[:, None] > 0,
    )
    entropy = -(np.where(share > 0, share * np.log(share + 1e-300), 0.0)).sum(axis=1)
    entropy = np.divide(
        entropy,
        np.log(np.maximum(destination_count, 2)),
        out=np.zeros_like(entropy),
        where=destination_count > 1,
    )
    maximum_share = share.max(axis=1)
    hhi = np.square(share).sum(axis=1)
    effective_destinations = np.divide(1.0, hhi, out=np.zeros_like(hhi), where=hhi > 0)
    scope = np.column_stack([
        np.log1p(external_flow),
        np.log1p(destination_count),
        entropy,
        maximum_share,
        np.log1p(effective_destinations),
    ])

    hospital_indices = [
        index for index, column in enumerate(columns)
        if column.endswith("_TH") or column.endswith("_SH") or "_TH_" in column or "_SH_" in column
    ]
    if len(hospital_indices) != 6:
        raise ValueError("expected three TH and three SH hospital indicators")
    hospital_context = share @ target[:, hospital_indices]
    configuration = np.eye(4, dtype=float)[node_type]
    feature_names = (
        [f"h_base_{index + 1}" for index in range(width)]
        + ["log_visible_external_flow", "log_visible_destination_count", "destination_entropy",
           "maximum_destination_share", "log_effective_destination_count"]
        + [f"flow_weighted_destination_{columns[index]}" for index in hospital_indices]
        + [f"baseline_configuration_{index}" for index in range(4)]
    )
    return np.column_stack([h[:, :width], scope, hospital_context, configuration]), feature_names


def type_typical_scope_features(
    features: np.ndarray,
    node_type: np.ndarray,
    folds: np.ndarray,
    width: int,
) -> np.ndarray:
    """Replace each held-out node's scope by its training-fold type median."""
    features = np.asarray(features, dtype=float)
    node_type = np.asarray(node_type, dtype=int)
    folds = np.asarray(folds)
    if features.shape[0] != len(node_type) or folds.shape != node_type.shape:
        raise ValueError("features, node_type, and folds must share node order")
    query = features.copy()
    scope = slice(width, width + 5)
    for fold in np.unique(folds):
        train = folds != fold
        test = ~train
        for configuration in range(4):
            train_type = train & (node_type == configuration)
            test_type = test & (node_type == configuration)
            if test_type.any() and not train_type.any():
                raise ValueError(f"no training nodes for baseline configuration {configuration}")
            if test_type.any():
                query[test_type, scope] = np.median(features[train_type, scope], axis=0)
    return query


def type_scope_summary(
    features: np.ndarray,
    node_type: np.ndarray,
    folds: np.ndarray,
    width: int,
) -> pd.DataFrame:
    """Report the fold-specific type ranges used to construct the reference."""
    transformed = np.asarray(features, dtype=float)[:, width:width + 5]
    node_type = np.asarray(node_type, dtype=int)
    folds = np.asarray(folds)
    rows: list[dict[str, object]] = []
    names = (
        "visible_external_flow", "visible_destination_count", "destination_entropy",
        "maximum_destination_share", "effective_destination_count",
    )
    for fold in np.unique(folds):
        train = folds != fold
        for configuration, label in enumerate(CONFIGURATION_LABELS):
            values = transformed[train & (node_type == configuration)]
            if not len(values):
                raise ValueError(f"no training nodes for baseline configuration {configuration}")
            quantiles = np.quantile(values, [0.25, 0.5, 0.75], axis=0)
            quantiles[:, [0, 1, 4]] = np.expm1(quantiles[:, [0, 1, 4]])
            row: dict[str, object] = {
                "heldout_fold": fold,
                "baseline_configuration": configuration,
                "configuration_label": label,
                "training_n": len(values),
            }
            for column, name in enumerate(names):
                for quantile, suffix in enumerate(("q25", "median", "q75")):
                    row[f"{name}_{suffix}"] = float(quantiles[quantile, column])
            rows.append(row)
    return pd.DataFrame(rows)


def make_reference_model(random_state: int) -> ExtraTreesRegressor:
    """Conditional-mean estimator selected by spatial-fold validation."""
    return ExtraTreesRegressor(
        n_estimators=REFERENCE_TREES,
        min_samples_leaf=REFERENCE_MIN_SAMPLES_LEAF,
        max_features=REFERENCE_MAX_FEATURES,
        random_state=random_state,
        n_jobs=-1,
    )


def linear_channel_change(
    coef: np.ndarray,
    h: np.ndarray,
    reference: np.ndarray,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return total external change and the two additive channel parts."""
    coef = np.asarray(coef, dtype=float)
    h = np.asarray(h, dtype=float)
    reference = np.asarray(reference, dtype=float)
    residual = h - reference
    np.testing.assert_allclose(residual[:, :width], 0.0, atol=1e-12)
    parts = np.stack([
        residual[:, width:2 * width] @ coef[:, width:2 * width].T,
        residual[:, 2 * width:3 * width] @ coef[:, 2 * width:3 * width].T,
    ])
    return parts.sum(axis=0), parts


def crossfit_type_knn_median_change(
    h: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    node_type: np.ndarray,
    neighbors: int = KNN_NEIGHBORS,
) -> dict[str, object]:
    """Use same-type nearest training nodes in h_base as the external reference."""
    h, width = split_channels(h)
    folds = np.asarray(folds)
    node_type = np.asarray(node_type, dtype=int)
    if folds.shape != (len(h),) or node_type.shape != (len(h),):
        raise ValueError("folds and node_type must be aligned with h")
    if neighbors < 1:
        raise ValueError("neighbors must be positive")

    reference = np.empty_like(h, dtype=float)
    decoded_aggregated = np.empty_like(target, dtype=float)
    decoded_reference = np.empty_like(target, dtype=float)
    parts = np.empty((2, *target.shape), dtype=float)
    neighbor_index = np.full((len(h), neighbors), -1, dtype=int)
    neighbor_distance = np.full((len(h), neighbors), np.nan, dtype=float)
    decoder_alphas: list[float] = []

    for fold in np.unique(folds):
        test = folds == fold
        train = ~test
        train_index = np.flatnonzero(train)
        test_index = np.flatnonzero(test)
        scaler = StandardScaler().fit(h[train, :width])
        train_base = scaler.transform(h[train, :width])
        test_base = scaler.transform(h[test, :width])
        typical_external = np.empty((len(test_index), h.shape[1] - width), dtype=float)
        for local_index, node_index in enumerate(test_index):
            same_type = np.flatnonzero(node_type[train_index] == node_type[node_index])
            if len(same_type) < neighbors:
                raise ValueError(
                    f"baseline configuration {node_type[node_index]} has only {len(same_type)} "
                    f"training nodes; {neighbors} neighbors requested"
                )
            distance_squared = np.square(train_base[same_type] - test_base[local_index]).sum(axis=1)
            selected_positions = np.argpartition(distance_squared, neighbors - 1)[:neighbors]
            selected_positions = selected_positions[np.argsort(distance_squared[selected_positions])]
            selected_local = same_type[selected_positions]
            selected = train_index[selected_local]
            neighbor_index[node_index] = selected
            neighbor_distance[node_index] = np.sqrt(
                np.square(train_base[selected_local] - test_base[local_index]).sum(axis=1)
            )
            typical_external[local_index] = np.median(h[selected, width:], axis=0)

        fold_reference = reference_vector(h[test], typical_external, width)
        reference[test] = fold_reference
        decoder = RidgeCV(alphas=ALPHAS).fit(h[train], target[train])
        decoded_aggregated[test] = decoder.predict(h[test])
        decoded_reference[test] = decoder.predict(fold_reference)
        fold_delta, fold_parts = linear_channel_change(decoder.coef_, h[test], fold_reference, width)
        np.testing.assert_allclose(
            decoded_aggregated[test] - decoded_reference[test], fold_delta, rtol=1e-10, atol=1e-10
        )
        parts[:, test] = fold_parts
        decoder_alphas.append(float(decoder.alpha_))

    delta = decoded_aggregated - decoded_reference
    np.testing.assert_allclose(parts.sum(axis=0), delta, rtol=1e-10, atol=1e-10)
    return {
        "width": width,
        "reference": reference,
        "decoded_aggregated": decoded_aggregated,
        "decoded_reference": decoded_reference,
        "delta": delta,
        "parts": parts,
        "neighbor_index": neighbor_index,
        "neighbor_distance": neighbor_distance,
        "decoder_fold_alphas": np.asarray(decoder_alphas),
    }


def crossfit_typical_change(
    h: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    reference_features: np.ndarray,
    reference_query_features: np.ndarray | None = None,
    random_state: int = REFERENCE_RANDOM_STATE,
) -> dict[str, object]:
    """Estimate a typical external reference and decoded shift out of fold."""
    h, width = split_channels(h)
    reference = np.empty_like(h, dtype=float)
    decoded_aggregated = np.empty_like(target, dtype=float)
    decoded_reference = np.empty_like(target, dtype=float)
    parts = np.empty((2, *target.shape), dtype=float)
    decoder_alphas: list[float] = []
    reference_features = np.asarray(reference_features, dtype=float)
    if reference_features.shape[0] != h.shape[0]:
        raise ValueError("reference features must be aligned with h")
    if reference_query_features is None:
        reference_query_features = reference_features
    reference_query_features = np.asarray(reference_query_features, dtype=float)
    if reference_query_features.shape != reference_features.shape:
        raise ValueError("reference query features must match the fitted reference features")
    for fold_index, fold in enumerate(np.unique(folds)):
        test = folds == fold
        decoder = RidgeCV(alphas=ALPHAS).fit(h[~test], target[~test])
        reference_model = make_reference_model(random_state + fold_index)
        reference_model.fit(reference_features[~test], h[~test, width:])
        fold_reference = reference_vector(
            h[test], reference_model.predict(reference_query_features[test]), width
        )
        reference[test] = fold_reference
        decoded_aggregated[test] = decoder.predict(h[test])
        decoded_reference[test] = decoder.predict(fold_reference)
        fold_delta, fold_parts = linear_channel_change(decoder.coef_, h[test], fold_reference, width)
        np.testing.assert_allclose(
            decoded_aggregated[test] - decoded_reference[test], fold_delta, rtol=1e-10, atol=1e-10
        )
        parts[:, test] = fold_parts
        decoder_alphas.append(float(decoder.alpha_))
    delta = decoded_aggregated - decoded_reference
    np.testing.assert_allclose(parts.sum(axis=0), delta, rtol=1e-10, atol=1e-10)
    return {
        "width": width,
        "reference": reference,
        "decoded_aggregated": decoded_aggregated,
        "decoded_reference": decoded_reference,
        "delta": delta,
        "parts": parts,
        "reference_query_features": reference_query_features,
        "decoder_fold_alphas": np.asarray(decoder_alphas),
    }


def reconstruct_seed(
    h: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
    reference_features: np.ndarray,
    reference_query_features: np.ndarray,
    random_state: int,
) -> dict[str, object]:
    h, width = split_channels(h)
    decoder = RidgeCV(alphas=ALPHAS).fit(h, target)
    reference_model = make_reference_model(random_state).fit(reference_features, h[:, width:])
    result = crossfit_typical_change(
        h, target, folds, reference_features, reference_query_features, random_state
    )
    result.update({"decoder": decoder, "reference_model": reference_model, "oof": result["decoded_aggregated"]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-results", type=Path, required=True)
    parser.add_argument("--baseline-table", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--ruler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--reference-method", choices=("type_knn_median", "extra_trees"),
                        default="type_knn_median")
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    split_path = args.mechanism_results / "fixed_split.npz"
    with np.load(split_path) as split:
        ids = split["ids"].astype(int)
        W = split["W"].astype(float)
        H = split["H"].astype(bool)
    ruler = frozen_ruler(args.ruler)
    columns = list(ruler["columns"])
    baseline = ordered_baseline(args.baseline_table, ids, columns)
    folds = ordered_folds(args.folds, ids)
    X = baseline[[f"before_{column}" for column in columns]].to_numpy(dtype=float)
    nominal = measure(X, ruler)
    if "baseline_type" in baseline:
        np.testing.assert_array_equal(nominal["class"], baseline["baseline_type"].to_numpy(dtype=int))

    frames: list[pd.DataFrame] = []
    performance: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    scope_summary: pd.DataFrame | None = None
    input_hashes = {"fixed_split": sha256(split_path), "baseline_table": sha256(args.baseline_table),
                    "folds": sha256(args.folds), "ruler": sha256(args.ruler)}

    for seed in args.seeds:
        array_path = args.mechanism_results / VARIANT / f"seed{seed}" / "arrays.npz"
        with np.load(array_path) as saved:
            h = saved["h"].astype(float)
        if h.shape[0] != len(ids):
            raise ValueError(f"seed {seed} representation does not match fixed node order")
        _, width = split_channels(h)
        reference_feature_names: list[str] = []
        reference_model = None
        if args.reference_method == "extra_trees":
            reference_features, reference_feature_names = hospital_service_range_features(
                h, X, W, H, columns, nominal["class"]
            )
            reference_query_features = type_typical_scope_features(
                reference_features, nominal["class"], folds, width
            )
            if scope_summary is None:
                scope_summary = type_scope_summary(reference_features, nominal["class"], folds, width)
            result = reconstruct_seed(
                h, X, folds, reference_features, reference_query_features,
                REFERENCE_RANDOM_STATE + 100 * seed
            )
            reference_model = result["reference_model"]
        else:
            result = crossfit_type_knn_median_change(h, X, folds, nominal["class"])
            result.update({"decoder": RidgeCV(alphas=ALPHAS).fit(h, X),
                           "oof": result["decoded_aggregated"]})
        delta = result["delta"]
        parts = result["parts"]
        after = X + delta
        after_measure = measure(after, ruler)
        delta_pc = delta @ ruler["U"]
        change_9 = np.sqrt(np.mean(delta ** 2, axis=1))
        change_pc = np.sqrt(np.sum(delta_pc ** 2 * ruler["weights"], axis=1))
        oof = result["oof"]
        oof_measure = measure(oof, ruler)
        decoder = result["decoder"]
        width = int(result["width"])
        reference_error = result["reference"][:, width:] - h[:, width:]
        reference_r2 = [r2_score(h[:, j], result["reference"][:, j]) for j in range(width, h.shape[1])]
        reference_variance = float(np.mean(np.square(h[:, width:] - h[:, width:].mean(axis=0))))

        for index, column in enumerate(columns):
            error = oof[:, index] - X[:, index]
            performance.append({"seed": seed, "indicator": column, "alpha": float(decoder.alpha_),
                                "oof_rmse": float(np.sqrt(np.mean(error ** 2))),
                                "oof_mae": float(np.mean(np.abs(error))),
                                "oof_r2": float(r2_score(X[:, index], oof[:, index]))})
        eligible = W.sum(axis=1) > 0
        summaries.append({"seed": seed, "alpha": float(decoder.alpha_),
                          "reference_model": args.reference_method,
                          "oof_rmse": float(np.sqrt(np.mean((oof - X) ** 2))),
                          "oof_min_r2": float(min(r2_score(X[:, j], oof[:, j]) for j in range(X.shape[1]))),
                          "oof_type_agreement": float(np.mean(oof_measure["class"] == nominal["class"])),
                          "reference_oof_rmse": float(np.sqrt(np.mean(reference_error ** 2))),
                          "reference_oof_nrmse": float(np.sqrt(np.mean(reference_error ** 2) / reference_variance)),
                          "reference_oof_variance_weighted_r2": float(r2_score(
                              h[:, width:], result["reference"][:, width:], multioutput="variance_weighted"
                          )),
                          "reference_oof_mean_r2": float(np.mean(reference_r2)),
                          "eligible_n": int(eligible.sum()),
                          "mean_change_9": float(change_9[eligible].mean()),
                          "mean_change_pc": float(change_pc[eligible].mean()),
                          "changed_type_n": int(np.sum((after_measure["class"] != nominal["class"]) & eligible))})

        frame = pd.DataFrame({"community_id": ids, "seed": seed, "eligible": eligible,
                              "baseline_type": nominal["class"], "after_type": after_measure["class"],
                              "retained": nominal["class"] == after_measure["class"],
                              "change_9": change_9, "change_pc": change_pc})
        if "name" in baseline:
            frame["name"] = baseline["name"].to_numpy()
        if args.reference_method == "type_knn_median":
            frame["reference_neighbor_mean_distance"] = result["neighbor_distance"].mean(axis=1)
            frame["reference_neighbor_max_distance"] = result["neighbor_distance"].max(axis=1)
        for index, column in enumerate(columns):
            frame[f"before_{column}"] = X[:, index]
            frame[f"after_{column}"] = after[:, index]
            frame[f"delta_{column}"] = delta[:, index]
            frame[f"delta_comp_{column}"] = parts[0, :, index]
            frame[f"delta_compete_{column}"] = parts[1, :, index]
        for component in range(delta_pc.shape[1]):
            frame[f"PC{component + 1}_before"] = nominal["pc"][:, component]
            frame[f"PC{component + 1}_after"] = after_measure["pc"][:, component]
            frame[f"delta_PC{component + 1}"] = delta_pc[:, component]
        frames.append(frame)
        payload = {
            "ids": ids, "h": h, "reference": result["reference"],
            "decoded_aggregated": result["decoded_aggregated"],
            "decoded_reference": result["decoded_reference"], "coef": decoder.coef_,
            "intercept": decoder.intercept_, "alpha": decoder.alpha_,
            "decoder_fold_alphas": result["decoder_fold_alphas"], "delta": delta,
            "delta_comp": parts[0], "delta_compete": parts[1], "oof": oof,
        }
        if args.reference_method == "extra_trees":
            assert reference_model is not None
            payload.update({
                "reference_feature_names": np.asarray(reference_feature_names),
                "reference_feature_importance": reference_model.feature_importances_,
                "reference_query_features": result["reference_query_features"],
            })
        else:
            payload.update({"neighbor_index": result["neighbor_index"],
                            "neighbor_distance": result["neighbor_distance"]})
        np.savez_compressed(output / f"seed{seed}_decoder.npz", **payload)
        input_hashes[f"seed{seed}_arrays"] = sha256(array_path)

    nodes = pd.concat(frames, ignore_index=True)
    nodes.to_csv(output / "node_supply_change.csv", index=False)
    pd.DataFrame(performance).to_csv(output / "decoder_performance.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "decoder_summary.csv", index=False)
    if scope_summary is not None:
        scope_summary.to_csv(output / "configuration_service_range_reference.csv", index=False)
    mean = nodes.groupby("community_id", sort=False).agg(
        baseline_type=("baseline_type", "first"), after_type_mode=("after_type", lambda x: int(x.mode().iloc[0])),
        retained_rate=("retained", "mean"), change_9_mean=("change_9", "mean"),
        change_9_sd=("change_9", "std"), change_pc_mean=("change_pc", "mean"),
        change_pc_sd=("change_pc", "std"),
    ).reset_index()
    mean.to_csv(output / "node_seed_mean.csv", index=False)

    if args.reference_method == "type_knn_median":
        reference_protocol = {
            "typical_external_reference": (
                "coordinate-wise median external aggregation of the 20 nearest training-fold nodes "
                "within the same baseline configuration"
            ),
            "reference_model_parameters": {
                "neighbors": KNN_NEIGHBORS,
                "distance": "Euclidean distance after training-fold StandardScaler on h_base",
                "aggregation": "coordinate-wise median of h_comp and h_compete",
            },
            "reference_features": "h_base for neighbor matching; baseline configuration as an exact stratum",
            "change": (
                "D^(-fold)(h_base,h_ext) - "
                "D^(-fold)(h_base,median_20NN^(-fold)[h_ext|h_base,T])"
            ),
            "mobility_scope": (
                "external aggregation shared by nearby h_base peers of the same configuration is treated as typical; "
                "facility level is not identified"
            ),
            "figure_reference_label": "same-type 20-NN median",
        }
    else:
        reference_protocol = {
            "typical_external_reference": (
                "spatial-fold cross-fitted ExtraTrees conditional mean E[(h_comp,h_compete) | "
                "h_base, fold-and-configuration-specific typical service range, "
                "flow-weighted TH/SH destination context, baseline configuration]"
            ),
            "configuration_specific_scope": (
                "for each held-out node, each of five service-range summaries is replaced by the median "
                "among training-fold nodes with the same baseline configuration"
            ),
            "configuration_scope_table": "configuration_service_range_reference.csv reports training-fold quartiles",
            "reference_model_parameters": {
                "n_estimators": REFERENCE_TREES,
                "min_samples_leaf": REFERENCE_MIN_SAMPLES_LEAF,
                "max_features": REFERENCE_MAX_FEATURES,
                "random_state_base": REFERENCE_RANDOM_STATE,
            },
            "reference_features": reference_feature_names,
            "change": (
                "D^(-fold)(h_base,h_ext) - D^(-fold)(h_base,"
                "E^(-fold)[h_ext|h_base,S_typical^(-fold)(T),P_visible X_hospital,T])"
            ),
            "mobility_scope": (
                "the reference explicitly absorbs repeatable mobility associated with visible service-range structure "
                "within each baseline configuration and TH/SH destination context; CHC destination indicators are excluded"
            ),
            "figure_reference_label": "type-specific hospital-range reference",
        }
    protocol = {
        "variant": VARIANT,
        "seeds": args.seeds,
        "reference_method": args.reference_method,
        "representation": "saved three-channel aggregated node vector h",
        "decoder": "spatial-fold cross-fitted RidgeCV from aggregated h to nine frozen-ruler standardized indicators",
        "alphas": ALPHAS.tolist(),
        **reference_protocol,
        "after": "frozen baseline indicators + relative decoded shift",
        "validation": "both decoder and typical-reference model evaluated out of spatial fold",
        "facility_level_limit": "aggregate OD does not identify visited facility level; the residual is not uniquely community-health-centre flow",
        "interpretation": "relative supply-representation shift from atypical external aggregation, not measured absolute supply change",
        "input_sha256": input_hashes,
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "verification.json", {
        "fixed_node_order": True,
        "three_equal_channel_blocks": True,
        "same_fold_decoder_for_aggregated_and_typical_reference": True,
        "typical_reference_is_out_of_fold": True,
        "reference_neighbors_are_same_type_and_training_fold_only": args.reference_method == "type_knn_median",
        "reference_uses_only_nonheldout_od_edges": args.reference_method == "extra_trees",
        "service_range_is_configuration_specific_and_training_fold_only": args.reference_method == "extra_trees",
        "reference_includes_hospital_but_not_chc_destination_context": args.reference_method == "extra_trees",
        "decoder_is_out_of_fold": True,
        "intercept_cancels": True,
        "external_channel_parts_add_exactly": True,
        "frozen_ruler_reused": True,
    })
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()

"""Reconstruct model-implied supply change from ``mech_no_pair`` node vectors.

The saved ``arrays.npz`` file contains the final three-channel node
representation ``h = [h_base, h_comp, h_compete]``.  A Ridge decoder is fitted
from that aggregated representation to the nine frozen-ruler supply
coordinates.  The local reference is the same representation with the two
external channel blocks set to zero.  The model-implied change is therefore

    delta_x = D(h_base, h_comp, h_compete) - D(h_base, 0, 0).

For a linear decoder the compensation and competition contributions add
exactly.  The decoded difference is a representation-level intervention; it is
not an independently observed post-interaction supply outcome.

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
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
sys.path.insert(0, str(COMMON))
from decode import ALPHAS  # noqa: E402
from ruler import measure  # noqa: E402

VARIANT = "mech_no_pair"
CHANNELS = ("base", "comp", "compete")


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


def local_reference(h: np.ndarray, width: int) -> np.ndarray:
    local = np.asarray(h, dtype=float).copy()
    local[:, width:] = 0.0
    return local


def linear_channel_change(coef: np.ndarray, h: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Return total external change and the two additive channel parts."""
    coef = np.asarray(coef, dtype=float)
    h = np.asarray(h, dtype=float)
    parts = np.stack([
        h[:, width:2 * width] @ coef[:, width:2 * width].T,
        h[:, 2 * width:3 * width] @ coef[:, 2 * width:3 * width].T,
    ])
    return parts.sum(axis=0), parts


def spatial_oof(h: np.ndarray, target: np.ndarray, folds: np.ndarray) -> np.ndarray:
    prediction = np.empty_like(target, dtype=float)
    for fold in np.unique(folds):
        test = folds == fold
        prediction[test] = RidgeCV(alphas=ALPHAS).fit(h[~test], target[~test]).predict(h[test])
    return prediction


def reconstruct_seed(h: np.ndarray, target: np.ndarray, folds: np.ndarray) -> dict[str, object]:
    h, width = split_channels(h)
    local = local_reference(h, width)
    decoder = RidgeCV(alphas=ALPHAS).fit(h, target)
    decoded_aggregated = decoder.predict(h)
    decoded_local = decoder.predict(local)
    delta = decoded_aggregated - decoded_local
    linear_delta, parts = linear_channel_change(decoder.coef_, h, width)
    np.testing.assert_allclose(delta, linear_delta, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(parts.sum(axis=0), delta, rtol=1e-10, atol=1e-10)
    oof = spatial_oof(h, target, folds)
    return {
        "decoder": decoder,
        "width": width,
        "local": local,
        "decoded_aggregated": decoded_aggregated,
        "decoded_local": decoded_local,
        "delta": delta,
        "parts": parts,
        "oof": oof,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-results", type=Path, required=True)
    parser.add_argument("--baseline-table", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--ruler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    split_path = args.mechanism_results / "fixed_split.npz"
    with np.load(split_path) as split:
        ids = split["ids"].astype(int)
        W = split["W"].astype(float)
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
    input_hashes = {"fixed_split": sha256(split_path), "baseline_table": sha256(args.baseline_table),
                    "folds": sha256(args.folds), "ruler": sha256(args.ruler)}

    for seed in args.seeds:
        array_path = args.mechanism_results / VARIANT / f"seed{seed}" / "arrays.npz"
        with np.load(array_path) as saved:
            h = saved["h"].astype(float)
        if h.shape[0] != len(ids):
            raise ValueError(f"seed {seed} representation does not match fixed node order")
        result = reconstruct_seed(h, X, folds)
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

        for index, column in enumerate(columns):
            error = oof[:, index] - X[:, index]
            performance.append({"seed": seed, "indicator": column, "alpha": float(decoder.alpha_),
                                "oof_rmse": float(np.sqrt(np.mean(error ** 2))),
                                "oof_mae": float(np.mean(np.abs(error))),
                                "oof_r2": float(r2_score(X[:, index], oof[:, index]))})
        eligible = W.sum(axis=1) > 0
        summaries.append({"seed": seed, "alpha": float(decoder.alpha_),
                          "oof_rmse": float(np.sqrt(np.mean((oof - X) ** 2))),
                          "oof_min_r2": float(min(r2_score(X[:, j], oof[:, j]) for j in range(X.shape[1]))),
                          "oof_type_agreement": float(np.mean(oof_measure["class"] == nominal["class"])),
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
        np.savez_compressed(output / f"seed{seed}_decoder.npz", ids=ids, h=h, local=result["local"],
                            coef=decoder.coef_, intercept=decoder.intercept_, alpha=decoder.alpha_,
                            delta=delta, delta_comp=parts[0], delta_compete=parts[1], oof=oof)
        input_hashes[f"seed{seed}_arrays"] = sha256(array_path)

    nodes = pd.concat(frames, ignore_index=True)
    nodes.to_csv(output / "node_supply_change.csv", index=False)
    pd.DataFrame(performance).to_csv(output / "decoder_performance.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "decoder_summary.csv", index=False)
    mean = nodes.groupby("community_id", sort=False).agg(
        baseline_type=("baseline_type", "first"), after_type_mode=("after_type", lambda x: int(x.mode().iloc[0])),
        retained_rate=("retained", "mean"), change_9_mean=("change_9", "mean"),
        change_9_sd=("change_9", "std"), change_pc_mean=("change_pc", "mean"),
        change_pc_sd=("change_pc", "std"),
    ).reset_index()
    mean.to_csv(output / "node_seed_mean.csv", index=False)

    protocol = {
        "variant": VARIANT,
        "seeds": args.seeds,
        "representation": "saved three-channel aggregated node vector h",
        "decoder": "RidgeCV from aggregated h to nine frozen-ruler standardized indicators",
        "alphas": ALPHAS.tolist(),
        "local_reference": "same h with compensation and competition blocks set to zero",
        "change": "D(h_base,h_comp,h_compete) - D(h_base,0,0)",
        "after": "frozen baseline indicators + decoded change",
        "validation": "spatial-fold out-of-fold reconstruction of baseline indicators",
        "interpretation": "model-implied representation change, not measured post-interaction supply",
        "input_sha256": input_hashes,
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "verification.json", {
        "fixed_node_order": True,
        "three_equal_channel_blocks": True,
        "same_decoder_for_aggregated_and_local_reference": True,
        "intercept_cancels": True,
        "external_channel_parts_add_exactly": True,
        "frozen_ruler_reused": True,
    })
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()

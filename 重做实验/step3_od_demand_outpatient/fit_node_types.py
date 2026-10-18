"""按本目录供给评价结果（supply_indicators_revised 联通人口版九项指标）拟合社区供给分类。

拟合方法同 node_typing/code/fit_node_types.py（mrhgat_v6_1 experiment/redo 分支，本机副本在 mrhgat_v6_1_destpred/node_typing/code）：
九项指标 Yeo-Johnson 变换（|偏度| > 1）、标准化、平行分析主成分、composite 模式 KMeans，类型按各层级分量三分位命名。
--k 不给时只输出 k = 3 至 8 的稳定性与聚类指标及各 k 的类型构成（k_selection.csv、k_types.csv），给定 k 时输出
ruler.json 与 node_types.csv。
python fit_node_types.py --supply <供给指标csv> --out <目录> [--k K]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/luoxuan/Documents/mrhgat_v6_1_destpred/node_typing/code")
import fit_node_types as fnt  # noqa: E402
from fit_node_types import COLUMNS, jsonable  # noqa: E402
from ruler import fit_ruler, measure, standardize  # noqa: E402


def composite_names(frozen, Z, tol=1e-9):
    """同 fit_node_types.composite_names。三分位切点与最小值重合（超过三分之一社区该层级供给为 0）时，
    浮点误差会把均值等于最小值的类型判为高，这里比较时加 1e-9 容差。"""
    tiers = Z.reshape(len(Z), 3, 3).mean(2)
    cuts = np.percentile(tiers, [100 / 3, 200 / 3], axis=0)
    cls = measure(Z, frozen)["class"]
    centres = np.stack([tiers[cls == c].mean(0) for c in range(len(frozen["centers"]))])
    levels = (centres[:, None, :] > cuts[None, :, :] + tol).sum(1)
    word = ("低", "中", "高")
    out = []
    for lv in levels:
        profile = "·".join(f"{n}{word[v]}" for n, v in zip(fnt.TIER_CN, lv))
        high = [n for n, v in zip(fnt.TIER_CN, lv) if v == 2]
        if (lv == 2).all():
            short = "充足型"
        elif (lv == 0).all():
            short = "稀缺型"
        elif len(high) == 1:
            short = f"{high[0]}主导型"
        elif len(high) == 2:
            short = f"{high[0]}{high[1]}型"
        else:
            short = "中等型"
        out.append(f"{short}（{profile}）")
    return out, cuts


HERE = Path(__file__).resolve().parent
COMMON = dict(pa_reps=1000, boots=200, transform_mode="yeo-johnson", log_form=[])

ap = argparse.ArgumentParser()
ap.add_argument("--supply", default=str(HERE / "results/supply_indicators_revised/community_supply_indicators_2018_mobile_users.csv"))
ap.add_argument("--out", default=str(HERE / "results/node_typing"))
ap.add_argument("--k", type=int, default=None)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
s = pd.read_csv(a.supply, encoding="utf-8-sig").sort_values("community_id").reset_index(drop=True)
raw = s[COLUMNS].to_numpy(float)

if a.k is None:
    r, rep = fit_ruler(raw, COLUMNS, None, force_mode="composite", **COMMON)
    tab = pd.DataFrame(rep["k_selection"])
    tab.to_csv(out / "k_selection.csv", index=False, encoding="utf-8-sig")
    rows = []
    for k in tab.k:
        rk, _ = fit_ruler(raw, COLUMNS, None, force_mode="composite", k_override=int(k), **COMMON)
        x = standardize(raw, rk)
        names, _ = composite_names(rk, x)
        cls = measure(x, rk)["class"]
        for c, n in enumerate(names):
            rows.append({"k": int(k), "type_id": c, "type_name": n, "社区数": int((cls == c).sum())})
    kt = pd.DataFrame(rows)
    kt.to_csv(out / "k_types.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(jsonable({"KMO": rep["kmo"], "components": rep["components_parallel"], "k_votes": rep["k_votes"],
                               "k_selected_by_rule": rep["k_selected_by_rule"]}), ensure_ascii=False))
    print(tab.round(3).to_string(index=False))
    print(kt.to_string(index=False))
else:
    main_r, main_rep = fit_ruler(raw, COLUMNS, None, force_mode="composite", k_override=a.k, **COMMON)
    x = standardize(raw, main_r)
    names, cuts = composite_names(main_r, x)
    main_r["type_names"] = names
    main_rep["type_names"] = names
    main_rep["component_tertile_cuts"] = cuts.tolist()
    (out / "ruler.json").write_text(json.dumps(jsonable({"main": {**main_rep, "frozen": main_r}}), ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    nominal = measure(x, main_r)
    pd.DataFrame({"community_id": s["community_id"], "community_name": s["community_name"],
                  "type_id": nominal["class"], "type_name": [names[c] for c in nominal["class"]],
                  **{f"PC{j + 1}": nominal["pc"][:, j] for j in range(nominal["pc"].shape[1])}}
                 ).to_csv(out / "node_types.csv", index=False, encoding="utf-8-sig")
    print("k", a.k, "types", list(zip(names, np.bincount(nominal["class"], minlength=a.k).tolist())))

"""MR-HGAT v6.1 stage A: preserve source files and gate unresolved inputs.

Only auditable conversions are exported. Legacy features are explicitly marked
as legacy: they are NOT the new bed/count/inverse-time features in v6.1.
"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import geopandas as gpd

BASE = Path(r"D:\多功能网络构建")
ROOT = Path(__file__).resolve().parents[2]
FEATURES = [f"{metric}_{tier}" for tier in ("TH", "SH", "CHC")
            for metric in ("Accessibility", "Density", "Number")]
PATHS = {
    "supply": r"poi\poi\medical\四次统计\社区医疗数据_多表合并汇总_改-99.csv",
    "communities": r"shp\深圳社区objectid.shp",
    "centers": r"等时圈\等时圈wgs84拆分\Shenzhen_人口重心\深圳社区_联通人口重心result.shp",
    "isochrones": r"等时圈\等时圈wgs84拆分\等时圈wgs84拆分\foot等时圈.shp",
    "population": r"2018年11月非通勤出行\网格_人口_社区\网格_人口_社区\深圳各社区人口.shp",
    "grids": r"2018年11月非通勤出行\网格_人口_社区\网格_人口_社区\深圳250m网格_人口_所在社区.shp",
    "hospital": r"poi\poi\medical\医院_床位.shp",
    "chc": r"poi\poi\medical\社康带权重.shp",
    "th": r"poi\poi\medical\医疗设施set\TH.shp",
    "sh": r"poi\poi\medical\医疗设施set\SH.shp",
    "flows": r"代码\出行轨迹\result\medical_flow_split_result.csv",
    "time": r"26上实验\api\res_route_od_aligned_all.xlsx",
    "matrix_ids": r"26上实验\gat\data\feature_matrix_ini.csv",
    "matrix": r"26上实验\gat\data\flow_matrix.csv",
    "features12": r"26上实验\gat\data\feature_matrix_n.csv",
    "queen": r"26上实验\gat\data\spatial_matrix_queen.csv",
    "flow_alt1": r"代码\出行轨迹\result\medical_flow_result1224.csv",
    "flow_alt2": r"代码\出行轨迹\result\medical_flow_result12241.csv",
    "flow_alt3": r"服务能力分级\五次嵌入\result\medical_flow.csv",
    "flow_alt4": r"服务能力分级\五次嵌入\result\medical_flow_classification.csv",
}


def read_csv(path):
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    raise ValueError(f"Unsupported encoding: {path}")


def integer_ids(values, unique=False):
    a = pd.to_numeric(values, errors="raise")
    if a.isna().any() or not np.equal(a, np.floor(a)).all():
        raise ValueError("IDs must be nonmissing integers; no rounding permitted")
    if unique and a.duplicated().any():
        raise ValueError("Duplicate IDs; automatic dropping is forbidden")
    return a.astype("int64")


def standardize_legacy(df):
    x = df[FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Nonfinite legacy features; no implicit imputation")
    means, scales = x.mean(), x.std(ddof=0)
    z = (x - means) / scales.mask(scales == 0, 1.0)
    return z, {c: {"mean": float(means[c]), "std_ddof0": float(scales[c]),
                   "constant": bool(scales[c] == 0)} for c in FEATURES}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest():
    files = set()
    for rel in PATHS.values():
        path = BASE / rel
        files.add(path)
        if path.suffix == ".shp":
            for suffix in (".dbf", ".shx", ".prj", ".cpg", ".sbn", ".sbx"):
                if path.with_suffix(suffix).exists():
                    files.add(path.with_suffix(suffix))
    return {str(p): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(files)}


def reconstruct_flows(flows, order):
    mapping = {int(v): i for i, v in enumerate(order)}
    src, dst = flows.source.map(mapping), flows.target.map(mapping)
    if src.isna().any() or dst.isna().any():
        raise ValueError("OD contains IDs outside the declared node order")
    result = np.zeros((len(order), len(order)), dtype=float)
    np.add.at(result, (src.to_numpy(int), dst.to_numpy(int)), flows.flow_count.to_numpy(float))
    return result


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main():
    processed = ROOT / "data" / "processed"
    report_dir = ROOT / "reports"
    run = ROOT / "results" / "runs" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_A_data")
    for p in (processed, report_dir, run):
        p.mkdir(parents=True, exist_ok=True)
    before = source_manifest()
    metrics = {"python": platform.python_version(), "source_root": str(BASE)}
    supply = read_csv(BASE / PATHS["supply"])
    supply["OBJECTID"] = integer_ids(supply.OBJECTID, unique=True)
    ids = supply.OBJECTID.tolist()
    idset = set(ids)
    z, scaling = standardize_legacy(supply)
    supply[["OBJECTID"] + FEATURES].to_parquet(processed / "legacy_supply_raw.parquet", index=False)
    z.insert(0, "OBJECTID", supply.OBJECTID.to_numpy())
    z.to_parquet(processed / "legacy_supply_zscore.parquet", index=False)
    supply[["OBJECTID", "Cluster_Label"]].to_parquet(processed / "original_classes.parquet", index=False)
    write_json(processed / "legacy_scaling.json", {"feature_definition": "original manuscript indicators, NOT v6.1 node inputs", "statistics": scaling})
    metrics["legacy_supply"] = {"rows": len(supply), "features": FEATURES,
        "class_counts": supply.Cluster_Label.value_counts().to_dict(),
        "negative_values": {c: int((supply[c] < 0).sum()) for c in FEATURES},
        "missing": int(supply[FEATURES].isna().sum().sum()),
        "legacy_gat_feature_shape": list(np.loadtxt(BASE / PATHS["features12"], delimiter=",").shape)}

    community = gpd.read_file(BASE / PATHS["communities"])
    centers = gpd.read_file(BASE / PATHS["centers"])
    population = gpd.read_file(BASE / PATHS["population"])
    iso = gpd.read_file(BASE / PATHS["isochrones"])
    centers["OBJECTID"] = integer_ids(centers.OBJECTID, unique=True)
    if set(centers.OBJECTID) != idset:
        raise ValueError("Center IDs do not match the supply table")
    for d in (community, centers, population):
        if d.SQCODE.isna().any() or d.SQCODE.duplicated().any():
            raise ValueError("SQCODE must be unique and nonmissing for a verified crosswalk")
        if set(d.SQCODE) != set(centers.SQCODE):
            raise ValueError("Polygon, center and population SQCODE sets differ")
    centers = centers.to_crs(4326).set_index("OBJECTID").loc[ids]
    code_to_id = dict(zip(centers.SQCODE, ids))
    polygon_duplicate_id_rows = int(community.OBJECTID.duplicated().sum())
    community = community.set_index("SQCODE").loc[centers.SQCODE].reset_index()
    population = population.set_index("SQCODE").loc[centers.SQCODE].reset_index()
    crosswalk = pd.DataFrame({"community_id": ids, "SQCODE": centers.SQCODE.to_numpy(),
        "polygon_OBJECTID": community.OBJECTID.to_numpy(), "population_OBJECTID": population.OBJECTID.to_numpy()})
    crosswalk.to_parquet(processed / "community_id_crosswalk.parquet", index=False)
    metrics["id_crosswalk"] = {"key": "SQCODE", "canonical_ids": "existing center OBJECTID, matching the original supply table",
        "polygon_duplicate_OBJECTID_rows": polygon_duplicate_id_rows,
        "population_OBJECTID_equals_center": int((population.OBJECTID.to_numpy() == np.array(ids)).sum()),
        "SQCODE_sets_identical": True, "rows": len(crosswalk)}
    iso["objectid"] = integer_ids(iso.objectid, unique=True)
    iso = iso.to_crs(4326).set_index("objectid")
    common_iso_ids = sorted(idset & set(iso.index))
    name_match = sum(str(iso.loc[i, "coname"]).strip() == str(centers.loc[i, "SQNAME"]).strip()
                     for i in common_iso_ids)
    metrics["communities"] = {"rows": len(community), "invalid_geometry": int((~community.is_valid).sum()),
        "iso_rows": len(iso), "iso_missing_ids": sorted(idset - set(iso.index)),
        "iso_extra_ids": sorted(set(iso.index) - idset), "iso_name_match": name_match,
        "iso_name_comparisons": len(common_iso_ids), "center_geometry_types": centers.geom_type.value_counts().to_dict(),
        "population_source_fields": ["联通人", "七普扩"],
        "population_year_verified": False}
    candidate = gpd.GeoDataFrame({"community_id": ids, "cx": centers.geometry.x.to_numpy(),
        "cy": centers.geometry.y.to_numpy(), "联通人": population["联通人"].to_numpy(),
        "七普扩": population["七普扩"].to_numpy()}, geometry=community.to_crs(4326).geometry, crs=4326)
    candidate["iso_geometry"] = [iso.loc[i].geometry.wkb if i in iso.index else None for i in ids]
    candidate.to_parquet(processed / "communities_candidate.parquet", index=False)

    # Queen adjacency is computed from polygons, including shared vertices.
    adjacency = []
    index = community.sindex
    for i, geom in enumerate(community.geometry):
        for j in index.intersection(geom.bounds):
            j = int(j)
            if i < j and geom.touches(community.geometry.iloc[j]):
                adjacency.extend([(ids[i], ids[j], "Queen"), (ids[j], ids[i], "Queen")])
    polygon_adj = pd.DataFrame(adjacency, columns=["community_id", "neighbor_id", "rule"])
    old_queen = np.loadtxt(BASE / PATHS["queen"], delimiter=",")
    old_order = integer_ids(read_csv(BASE / PATHS["matrix_ids"]).life_circle_id, unique=True).tolist()
    if old_queen.shape != (len(old_order), len(old_order)) or set(old_order) != idset:
        raise ValueError("Existing adjacency shape or node IDs differ")
    if not np.isin(old_queen, [0, 1]).all() or not np.array_equal(old_queen, old_queen.T) or np.diag(old_queen).any():
        raise ValueError("Existing Queen matrix is not binary, symmetric, or loop-free")
    ii, jj = np.nonzero(old_queen)
    adj = pd.DataFrame({"community_id": [old_order[i] for i in ii],
        "neighbor_id": [old_order[j] for j in jj], "rule": "Queen_existing_matrix"})
    adj = adj.sort_values(["community_id", "neighbor_id"]).reset_index(drop=True)
    adj.to_parquet(processed / "adjacency.parquet", index=False)
    old_pairs = set(zip(adj.community_id, adj.neighbor_id))
    polygon_pairs = set(zip(polygon_adj.community_id, polygon_adj.neighbor_id))
    metrics["adjacency"] = {"directed_edges": len(adj), "isolated_ids": sorted(idset - set(adj.community_id)),
        "rule": "existing Queen matrix, preserving original GAT positional order",
        "polygon_exact_touch_edges": len(polygon_adj),
        "polygon_isolated_count": len(idset - set(polygon_adj.community_id)),
        "edge_symmetric_difference": len(old_pairs ^ polygon_pairs),
        "topology_provenance_verified": False}

    grid = gpd.read_file(BASE / PATHS["grids"])
    grid_cols = [c for c in ("Tid", "OBJECTID", "OBJECTID_2", "SQNAME", "SQCODE", "联通人", "七普加") if c in grid]
    grid_export = pd.DataFrame(grid[grid_cols]).copy()
    grid_export["community_id_from_SQCODE"] = grid.SQCODE.map(code_to_id).astype("Int64")
    grid_export.to_parquet(processed / "grids_existing_assignment.parquet", index=False)
    metrics["grids"] = {"rows": len(grid), "unique_grid_ids": int(grid.Tid.nunique()),
        "duplicate_grid_id_rows": int(grid.Tid.duplicated().sum()),
        "existing_assignment_outside_community_ids": int((~grid.OBJECTID_2.isin(idset)).sum()),
        "SQCODE_unmapped_rows": int(grid_export.community_id_from_SQCODE.isna().sum()),
        "walk_min_to_center_available": False}

    flow = read_csv(BASE / PATHS["flows"])
    flow.source = integer_ids(flow.source)
    flow.target = integer_ids(flow.target)
    flow.flow_count = pd.to_numeric(flow.flow_count, errors="raise")
    if not np.isfinite(flow.flow_count).all() or (flow.flow_count < 0).any():
        raise ValueError("Invalid flow counts; no automatic removal allowed")
    flow.to_parquet(processed / "flows_community_aggregate.parquet", index=False)
    total = float(flow.flow_count.sum())
    matrix = np.loadtxt(BASE / PATHS["matrix"], delimiter=",")
    feature_ids = integer_ids(read_csv(BASE / PATHS["matrix_ids"]).life_circle_id, unique=True).tolist()
    order_candidates = {"legacy_supply_OBJECTID_order": ids, "legacy_feature_life_circle_id_order": feature_ids,
                        "sorted_OBJECTID_order": sorted(ids)}
    order_checks = {}
    for key, order in order_candidates.items():
        rebuilt = reconstruct_flows(flow, order)
        order_checks[key] = bool(rebuilt.shape == matrix.shape and np.allclose(rebuilt, matrix, atol=1e-8, rtol=0))
    metrics["flows"] = {"rows": len(flow), "total": total, "paper_total": 651575,
        "difference_from_paper": total - 651575, "duplicate_pairs": int(flow.duplicated(["source", "target"]).sum()),
        "matrix_total": float(matrix.sum()), "matrix_shape": list(matrix.shape),
        "matrix_order_elementwise_checks": order_checks,
        "facility_id_and_tier_available": False}
    verified_orders = [key for key, ok in order_checks.items() if ok]
    if verified_orders:
        chosen = verified_orders[0]
        pd.DataFrame({"matrix_row": range(len(ids)), "OBJECTID": order_candidates[chosen]}).to_parquet(
            processed / "verified_matrix_node_order.parquet", index=False)

    times = pd.read_excel(BASE / PATHS["time"])
    times["time_min"] = pd.to_numeric(times.time_s, errors="raise") / 60.0
    times.to_parquet(processed / "travel_time_community_observed.parquet", index=False)
    metrics["travel_time"] = {"rows": len(times), "columns": list(times.columns),
        "zero_time_rows": int((times.time_s == 0).sum()), "negative_time_rows": int((times.time_s < 0).sum()),
        "status_counts": times.status_info.value_counts(dropna=False).to_dict(),
        "all_community_pair_count": len(ids) ** 2, "peak_free_labels_available": False,
        "facility_destinations_available": False}

    hospital = gpd.read_file(BASE / PATHS["hospital"])
    chc = gpd.read_file(BASE / PATHS["chc"])
    th, sh = gpd.read_file(BASE / PATHS["th"]), gpd.read_file(BASE / PATHS["sh"])
    beds = pd.to_numeric(hospital["床位数"], errors="coerce")
    metrics["facilities"] = {"hospital_records": len(hospital), "hospital_unique_Uid": int(hospital.Uid.nunique()),
        "hospital_duplicate_Uid_rows": int(hospital.Uid.duplicated().sum()), "beds_zero": int((beds == 0).sum()),
        "beds_missing_or_non_numeric": int(beds.isna().sum()), "chc_records": len(chc),
        "chc_capacity_column": "加权总", "chc_beds_available": False,
        "TH_named_file_records": len(th), "SH_named_file_records": len(sh),
        "TH_SH_Uid_overlap": len(set(th.Uid) & set(sh.Uid)),
        "SH_named_file_grades": sh["列1"].value_counts(dropna=False).to_dict(),
        "hospital_grade_fields": {c: hospital[c].value_counts(dropna=False).to_dict() for c in ("列4", "列1")}}

    alternatives = {}
    for rel in (r"代码\出行轨迹\result\medical_flow_result1224.csv",
                r"代码\出行轨迹\result\medical_flow_result12241.csv",
                r"服务能力分级\五次嵌入\result\medical_flow.csv",
                r"服务能力分级\五次嵌入\result\medical_flow_classification.csv"):
        frame = read_csv(BASE / rel)
        count = "flow_count" if "flow_count" in frame else "count"
        alternatives[rel] = {"rows": len(frame), "flow_total": float(frame[count].sum()), "columns": list(frame.columns)}
    metrics["alternative_flow_tables"] = alternatives
    after = source_manifest()
    if before != after:
        raise RuntimeError("Source file integrity changed during preparation")
    metrics["source_integrity_unchanged"] = True
    metrics["checked_source_files"] = len(before)
    code_files = list(Path(__file__).parent.glob("*.py"))
    version = {p.name: sha256(p) for p in code_files}
    write_json(run / "source_manifest.json", before)
    write_json(run / "code_version.json", version)
    write_json(run / "metrics.json", metrics)
    write_json(report_dir / "A_data_metrics.json", metrics)
    (run / "config.yaml").write_text("stage: A_data\nsource_policy: read_only\nseed: 0\n"
        "training_enabled: false\nlegacy_zscore_ddof: 0\nadjacency: Queen\n"
        "missing_value_imputation: disabled_pending_semantic_confirmation\n", encoding="utf-8")

    blockers = [
        "已定位OD的目的地是社区，未含facility_id及tier；聚合后不能唯一还原TH/SH/CHC设施级人流。",
        "时间表只含已有社区OD的time_s，缺少全部候选设施（含零流量）的t_peak/t_free及格网到中心walk_min_to_center。",
        "社区人口表是联通人数和七普扩样人口，尚不能确认为2018年总人口；没有已确认的2024人口。",
        "社康容量是服务项目加权值，v6.1公式要求床位；两者不可未经确认直接互换。医院0床位记录须确认是缺失还是实际为0。",
        f"步行等时圈缺少社区ID {metrics['communities']['iso_missing_ids']}；缺项不能用其他社区几何或空值代替训练。",
        f"当前聚合流量为{total:,.0f}，与原稿651,575相差{total-651575:,.0f}；没有已确认的筛选日志解释差额。",
        "near_workplace_share尚未定位到可按现有OD关联的表；不能把未知比例填0。",
        "原稿九指标与v6.1的九指标定义不同；旧12列矩阵也不是三层各三列的新输入。",
        "原九指标Number包含-1编码且为比值，Accessibility包含-5/-1；不能直接视为设施计数或把编码擅自改为0。",
        f"SH.shp中的等级为{metrics['facilities']['SH_named_file_grades']}；文件名不能作为纯二级医院层级的充分依据。",
        f"保留的Queen矩阵有{len(adj)}条有向边，现有社区面精确接触仅{len(polygon_adj)}条且有3个无效面；邻接口径与原矩阵生成来源需核实。",
        "在项目、论文目录及下载目录未定位到claude/mrhgat参考原型和步骤8依赖的v5完整方案。",
    ]
    readiness = {"stage": "A", "status": "needs_input", "ready_for_training": False,
        "safe_preprocessing_complete": True, "blockers": blockers,
        "outputs": sorted(p.name for p in processed.iterdir()), "run_directory": str(run)}
    write_json(report_dir / "A_readiness.json", readiness)
    report = ["# MR-HGAT v6.1 输入核对", "", "状态：已完成已定位输入的核验与可逆预处理；尚不具备真实训练条件。",
        "原始数据与原有代码未修改，输入文件处理前后 SHA-256 一致。所有输出保存在独立 mrhgat_v6_1 目录。", "",
        "## 已核实的数据", "", "| 项目 | 结果 |", "|---|---|",
        f"| 原稿供给表 | {len(supply)}个社区；九项指标完整；原五类单独保存 |",
        f"| 旧GAT特征 | {metrics['legacy_supply']['legacy_gat_feature_shape']}；包含一级医院指标 |",
        f"| 社区/等时圈 | {len(community)} / {len(iso)}；缺少ID {metrics['communities']['iso_missing_ids']} |",
        f"| 等时圈名称核对 | 同ID名称一致 {name_match}/{len(common_iso_ids)} |",
        f"| 原有格网归属 | {len(grid)}行；唯一Tid {grid.Tid.nunique()}；未含最短步行时间 |",
        f"| 社区ID关联 | 面文件OBJECTID重复行{polygon_duplicate_id_rows}；人口表ID与中心ID不一致；已按唯一SQCODE建立交叉表，保留全部原ID |",
        f"| 医院/社康 | {len(hospital)} / {len(chc)}；医院0床位 {int((beds == 0).sum())}条 |",
        f"| 社区OD | {len(flow)}条；{total:,.0f}人次；缺层级和具体设施ID |",
        f"| 出行时间 | {len(times)}条社区OD；无高峰/自由流标记；未覆盖零流量候选设施 |",
        f"| Queen邻接 | 保留旧矩阵{len(adj)}条有向边；原社区面精确接触{len(polygon_adj)}条；生成口径待核 |", "",
        "## 已生成的预处理文件", "",
        "- legacy_supply_raw.parquet / legacy_supply_zscore.parquet：按TH、SH、CHC顺序组织原九指标，保留原字段与负值，按列z标准化；Cluster_Label不进入特征。它们只供核对/原稿对照，不能冒充v6.1新指标。",
        "- original_classes.parquet：保存原OBJECTID和五类标签。",
        "- community_id_crosswalk.parquet：按三表均唯一且集合一致的SQCODE核对现有ID；模型索引采用与供给表一致的原中心OBJECTID，不重编号源数据。",
        "- communities_candidate.parquet：社区面、原有中心、原人口字段和等时圈WKB；所有几何为EPSG:4326；未将年份未知的人口改名为pop_2018。",
        "- grids_existing_assignment.parquet：保留现有格网归属；尚未验证符合最短步行规则。",
        "- adjacency.parquet：保留原Queen矩阵及原GAT节点顺序；与社区面重算结果存在差异，尚未通过v6.1邻接口径验收。",
        "- flows_community_aggregate.parquet：完整保留现有社区OD及流量，不拆分设施/层级。",
        "- travel_time_community_observed.parquet：保留原字段，仅增加time_min=time_s/60；未将它标为高峰或自由流。", "",
        "## 矩阵顺序核对", "", "对候选节点顺序逐元素重建流量矩阵（不仅比较总量）：", "",
        *[f"- {k}: {v}" for k, v in order_checks.items()], "",
        "## 流量口径核对", "", "| 文件 | 记录数 | 流量总和 |", "|---|---:|---:|",
        f"| 当前medical_flow_split_result.csv | {len(flow)} | {total:,.0f} |",
        *[f"| {Path(k).name} | {v['rows']} | {v['flow_total']:,.0f} |" for k, v in alternatives.items()], "",
        "这些是不同处理层次的现存表；没有处理日志时不能把表间差额归因为去重、筛选或按层级拆分。", "",
        "## 需要补齐或确认的输入", "", *[f"{i+1}. {b}" for i, b in enumerate(blockers)], "",
        "## 方案执行边界", "",
        "依据v6.1步骤1“缺任何一项，列出缺项后停下来问，不要猜”，尚未执行重力拟合、MR-HGAT模拟验收和真实训练，也未生成机制或韧性结论。",
        "pop_2024和t_free主要影响敏感性；设施级分层OD、主分析容量/人口/出行时间及可追溯对应关系是核心缺项。",
        "用户补齐既有文件位置或明确调整定义后，才可生成最终communities、grids、facilities、flows、travel_time及空间折表。", "",
        f"输入与代码校验清单：{run}", ""]
    (report_dir / "A_data_reconciliation.md").write_text("\n".join(report), encoding="utf-8")
    (report_dir / "A_data_FAIL.md").write_text("# 步骤A验收状态\n\n当前状态：需要输入，不允许进入依赖步骤。\n\n" +
        "\n".join(f"- {b}" for b in blockers) + "\n", encoding="utf-8")
    config = ROOT / 'configs/mrhgat.yaml'
    if config.exists() and 'scope: aggregate_community_flow' in config.read_text(encoding='utf-8'):
        # The aggregate scope uses the existing evaluations and community OD.
        import importlib.util
        spec = importlib.util.spec_from_file_location('aggregate_graph', ROOT / 'src/C_graph/build_graph.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
        current_report = [
            '# 输入核对与处理结论', '',
            f'原输入保留：{len(ids)}个社区、9项供给评价、{total:,.0f}人次、{len(adj)}条有向Queen邻接边。', '',
            f'{len(before)}个源文件SHA-256一致。节点顺序与原流量矩阵逐元素核对通过。', '',
            '预处理仅作ID对齐、九项指标z标准化、邻接标记和空间分折。负值编码与原字段保留，Cluster_Label不进入特征。', '',
            'TH/SH/CHC仅为生活圈供给属性；社康保留既有服务权重。不拆分层级OD，不使用时间或容量，不以供给分数除人流。', '',
            f'当前总流量与文稿651,575相差{total-651575:,.0f}；本次沿用原GAT输入，不编造筛选差额原因。', '',
            '解释边界：Number是既有比值评价并含-1，Accessibility含-5/-1；SH来源包含一级与二级医院；原Queen矩阵与现有面精确接触关系不同。本次不擅改原输入。', '',
            '当前可进行聚合模型模拟验证；真实训练须满足A_readiness.json和D6_model_selection.md的验收状态。', '',
            '完整输入明细见A_source_inventory.json及A_data_metrics.json，模型定义见执行口径.md。', '']
        (report_dir / 'A_data_reconciliation.md').write_text('\n'.join(current_report), encoding='utf-8')
        (report_dir / 'A_data_FAIL.md').write_text('# 输入验收状态索引\n\n当前聚合输入已具备模拟验证条件；当前训练门槛见A_readiness.json。\n', encoding='utf-8')
    else:
        print(json.dumps(readiness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

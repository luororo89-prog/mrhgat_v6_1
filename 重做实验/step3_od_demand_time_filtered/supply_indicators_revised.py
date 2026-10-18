"""计算行政社区生活圈的分层医疗设施供给指标（重做实验，按时间筛选门诊 OD 的设施与网格归属修订重算）。

复制 step2_supply/calculate_supply_indicators_redo.py，只改输入与输出：需求人口取 facility_grid250.py 修订后的
results/facility_grid250/grid250_demand_allocation_revised.csv，社康坐标取 results/facility_grid250/facility_location_final.csv
（2018年深圳CHC_912个点位_XY_WGS84.xlsx 的新点位），结果写入 results/supply_indicators_revised/。

评价思路沿用 2supply_indicators/code/calculate_supply_indicators_2018.py，改动只在输入：
1. 需求人口：250 m 网格人口按 step1 定稿规则归属社区（无争议网格按 maup 整格归属，有争议网格按城市建设用地面积比例分配），
   取 step1_grid_community/results/landuse_demand/grid250_demand_allocation.csv；网格仍取中心点。
2. 设施：社康与医院名单、服务能力和床位取 supply input/facility_supply_audit_2018_mobile_users.csv（912 个社康、17 家二级、23 家三级）。
   医院位置取 supply input/hospital_aoi_35.shp（40 个 AOI）：生活圈归属按 AOI 与等时圈相交判定，路径计算取 AOI 代表点。
   社康仍取点位。
3. SA 中医院的服务范围：二级医院 25 分钟驾车，三级医院 30 分钟驾车。
4. 社康坐标全部取 facility_location_final.csv 中的新点位。
设施所在社区见 facility_community.py，只作为归属表输出，不参与本脚本的生活圈判定。
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from heapq import heappop, heappush
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "supply input"
INPUT_SOURCES = HERE.parents[1] / "1input_sources"
ROAD_DIR = Path("/Users/luoxuan/Documents/博后三胎/供给评价代码与数据/outputs/道路预处理/geofabrik_2019_01_01")
RESULT_DIR = HERE / "results" / "supply_indicators_revised"

COMMUNITY_PATH = SRC / "深圳社区objectid.shp"
CENTER_PATH = INPUT_SOURCES / "等时圈" / "等时圈wgs84拆分" / "Shenzhen_人口重心" / "深圳社区_联通人口重心result.shp"
ISOCHRONE_PATH = INPUT_SOURCES / "等时圈" / "等时圈wgs84拆分" / "等时圈wgs84拆分" / "foot等时圈.shp"
POPULATION_GRID_PATH = SRC / "深圳250m网格_人口_所在社区.shp"
DEMAND_ALLOCATION_PATH = HERE / "results" / "facility_grid250" / "grid250_demand_allocation_revised.csv"
ROAD_PATH = ROAD_DIR / "shenzhen_osm_roads_190101.gpkg"
ROAD_META_PATH = ROAD_DIR / "shenzhen_osm_roads_190101.json"
FACILITY_PATH = SRC / "facility_supply_audit_2018_mobile_users.csv"
HOSPITAL_AOI_PATH = SRC / "hospital_aoi_35.shp"
CORRECTION_PATH = HERE / "results" / "facility_grid250" / "facility_location_final.csv"

PROJECTED_CRS = "EPSG:32650"
LAMBDA = 0.5
WALK_M_PER_MIN = 80.0
CYCLE_M_PER_MIN = 150.0
DRIVE_CONNECTOR_M_PER_MIN = 500.0
BRIDGE_TOLERANCE_M = 10.0
GRID_ASSIGNMENT_MAX_M = 500.0

TIER_ORDER = ["CHC", "SH", "TH"]
H2_CATCHMENT_MIN = {"CHC": 15.0, "SH": 25.0, "TH": 30.0}
TIER_MODE = {"CHC": "walk", "SH": "walk", "TH": "walk"}
H2_MODE = {"CHC": "walk", "SH": "drive", "TH": "drive"}
TRAVEL_BANDS = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]
TRAVEL_WEIGHTS = np.array([0.5, 0.3, 0.2], dtype=float)
SPEED_CLASS_KMH = {1: 130.0, 2: 110.0, 3: 95.0, 4: 80.0, 5: 60.0, 6: 40.0, 7: 20.0, 8: 10.0}


def normalize_code(values: pd.Series) -> pd.Series:
    return values.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()


def load_population_grids(centers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One row per grid-community piece; split grids share the grid geometry and centre point."""
    grids = gpd.read_file(POPULATION_GRID_PATH)[["Tid", "联通人", "七普加", "geometry"]].to_crs(4326)
    if grids["Tid"].isna().any() or grids["Tid"].duplicated().any():
        raise ValueError("250m population grid Tid must be complete and unique")
    allocation = pd.read_csv(DEMAND_ALLOCATION_PATH, dtype={"SQCODE": str})
    if not set(allocation["Tid"]) == set(grids["Tid"]):
        raise ValueError("Demand allocation must cover every 250m grid")
    allocated = allocation.groupby("Tid")[["联通人_分配", "七普加_分配"]].sum()
    for field in ("联通人", "七普加"):
        if not np.allclose(grids.set_index("Tid")[field].astype(float), allocated[f"{field}_分配"].reindex(grids["Tid"])):
            raise ValueError(f"Allocated {field} must sum to the grid population")
    grids = grids[["Tid", "geometry"]].merge(
        allocation[["Tid", "SQCODE", "联通人_分配", "七普加_分配"]].rename(
            columns={"联通人_分配": "联通人", "七普加_分配": "七普加"}),
        on="Tid", how="inner", validate="one_to_many",
    )
    grids["SQCODE"] = normalize_code(grids["SQCODE"])
    grids["community_index"] = grids["SQCODE"].map(
        dict(zip(centers["SQCODE"], centers["community_index"]))
    )
    for field in ("联通人", "七普加"):
        grids[field] = pd.to_numeric(grids[field], errors="raise").astype(float)
        if grids[field].isna().any() or (grids[field] < 0).any():
            raise ValueError(f"250m population field {field} must be nonnegative and complete")
    grids = grids.loc[(grids["联通人"] > 0) | (grids["七普加"] > 0)].copy()
    unmatched = grids["community_index"].isna()
    excluded = {
        "grid_count": int(unmatched.sum()),
        "mobile_users": float(grids.loc[unmatched, "联通人"].sum()),
        "census_expanded": float(grids.loc[unmatched, "七普加"].sum()),
    }
    if unmatched.any():
        print(
            f"Excluded {int(unmatched.sum())} positive-population 250m grid(s) "
            f"without modeled community: "
            f"{grids.loc[unmatched, ['联通人', '七普加']].sum().to_dict()}"
        )
        grids = grids.loc[~unmatched].copy()
    grids["community_index"] = grids["community_index"].astype(int)
    grids = grids.reset_index(drop=True)
    grids.attrs["excluded_without_community"] = excluded
    return grids


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan")
    order = np.argsort(values[valid], kind="mergesort")
    x = values[valid][order]
    w = weights[valid][order]
    cutoff = quantile * w.sum()
    return float(x[min(np.searchsorted(np.cumsum(w), cutoff, side="left"), len(x) - 1)])


def gaussian_decay(times: np.ndarray, threshold: float) -> np.ndarray:
    base = math.exp(-0.5)
    result = (np.exp(-0.5 * np.square(times / threshold)) - base) / (1.0 - base)
    return np.clip(result, 0.0, 1.0)


def add_undirected(adj: list[list[tuple[int, float]]], u: int, v: int, weight: float) -> None:
    if u == v or weight <= 0:
        return
    adj[u].append((v, float(weight)))
    adj[v].append((u, float(weight)))


def dijkstra(
    adj: list[list[tuple[int, float]]],
    sources: tuple[tuple[int, float], tuple[int, float]],
    cutoff: float,
) -> dict[int, float]:
    dist: dict[int, float] = {}
    heap: list[tuple[float, int]] = []
    for node, initial in sources:
        if initial <= cutoff and initial < dist.get(node, math.inf):
            dist[node] = float(initial)
            heappush(heap, (float(initial), int(node)))
    while heap:
        current, node = heappop(heap)
        if current != dist.get(node) or current > cutoff:
            continue
        for neighbor, edge_cost in adj[node]:
            candidate = current + edge_cost
            if candidate <= cutoff and candidate < dist.get(neighbor, math.inf):
                dist[neighbor] = candidate
                heappush(heap, (candidate, neighbor))
    return dist


def route_values(
    dist: dict[int, float],
    snap: dict[str, np.ndarray],
    indices: np.ndarray,
    source_side: bool = False,
    source_snap: dict[str, np.ndarray] | None = None,
    source_index: int | None = None,
    mode: str | None = None,
) -> np.ndarray:
    result = np.full(len(indices), np.inf, dtype=float)
    u = snap["u"][indices]
    v = snap["v"][indices]
    prefix = "source_cost_" if source_side else "cost_"
    cu = snap[f"{prefix}u"][indices]
    cv = snap[f"{prefix}v"][indices]
    for pos in range(len(indices)):
        du = dist.get(int(u[pos]), math.inf) + float(cu[pos])
        dv = dist.get(int(v[pos]), math.inf) + float(cv[pos])
        result[pos] = min(du, dv)
    if source_snap is not None:
        if source_index is None or mode not in {"walk", "cycle", "drive"}:
            raise ValueError("Same-edge routing requires source index and travel mode")
        same = snap["matched_edge"][indices] == source_snap["matched_edge"][source_index]
        if same.any():
            target_along = snap["along_m"][indices[same]]
            source_along = float(source_snap["along_m"][source_index])
            along_difference = np.abs(target_along - source_along)
            connector = (
                float(source_snap["connector_m"][source_index])
                + snap["connector_m"][indices[same]]
            )
            if mode == "drive":
                direction = int(source_snap["edge_direction"][source_index])
                origin_along = target_along if source_side else source_along
                destination_along = source_along if source_side else target_along
                allowed = ((direction == 0)
                           | ((direction > 0) & (destination_along >= origin_along))
                           | ((direction < 0) & (destination_along <= origin_along)))
                direct = connector / DRIVE_CONNECTOR_M_PER_MIN + (
                    along_difference / float(source_snap["edge_length_m"][source_index])
                    * float(source_snap["edge_drive_min"][source_index])
                )
                direct = np.where(allowed, direct, math.inf)
            else:
                direct = connector + along_difference
            result[same] = np.minimum(result[same], direct)
    return result


def build_facilities() -> tuple[gpd.GeoDataFrame, gpd.GeoSeries]:
    """Return facilities located at CHC points / AOI representative points, and their footprints
    (CHC point, hospital AOI polygon) in the projected CRS for life-circle membership."""
    table = pd.read_csv(FACILITY_PATH, encoding="utf-8-sig").sort_values("facility_index")
    moved = pd.read_csv(CORRECTION_PATH, encoding="utf-8-sig").dropna(subset=["longitude"]).set_index("facility_id")
    for col in ("longitude", "latitude"):
        table.loc[table["facility_id"].isin(moved.index), col] = table["facility_id"].map(moved[col])
    facilities = gpd.GeoDataFrame(
        table[["facility_id", "facility_name", "tier", "source_grade", "capacity", "service_items"]].reset_index(drop=True),
        geometry=gpd.points_from_xy(table["longitude"], table["latitude"]),
        crs=4326,
    )
    facilities["capacity"] = pd.to_numeric(facilities["capacity"], errors="raise").astype(float)
    if not facilities["tier"].isin(TIER_ORDER).all() or facilities["facility_id"].duplicated().any():
        raise ValueError("Facility table must have unique ids and known tiers")

    aoi = gpd.read_file(HOSPITAL_AOI_PATH).to_crs(PROJECTED_CRS)
    aoi["geometry"] = shapely.make_valid(aoi.geometry.to_numpy())
    hospital = facilities["tier"] != "CHC"
    if aoi["fac_name"].duplicated().any() or set(aoi["fac_name"]) != set(facilities.loc[hospital, "facility_name"]):
        raise ValueError("Every secondary and tertiary hospital must have exactly one AOI")
    aoi_by_name = aoi.set_index("fac_name").geometry
    footprints = facilities.geometry.to_crs(PROJECTED_CRS).copy()
    footprints.loc[hospital] = facilities.loc[hospital, "facility_name"].map(aoi_by_name).to_numpy()
    representative = gpd.GeoSeries(
        shapely.point_on_surface(footprints.loc[hospital].to_numpy()), crs=PROJECTED_CRS
    ).to_crs(4326)
    facilities.loc[hospital, "geometry"] = representative.to_numpy()
    facilities["location"] = np.where(hospital, "AOI 代表点", "社康点位")
    facilities["facility_index"] = np.arange(len(facilities), dtype=int)

    medians: dict[str, float] = {}
    for tier in TIER_ORDER:
        positive = facilities.loc[(facilities["tier"] == tier) & (facilities["capacity"] > 0), "capacity"]
        medians[tier] = float(positive.median())
    facilities["capacity_median_tier"] = facilities["tier"].map(medians)
    facilities["resource_weight"] = (1.0 - LAMBDA) + LAMBDA * (
        facilities["capacity"] / facilities["capacity_median_tier"]
    )
    return facilities, footprints


def build_network(roads: gpd.GeoDataFrame) -> dict:
    if {"u_osm", "v_osm", "walk_allowed", "cycle_allowed", "drive_allowed"}.issubset(roads):
        return build_osm_network(roads)
    node_lookup: dict[tuple[float, float], int] = {}
    node_xy: list[tuple[float, float]] = []
    node_nonexpress: list[bool] = []
    edge_records: list[dict] = []

    def node_id(coordinate, nonexpress: bool) -> int:
        key = (round(float(coordinate[0]), 2), round(float(coordinate[1]), 2))
        if key not in node_lookup:
            node_lookup[key] = len(node_xy)
            node_xy.append(key)
            node_nonexpress.append(False)
        idx = node_lookup[key]
        if nonexpress:
            node_nonexpress[idx] = True
        return idx

    for road_idx, row in roads.iterrows():
        geom = row.geometry
        coordinates = list(geom.coords)
        nonexpress = str(row["src_type"]) != "express"
        u = node_id(coordinates[0], nonexpress)
        v = node_id(coordinates[-1], nonexpress)
        length_m = float(geom.length)
        speed_class = int(row["speed_cls"]) if pd.notna(row["speed_cls"]) else 7
        speed_kmh = SPEED_CLASS_KMH.get(speed_class, 20.0)
        drive_min = length_m / (speed_kmh * 1000.0 / 60.0)
        edge_records.append(
            {
                "road_index": int(road_idx),
                "u": u,
                "v": v,
                "length_m": length_m,
                "drive_min": drive_min,
                "nonexpress": nonexpress,
                "geometry": geom,
            }
        )

    n_nodes = len(node_xy)
    distance_adj: list[list[tuple[int, float]]] = [[] for _ in range(n_nodes)]
    drive_adj: list[list[tuple[int, float]]] = [[] for _ in range(n_nodes)]
    for edge in edge_records:
        add_undirected(drive_adj, edge["u"], edge["v"], edge["drive_min"])
        if edge["nonexpress"]:
            add_undirected(distance_adj, edge["u"], edge["v"], edge["length_m"])

    bridge_count = 0
    cell = BRIDGE_TOLERANCE_M
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (x, y) in enumerate(node_xy):
        cx, cy = int(math.floor(x / cell)), int(math.floor(y / cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((cx + dx, cy + dy), []):
                    x2, y2 = node_xy[j]
                    distance = math.hypot(x - x2, y - y2)
                    if 0 < distance <= BRIDGE_TOLERANCE_M:
                        add_undirected(drive_adj, i, j, distance / DRIVE_CONNECTOR_M_PER_MIN)
                        if node_nonexpress[i] and node_nonexpress[j]:
                            add_undirected(distance_adj, i, j, distance)
                        bridge_count += 1
        buckets[(cx, cy)].append(i)

    all_edges = pd.DataFrame(edge_records)
    nonexpress_edges = all_edges.loc[all_edges["nonexpress"]].reset_index(drop=True)
    return {
        "node_xy": np.asarray(node_xy, dtype=float),
        "distance_adj": distance_adj,
        "drive_adj": drive_adj,
        "all_edges": all_edges,
        "nonexpress_edges": nonexpress_edges,
        "bridge_count": bridge_count,
    }


def build_osm_network(roads: gpd.GeoDataFrame) -> dict:
    """Use shared OSM node identifiers as road junctions."""
    node_lookup: dict[int, int] = {}
    node_xy: list[tuple[float, float]] = []
    records: list[dict] = []

    def node_id(osm_id: int, coordinate) -> int:
        if osm_id not in node_lookup:
            node_lookup[osm_id] = len(node_xy)
            node_xy.append((float(coordinate[0]), float(coordinate[1])))
        return node_lookup[osm_id]

    for road_idx, row in roads.iterrows():
        geom = row.geometry
        coords = list(geom.coords)
        if len(coords) != 2:
            raise ValueError("OSM road segments must contain exactly two OSM nodes")
        u = node_id(int(row["u_osm"]), coords[0])
        v = node_id(int(row["v_osm"]), coords[1])
        length_m = float(geom.length)
        speed_kmh = float(row["speed_kmh"])
        records.append({
            "road_index": int(road_idx),
            "u": u,
            "v": v,
            "length_m": length_m,
            "drive_min": length_m / (speed_kmh * 1000.0 / 60.0),
            "drive_direction": int(row["drive_direction"]),
            "walk_allowed": bool(row["walk_allowed"]),
            "cycle_allowed": bool(row["cycle_allowed"]),
            "drive_allowed": bool(row["drive_allowed"]),
            "geometry": geom,
        })

    distance_adj: list[list[tuple[int, float]]] = [[] for _ in node_xy]
    cycle_adj: list[list[tuple[int, float]]] = [[] for _ in node_xy]
    drive_adj: list[list[tuple[int, float]]] = [[] for _ in node_xy]
    drive_reverse_adj: list[list[tuple[int, float]]] = [[] for _ in node_xy]
    for edge in records:
        u, v = edge["u"], edge["v"]
        if edge["walk_allowed"]:
            add_undirected(distance_adj, u, v, edge["length_m"])
        if edge["cycle_allowed"]:
            add_undirected(cycle_adj, u, v, edge["length_m"])
        if edge["drive_allowed"]:
            if edge["drive_direction"] >= 0:
                drive_adj[u].append((v, edge["drive_min"]))
                drive_reverse_adj[v].append((u, edge["drive_min"]))
            if edge["drive_direction"] <= 0:
                drive_adj[v].append((u, edge["drive_min"]))
                drive_reverse_adj[u].append((v, edge["drive_min"]))

    frame = pd.DataFrame(records)
    return {
        "node_xy": np.asarray(node_xy, dtype=float),
        "distance_adj": distance_adj,
        "cycle_adj": cycle_adj,
        "drive_adj": drive_adj,
        "drive_reverse_adj": drive_reverse_adj,
        "all_edges": frame.loc[frame["drive_allowed"]].reset_index(drop=True),
        "nonexpress_edges": frame.loc[frame["walk_allowed"]].reset_index(drop=True),
        "cycle_edges": frame.loc[frame["cycle_allowed"]].reset_index(drop=True),
        "bridge_count": 0,
    }


def snap_points(points, edges: pd.DataFrame, mode: str) -> dict[str, np.ndarray]:
    geometries = np.asarray(edges["geometry"].tolist(), dtype=object)
    tree = shapely.STRtree(geometries)
    point_array = np.asarray(list(points), dtype=object)
    pair_indices, distances = tree.query_nearest(
        point_array,
        all_matches=False,
        return_distance=True,
    )
    matched_edge = np.empty(len(point_array), dtype=int)
    connector_m = np.empty(len(point_array), dtype=float)
    matched_edge[pair_indices[0]] = pair_indices[1]
    connector_m[pair_indices[0]] = distances

    u = np.empty(len(point_array), dtype=int)
    v = np.empty(len(point_array), dtype=int)
    cost_u = np.empty(len(point_array), dtype=float)
    cost_v = np.empty(len(point_array), dtype=float)
    source_cost_u = np.empty(len(point_array), dtype=float)
    source_cost_v = np.empty(len(point_array), dtype=float)
    along_distances_m = np.empty(len(point_array), dtype=float)
    edge_lengths_m = np.empty(len(point_array), dtype=float)
    edge_drive_minutes = np.empty(len(point_array), dtype=float)
    edge_directions = np.empty(len(point_array), dtype=int)
    for i, edge_idx in enumerate(matched_edge):
        edge = edges.iloc[int(edge_idx)]
        geom = edge["geometry"]
        along_m = float(geom.project(point_array[i]))
        along_distances_m[i] = along_m
        edge_lengths_m[i] = float(edge["length_m"])
        edge_drive_minutes[i] = float(edge["drive_min"])
        edge_directions[i] = int(edge["drive_direction"]) if "drive_direction" in edge.index else 0
        remaining_m = max(float(edge["length_m"]) - along_m, 0.0)
        u[i], v[i] = int(edge["u"]), int(edge["v"])
        if mode == "drive":
            fraction = along_m / float(edge["length_m"]) if edge["length_m"] > 0 else 0.0
            connector_min = connector_m[i] / DRIVE_CONNECTOR_M_PER_MIN
            to_u = connector_min + float(edge["drive_min"]) * fraction
            to_v = connector_min + float(edge["drive_min"]) * (1.0 - fraction)
            direction = int(edge["drive_direction"]) if "drive_direction" in edge.index else 0
            cost_u[i] = to_u if direction >= 0 else math.inf
            cost_v[i] = to_v if direction <= 0 else math.inf
            source_cost_u[i] = to_u if direction <= 0 else math.inf
            source_cost_v[i] = to_v if direction >= 0 else math.inf
        else:
            cost_u[i] = connector_m[i] + along_m
            cost_v[i] = connector_m[i] + remaining_m
            source_cost_u[i] = cost_u[i]
            source_cost_v[i] = cost_v[i]
    return {
        "u": u,
        "v": v,
        "cost_u": cost_u,
        "cost_v": cost_v,
        "source_cost_u": source_cost_u,
        "source_cost_v": source_cost_v,
        "connector_m": connector_m,
        "matched_edge": matched_edge,
        "along_m": along_distances_m,
        "edge_length_m": edge_lengths_m,
        "edge_drive_min": edge_drive_minutes,
        "edge_direction": edge_directions,
    }


def assign_points_to_grids(
    points: gpd.GeoSeries, grid: gpd.GeoDataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Assign points to their 500 m cell, allowing a nearby boundary mismatch."""
    frame = gpd.GeoDataFrame(
        {"point_index": np.arange(len(points), dtype=int)},
        geometry=gpd.GeoSeries(points.to_numpy(), crs=grid.crs),
        crs=grid.crs,
    )
    joined = gpd.sjoin(frame, grid[["geometry"]], how="left", predicate="intersects")
    assigned = np.full(len(points), -1, dtype=int)
    matched = joined.loc[joined["index_right"].notna()].groupby("point_index")["index_right"].min()
    assigned[matched.index.to_numpy(dtype=int)] = matched.to_numpy(dtype=int)

    centers = np.column_stack((grid.geometry.centroid.x, grid.geometry.centroid.y))
    point_xy = np.column_stack((points.x, points.y))
    nearest_m, nearest_index = cKDTree(centers).query(point_xy)
    unmatched = assigned < 0
    nearby = unmatched & (nearest_m <= GRID_ASSIGNMENT_MAX_M)
    assigned[nearby] = nearest_index[nearby]
    snap_m = nearest_m.copy()
    inside = assigned >= 0
    snap_m[inside] = np.linalg.norm(point_xy[inside] - centers[assigned[inside]], axis=1)
    return assigned, snap_m


def build_grid_road_graph(roads: gpd.GeoDataFrame, grid: gpd.GeoDataFrame) -> dict:
    """Contract roads to adjacent 500 m cells and keep the shortest crossing."""
    grid_geometry = grid.geometry.to_numpy()
    grid_xy = np.column_stack((grid.geometry.centroid.x, grid.geometry.centroid.y))
    overlap = gpd.sjoin(
        roads[["src_type", "speed_cls", "geometry"]],
        grid[["geometry"]],
        how="inner",
        predicate="intersects",
    )
    drive_edges: dict[tuple[int, int], tuple[float, float]] = {}
    walk_edges: dict[tuple[int, int], tuple[float, float]] = {}

    def keep_shortest(
        edges: dict[tuple[int, int], tuple[float, float]],
        a: int,
        b: int,
        length_m: float,
        drive_min: float,
    ) -> None:
        key = (min(a, b), max(a, b))
        previous = edges.get(key)
        if previous is None or (length_m, drive_min) < previous:
            edges[key] = (length_m, drive_min)

    for road_index, matched in overlap.groupby(level=0, sort=False):
        cell_indices = matched["index_right"].to_numpy(dtype=int)
        if len(cell_indices) < 2:
            continue
        line = roads.geometry.iloc[int(road_index)]
        passages: list[tuple[float, int]] = []
        for cell_index in cell_indices:
            clipped = line.intersection(grid_geometry[cell_index])
            if clipped.length <= 1.0:
                continue
            segments = [clipped] if clipped.geom_type == "LineString" else [
                part for part in getattr(clipped, "geoms", []) if part.geom_type == "LineString"
            ]
            if not segments:
                continue
            segment = max(segments, key=lambda part: part.length)
            midpoint = segment.interpolate(segment.length / 2.0)
            passages.append((float(line.project(midpoint)), int(cell_index)))
        passages.sort()
        if len(passages) < 2:
            continue
        road = roads.iloc[int(road_index)]
        speed_class = int(road["speed_cls"]) if pd.notna(road["speed_cls"]) else 7
        speed_m_per_min = SPEED_CLASS_KMH.get(speed_class, 20.0) * 1000.0 / 60.0
        for (position_a, a), (position_b, b) in zip(passages, passages[1:]):
            if a == b or grid_geometry[a].distance(grid_geometry[b]) > 5.0:
                continue
            center_m = float(np.linalg.norm(grid_xy[a] - grid_xy[b]))
            if center_m > 900.0:
                continue
            road_a = line.interpolate(position_a)
            road_b = line.interpolate(position_b)
            direct_m = max(road_a.distance(road_b), 1.0)
            stretch = max(1.0, (position_b - position_a) / direct_m)
            length_m = center_m * stretch
            drive_min = length_m / speed_m_per_min
            keep_shortest(drive_edges, a, b, length_m, drive_min)
            if str(road["src_type"]) != "express":
                keep_shortest(walk_edges, a, b, length_m, drive_min)

    def add_access_connectors(
        edges: dict[tuple[int, int], tuple[float, float]]
    ) -> np.ndarray:
        connected = np.zeros(len(grid), dtype=bool)
        for a, b in edges:
            connected[a] = True
            connected[b] = True
        if not connected.any():
            raise ValueError("No adjacent 500 m cells are linked by roads")
        isolated = np.flatnonzero(~connected)
        nearest_m, nearest_pos = cKDTree(grid_xy[connected]).query(grid_xy[isolated])
        road_indices = np.flatnonzero(connected)
        for a, distance_m, position in zip(isolated, nearest_m, nearest_pos):
            b = int(road_indices[position])
            keep_shortest(
                edges, int(a), b, float(distance_m),
                float(distance_m) / DRIVE_CONNECTOR_M_PER_MIN,
            )
        return nearest_m

    drive_connectors = add_access_connectors(drive_edges)
    walk_connectors = add_access_connectors(walk_edges)

    def adjacency(edges: dict[tuple[int, int], tuple[float, float]]):
        adj: list[list[tuple[int, float, float]]] = [[] for _ in range(len(grid))]
        for (a, b), (length_m, drive_min) in edges.items():
            adj[a].append((b, length_m, drive_min))
            adj[b].append((a, length_m, drive_min))
        return adj

    return {
        "drive_adj": adjacency(drive_edges),
        "walk_adj": adjacency(walk_edges),
        "drive_road_edge_count": len(drive_edges) - len(drive_connectors),
        "walk_road_edge_count": len(walk_edges) - len(walk_connectors),
        "drive_connector_m": drive_connectors,
        "walk_connector_m": walk_connectors,
    }


def shortest_grid_road_paths(
    adj: list[list[tuple[int, float, float]]], source: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return road distance and time along each shortest-distance grid path."""
    distance_m = np.full(len(adj), np.inf, dtype=float)
    drive_min = np.full(len(adj), np.inf, dtype=float)
    if source < 0:
        return distance_m, drive_min
    distance_m[source] = 0.0
    drive_min[source] = 0.0
    heap: list[tuple[float, float, int]] = [(0.0, 0.0, source)]
    while heap:
        current_m, current_min, node = heappop(heap)
        if current_m != distance_m[node] or current_min != drive_min[node]:
            continue
        for neighbor, edge_m, edge_min in adj[node]:
            candidate_m = current_m + edge_m
            candidate_min = current_min + edge_min
            if (candidate_m, candidate_min) < (distance_m[neighbor], drive_min[neighbor]):
                distance_m[neighbor] = candidate_m
                drive_min[neighbor] = candidate_min
                heappush(heap, (candidate_m, candidate_min, neighbor))
    return distance_m, drive_min


def topsis(scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    norm = np.sqrt(np.square(scores).sum(axis=0))
    normalized = np.divide(scores, norm, out=np.zeros_like(scores), where=norm > 0)
    weighted = normalized * weights
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)
    d_best = np.sqrt(np.square(weighted - ideal_best).sum(axis=1))
    d_worst = np.sqrt(np.square(weighted - ideal_worst).sum(axis=1))
    denom = d_best + d_worst
    return np.divide(d_worst, denom, out=np.full_like(d_worst, 0.5), where=denom > 0)


def main(demand_source: str = "census_expanded") -> None:
    demand_fields = {"mobile_users": "联通人", "census_expanded": "七普加"}
    if demand_source not in demand_fields:
        raise ValueError(f"Unknown population demand source: {demand_source}")
    output_suffix = "" if demand_source == "census_expanded" else "_mobile_users"

    def result_path(stem: str, extension: str) -> Path:
        return RESULT_DIR / f"{stem}{output_suffix}.{extension}"

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    communities = gpd.read_file(COMMUNITY_PATH).to_crs(4326)
    centers = gpd.read_file(CENTER_PATH).to_crs(4326)
    communities["SQCODE"] = normalize_code(communities["SQCODE"])
    centers["SQCODE"] = normalize_code(centers["SQCODE"])
    centers["community_id"] = pd.to_numeric(centers["OBJECTID"], errors="raise").astype(int)
    centers = centers.sort_values("community_id").reset_index(drop=True)
    centers["community_index"] = np.arange(len(centers), dtype=int)
    grids = load_population_grids(centers)

    facilities, footprints = build_facilities()
    facilities_proj = facilities.to_crs(PROJECTED_CRS)
    centers_proj = centers.to_crs(PROJECTED_CRS)
    grids_proj = grids.to_crs(PROJECTED_CRS)
    demand_points = grids_proj.geometry.centroid
    roads = gpd.read_file(ROAD_PATH).to_crs(PROJECTED_CRS).reset_index(drop=True)
    road_metadata = json.loads(ROAD_META_PATH.read_text(encoding="utf-8"))
    network = build_network(roads)
    facility_points = facilities_proj.geometry
    grid_points = demand_points
    walk_facility_snap = snap_points(facility_points, network["nonexpress_edges"], "walk")
    walk_center_snap = snap_points(centers_proj.geometry, network["nonexpress_edges"], "walk")
    walk_grid_snap = snap_points(grid_points, network["nonexpress_edges"], "walk")
    cycle_facility_snap = snap_points(facility_points, network["cycle_edges"], "cycle")
    cycle_center_snap = snap_points(centers_proj.geometry, network["cycle_edges"], "cycle")
    cycle_grid_snap = snap_points(grid_points, network["cycle_edges"], "cycle")
    drive_facility_snap = snap_points(facility_points, network["all_edges"], "drive")
    drive_center_snap = snap_points(centers_proj.geometry, network["all_edges"], "drive")
    drive_grid_snap = snap_points(grid_points, network["all_edges"], "drive")

    n_facilities = len(facilities)
    n_communities = len(centers)
    n_grids = len(grids)
    grid_community = grids["community_index"].to_numpy(dtype=int)
    demand = grids[demand_fields[demand_source]].to_numpy(dtype=float)
    demand_by_community = np.bincount(grid_community, weights=demand, minlength=n_communities)
    grid_indices_by_community = [np.flatnonzero(grid_community == c) for c in range(n_communities)]
    facility_candidates: list[set[int]] = [set() for _ in range(n_facilities)]
    candidate_records: list[dict] = []
    travel_counts = {tier: np.zeros((n_grids, 3), dtype=np.int16) for tier in TIER_ORDER}
    nearest_time = {tier: np.full(n_grids, np.inf, dtype=float) for tier in TIER_ORDER}
    center_indices = np.arange(n_communities, dtype=int)

    # Life circle: the 15-minute walking isochrone; the community boundary only where no isochrone exists.
    index_by_code = dict(zip(centers["SQCODE"], centers["community_index"]))
    boundaries = communities.to_crs(PROJECTED_CRS)
    boundaries["community_index"] = boundaries["SQCODE"].map(index_by_code)
    if boundaries["community_index"].isna().any():
        raise ValueError("Every community boundary must match a modeled community")
    circles = gpd.read_file(ISOCHRONE_PATH).to_crs(PROJECTED_CRS)
    circles["community_index"] = normalize_code(circles["num"]).map(index_by_code)
    circles = circles.loc[circles["community_index"].notna()]
    no_circle = boundaries.loc[~boundaries["community_index"].isin(circles["community_index"])]
    life_circles = gpd.GeoDataFrame(
        pd.concat([no_circle[["community_index", "geometry"]], circles[["community_index", "geometry"]]],
                  ignore_index=True),
        geometry="geometry", crs=PROJECTED_CRS,
    )
    life_circles["community_index"] = life_circles["community_index"].astype(int)
    life_circles["geometry"] = shapely.make_valid(life_circles.geometry.to_numpy())
    member_pairs = gpd.sjoin(
        gpd.GeoDataFrame({"facility_index": np.arange(n_facilities)}, geometry=footprints.to_numpy(),
                         crs=PROJECTED_CRS),
        life_circles, how="inner", predicate="intersects",
    )[["facility_index", "community_index"]].drop_duplicates()
    life_circle_members = {
        int(f): sorted(group["community_index"].astype(int))
        for f, group in member_pairs.groupby("facility_index")
    }
    grid_pairs = gpd.sjoin(
        gpd.GeoDataFrame({"grid_index": np.arange(n_grids)}, geometry=grid_points.to_numpy(), crs=PROJECTED_CRS),
        life_circles, how="inner", predicate="intersects",
    )[["grid_index", "community_index"]].drop_duplicates()

    for facility_idx in range(n_facilities):
        tier = str(facilities.at[facility_idx, "tier"])
        mode = TIER_MODE[tier]
        if mode == "walk":
            facility_snap = walk_facility_snap
            center_snap = walk_center_snap
            grid_snap = walk_grid_snap
            adjacency = network["distance_adj"]
            speed = WALK_M_PER_MIN
            source_side = False
        elif mode == "cycle":
            facility_snap = cycle_facility_snap
            center_snap = cycle_center_snap
            grid_snap = cycle_grid_snap
            adjacency = network["cycle_adj"]
            speed = CYCLE_M_PER_MIN
            source_side = False
        else:
            facility_snap = drive_facility_snap
            center_snap = drive_center_snap
            grid_snap = drive_grid_snap
            adjacency = network["drive_reverse_adj"]
            speed = 1.0
            source_side = True
        sources = (
            (int(facility_snap["u"][facility_idx]), float(facility_snap["cost_u"][facility_idx])),
            (int(facility_snap["v"][facility_idx]), float(facility_snap["cost_v"][facility_idx])),
        )
        distances = dijkstra(adjacency, sources, 15.0 * speed)
        center_time_min = route_values(
            distances, center_snap, center_indices, source_side=source_side,
            source_snap=facility_snap, source_index=facility_idx, mode=mode,
        ) / speed
        selected = life_circle_members.get(facility_idx, [])
        facility_candidates[facility_idx].update(selected)
        for community_idx in selected:
            candidate_records.append(
                {
                    "facility_id": facilities.at[facility_idx, "facility_id"],
                    "tier": facilities.at[facility_idx, "tier"],
                    "community_id": int(centers.at[community_idx, "community_id"]),
                    "SQCODE": centers.at[community_idx, "SQCODE"],
                    "community_name": centers.at[community_idx, "SQNAME"],
                    "center_travel_min": (float(center_time_min[community_idx])
                                          if np.isfinite(center_time_min[community_idx]) else np.nan),
                    "travel_mode": mode,
                    "membership_rule": "社康点位或医院AOI与行政社区15分钟步行等时圈相交",
                }
            )
        if len(selected) == 0:
            continue
        candidate_grid_indices = np.concatenate([grid_indices_by_community[int(c)] for c in selected])
        grid_time_min = route_values(
            distances, grid_snap, candidate_grid_indices, source_side=source_side,
            source_snap=facility_snap, source_index=facility_idx, mode=mode,
        ) / speed
        within = grid_time_min <= 15.0
        valid_indices = candidate_grid_indices[within]
        nearest_time[tier][valid_indices] = np.minimum(nearest_time[tier][valid_indices], grid_time_min[within])
        # Life-circle facilities beyond 15 road minutes are counted at 15 minutes.
        band_times = np.minimum(grid_time_min, 15.0)
        for band_idx, (lower, upper) in enumerate(TRAVEL_BANDS):
            band_mask = (band_times > lower) & (band_times <= upper)
            if lower == 0:
                band_mask = (band_times >= lower) & (band_times <= upper)
            travel_counts[tier][candidate_grid_indices[band_mask], band_idx] += 1

    facility_candidate_count = np.array([len(v) for v in facility_candidates], dtype=int)
    facilities["candidate_community_count"] = facility_candidate_count

    resource_availability = {tier: np.zeros(n_communities, dtype=float) for tier in TIER_ORDER}
    facility_count = {tier: np.zeros(n_communities, dtype=int) for tier in TIER_ORDER}
    capacity_sum = {tier: np.zeros(n_communities, dtype=float) for tier in TIER_ORDER}
    for facility_idx, candidate_set in enumerate(facility_candidates):
        tier = str(facilities.at[facility_idx, "tier"])
        for community_idx in candidate_set:
            facility_count[tier][community_idx] += 1
            capacity_sum[tier][community_idx] += float(facilities.at[facility_idx, "capacity"])
            resource_availability[tier][community_idx] += float(facilities.at[facility_idx, "resource_weight"])

    accessibility_community = {tier: np.zeros(n_communities, dtype=float) for tier in TIER_ORDER}
    h2_denominator = np.zeros(n_facilities, dtype=float)
    h2_ratio = np.zeros(n_facilities, dtype=float)
    h2_grid_count = np.zeros(n_facilities, dtype=int)
    grid_point_tree = shapely.STRtree(np.asarray(list(grid_points.array), dtype=object))

    # CHC service range is the community's own life circle; grids are assigned by their centre point.
    life_circle_demand = np.bincount(
        grid_pairs["community_index"].to_numpy(dtype=int),
        weights=demand[grid_pairs["grid_index"].to_numpy(dtype=int)],
        minlength=n_communities,
    )
    accessibility_community["CHC"] = np.divide(
        capacity_sum["CHC"], life_circle_demand,
        out=np.zeros(n_communities, dtype=float), where=life_circle_demand > 0,
    )

    for facility_idx in range(n_facilities):
        tier = str(facilities.at[facility_idx, "tier"])
        if tier == "CHC":
            h2_denominator[facility_idx] = np.nan
            h2_ratio[facility_idx] = np.nan
            continue
        threshold = H2_CATCHMENT_MIN[tier]
        mode = H2_MODE[tier]
        if mode in {"walk", "cycle"}:
            speed = WALK_M_PER_MIN if mode == "walk" else CYCLE_M_PER_MIN
            facility_snap = walk_facility_snap if mode == "walk" else cycle_facility_snap
            grid_snap = walk_grid_snap if mode == "walk" else cycle_grid_snap
            adjacency = network["distance_adj"] if mode == "walk" else network["cycle_adj"]
            cutoff = threshold * speed
            sources = (
                (int(facility_snap["u"][facility_idx]), float(facility_snap["cost_u"][facility_idx])),
                (int(facility_snap["v"][facility_idx]), float(facility_snap["cost_v"][facility_idx])),
            )
            distances = dijkstra(adjacency, sources, cutoff)
            candidate_grid_indices = np.asarray(
                grid_point_tree.query(facility_points.iloc[facility_idx].buffer(cutoff)), dtype=int
            )
            route_time = route_values(
                distances, grid_snap, candidate_grid_indices,
                source_snap=facility_snap, source_index=facility_idx, mode=mode,
            ) / speed
        else:
            sources = (
                (int(drive_facility_snap["u"][facility_idx]), float(drive_facility_snap["cost_u"][facility_idx])),
                (int(drive_facility_snap["v"][facility_idx]), float(drive_facility_snap["cost_v"][facility_idx])),
            )
            distances = dijkstra(network["drive_reverse_adj"], sources, threshold)
            candidate_grid_indices = np.arange(n_grids, dtype=int)
            route_time = route_values(
                distances, drive_grid_snap, candidate_grid_indices, source_side=True,
                source_snap=drive_facility_snap, source_index=facility_idx, mode="drive",
            )

        within = route_time <= threshold
        if not within.any():
            continue
        reached_grid_indices = candidate_grid_indices[within]
        denominator = float(np.sum(demand[reached_grid_indices]))
        supply = float(facilities.at[facility_idx, "capacity"])
        ratio = supply / denominator if denominator > 0 else 0.0
        h2_denominator[facility_idx] = denominator
        h2_ratio[facility_idx] = ratio
        h2_grid_count[facility_idx] = len(reached_grid_indices)

        candidate_set = facility_candidates[facility_idx]
        if ratio <= 0 or len(candidate_set) == 0:
            continue
        accessibility_community[tier][list(candidate_set)] += ratio

    accessibility_grid = {
        tier: accessibility_community[tier][grid_community] for tier in TIER_ORDER
    }

    facilities["h2_denominator_weighted_demand"] = h2_denominator
    facilities["h2_supply_demand_ratio"] = h2_ratio
    facilities["h2_catchment_grid_count"] = h2_grid_count
    facilities["walk_snap_m"] = walk_facility_snap["connector_m"]
    facilities["cycle_snap_m"] = cycle_facility_snap["connector_m"]
    facilities["drive_snap_m"] = drive_facility_snap["connector_m"]

    output = pd.DataFrame(
        {
            "community_id": centers["community_id"].astype(int),
            "SQCODE": centers["SQCODE"].astype("string"),
            "community_name": centers["SQNAME"].astype("string"),
            "population_demand": demand_by_community,
            "positive_demand_grid_count": np.bincount(grid_community, minlength=n_communities),
        }
    )
    for tier in TIER_ORDER:
        output[f"facility_count_{tier}"] = facility_count[tier]
        output[f"capacity_sum_{tier}"] = capacity_sum[tier]
        output[f"RA_{tier}"] = resource_availability[tier]

        mean = np.where(demand_by_community > 0, accessibility_community[tier], np.nan)
        output[f"SA_{tier}"] = mean
        output[f"SA_{tier}_per_10000"] = mean * 10000.0
        output[f"SA_CV_{tier}"] = np.where(mean > 0, 0.0, np.nan)
        output[f"SA_zero_access_demand_pct_{tier}"] = np.where(
            demand_by_community > 0, np.where(mean > 0, 0.0, 100.0), np.nan,
        )

        q = np.zeros((n_communities, 3), dtype=float)
        for band_idx, (lower, upper) in enumerate(TRAVEL_BANDS):
            q[:, band_idx] = np.divide(
                np.bincount(
                    grid_community,
                    weights=demand * travel_counts[tier][:, band_idx],
                    minlength=n_communities,
                ),
                demand_by_community,
                out=np.zeros(n_communities, dtype=float),
                where=demand_by_community > 0,
            )
            output[f"Q_{int(lower)}_{int(upper)}_{tier}"] = np.where(
                demand_by_community > 0, q[:, band_idx], np.nan
            )
        output[f"TC_{tier}"] = np.where(
            demand_by_community > 0, topsis(q, TRAVEL_WEIGHTS), np.nan
        )

        reached = np.isfinite(nearest_time[tier])
        reached_demand = np.bincount(
            grid_community,
            weights=demand * reached,
            minlength=n_communities,
        )
        reached_time_sum = np.bincount(
            grid_community[reached],
            weights=demand[reached] * nearest_time[tier][reached],
            minlength=n_communities,
        )
        output[f"within_15min_population_pct_{tier}"] = np.divide(
            reached_demand * 100.0,
            demand_by_community,
            out=np.full(n_communities, np.nan),
            where=demand_by_community > 0,
        )
        output[f"nearest_travel_mean_min_served_{tier}"] = np.divide(
            reached_time_sum,
            reached_demand,
            out=np.full(n_communities, np.nan),
            where=reached_demand > 0,
        )
        p90 = np.full(n_communities, np.nan, dtype=float)
        for community_idx, indices in enumerate(grid_indices_by_community):
            valid_indices = indices[np.isfinite(nearest_time[tier][indices])]
            p90[community_idx] = weighted_quantile(
                nearest_time[tier][valid_indices],
                demand[valid_indices],
                0.90,
            )
        output[f"nearest_travel_p90_min_served_{tier}"] = p90

    output_path = result_path("community_supply_indicators_2018", "csv")
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    candidate_frame = pd.DataFrame(candidate_records)
    candidate_frame.to_csv(RESULT_DIR / "facility_community_15min.csv", index=False, encoding="utf-8-sig")

    facility_export = facilities.drop(columns="geometry").copy()
    facility_export["longitude"] = facilities.geometry.x
    facility_export["latitude"] = facilities.geometry.y
    facility_export.to_csv(result_path("facility_supply_audit_2018", "csv"), index=False, encoding="utf-8-sig")

    grid_export = pd.DataFrame(
        {
            "Tid": grids["Tid"].to_numpy(),
            "community_id": centers.iloc[grid_community]["community_id"].to_numpy(),
            "SQCODE": grids["SQCODE"].to_numpy(),
            "population_demand": demand,
            "walk_snap_m": walk_grid_snap["connector_m"],
            "cycle_snap_m": cycle_grid_snap["connector_m"],
            "drive_snap_m": drive_grid_snap["connector_m"],
        }
    )
    for tier in TIER_ORDER:
        grid_export[f"accessibility_{tier}"] = accessibility_grid[tier]
        for band_idx, (lower, upper) in enumerate(TRAVEL_BANDS):
            grid_export[f"facility_count_{int(lower)}_{int(upper)}min_{tier}"] = travel_counts[tier][:, band_idx]
        grid_export[f"nearest_travel_min_{tier}"] = np.where(
            np.isfinite(nearest_time[tier]), nearest_time[tier], np.nan
        )
    grid_export.to_csv(result_path("grid_supply_diagnostics_2018", "csv"), index=False, encoding="utf-8-sig")

    polygon_output = communities.merge(output, on="SQCODE", how="right", suffixes=("_polygon", ""))
    polygon_output = gpd.GeoDataFrame(polygon_output, geometry="geometry", crs=communities.crs)
    polygon_output.to_file(
        result_path("community_supply_indicators_2018", "gpkg"),
        layer="community_supply_indicators",
        driver="GPKG",
    )

    tier_medians = {
        tier: float(facilities.loc[facilities["tier"] == tier, "capacity_median_tier"].iloc[0])
        for tier in TIER_ORDER
    }
    audit = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "community": str(COMMUNITY_PATH),
            "population_centers": str(CENTER_PATH),
            "resident_population_grids_250m": str(POPULATION_GRID_PATH),
            "roads": str(ROAD_PATH),
            "road_snapshot": road_metadata["source_url"],
            "road_snapshot_date": road_metadata["source_date"],
            "road_snapshot_sha256": road_metadata["source_pbf_sha256"],
            "road_license": road_metadata["source_license"],
            "facilities": str(FACILITY_PATH),
            "hospital_aoi": str(HOSPITAL_AOI_PATH),
            "demand_allocation": str(DEMAND_ALLOCATION_PATH),
            "life_circle_isochrone": str(ISOCHRONE_PATH),
        },
        "demand_definition": {
            "source": demand_source,
            "field": demand_fields[demand_source],
            "rule": "250米居住网格人口按step1定稿规则归属社区（无争议网格maup整格归属，有争议网格按城市建设用地面积比例分配），网格取中心点；行政社区需求为归属该社区的网格人口之和",
            "positive_demand_grid_unique": int(grids["Tid"].nunique()),
            "split_grid_count": int((grids["Tid"].value_counts() > 1).sum()),
            "positive_demand_grid_count": int(n_grids),
            "community_count": int(n_communities),
            "communities_with_positive_demand": int((demand_by_community > 0).sum()),
            "total_population_demand": float(demand.sum()),
            "excluded_without_community": grids.attrs["excluded_without_community"],
        },
        "facility_definition": {
            "CHC": "facility_supply_audit_2018_mobile_users.csv中的2018年社康点位",
            "SH": "facility_supply_audit_2018_mobile_users.csv中的二级医院，位置取hospital_aoi_35.shp的AOI",
            "TH": "facility_supply_audit_2018_mobile_users.csv中的三级医院，位置取hospital_aoi_35.shp的AOI",
            "hospital_location": "生活圈归属按AOI与等时圈相交判定；路径计算取AOI代表点（point_on_surface）",
            "maternal_child_health_hospitals_in_source": int(
                facilities.loc[facilities["tier"].isin(["SH", "TH"]), "facility_name"]
                .astype("string").str.contains("妇幼保健院", na=False).sum()
            ),
            "counts": facilities["tier"].value_counts().reindex(TIER_ORDER).astype(int).to_dict(),
        },
        "resource_availability": {
            "formula": "RA_c^l = sum_j[(1-lambda)+lambda*(C_j/median_l)]",
            "lambda": LAMBDA,
            "capacity": {"CHC": "服务能力评分", "SH": "床位数", "TH": "床位数"},
            "positive_capacity_median": tier_medians,
            "area_used": False,
        },
        "spatial_accessibility": {
            "method": "CHC: total CHC capacity in the community's life circle divided by total population in that life circle; SH/TH: facility capacity divided by total population in its tier-specific service catchment, summed over facilities in each community's life circle; life circle = 15-minute walking isochrone (community boundary where no isochrone exists)",
            "gaussian_decay": "not used in SA",
            "catchment_minutes": H2_CATCHMENT_MIN,
            "modes": H2_MODE,
            "secondary_tertiary_drive_speed": "OSM highway类别默认速度，并以有效maxspeed限速作为上限；见道路提取元数据",
            "supply": {"CHC": "服务能力评分", "SH": "床位数", "TH": "床位数"},
            "competition_demand": {
                "CHC": "本社区生活圈内250米居住网格人口，生活圈为行政社区15分钟步行等时圈（无等时圈的社区用社区边界），网格按中心点判定",
                "SH": f"设施{H2_CATCHMENT_MIN['SH']:.0f}分钟驾车服务范围内250米居住网格人口",
                "TH": f"设施{H2_CATCHMENT_MIN['TH']:.0f}分钟驾车服务范围内250米居住网格人口",
            },
            "aggregation": "生活圈内同层级设施的服务能力与其服务范围内人口之比求和",
        },
        "travel_convenience": {
            "facility_membership": "社康点位或医院AOI与行政社区15分钟步行等时圈相交（无等时圈的社区用社区边界）；分档时间为居住网格到设施的OSM步行最短路时间，超过15分钟或不连通的按15分钟计",
            "modes": TIER_MODE,
            "walking_speed_kmh": WALK_M_PER_MIN * 60.0 / 1000.0,
            "bands_minutes": TRAVEL_BANDS,
            "topsis_weights": TRAVEL_WEIGHTS.tolist(),
            "aggregation": "各250米居住网格分档内可达设施数的人口加权平均，再分层TOPSIS",
        },
        "network": {
            "crs": PROJECTED_CRS,
            "driving_respects_oneway": True,
            "hospital_access_direction": "居住网格到医院；以反向驾车图从医院计算最短路径",
            "junctions_by_shared_osm_node_id": True,
            "walking_and_cycling_use_osm_access_tags": True,
            "method": "社康点位、医院AOI代表点、行政社区人口重心和250米人口网格中心点接入OSM道路线段；按通行方式使用道路最短距离或驾车最短时间",
            "artificial_bridge_tolerance_m": 0.0,
            "road_segment_count": int(len(roads)),
            "network_node_count": int(len(network["node_xy"])),
            "road_segments_by_mode": road_metadata["segments_by_mode"],
            "snap_distance_m": {
                "facility_walk_to_road": {
                    "median": float(np.median(walk_facility_snap["connector_m"])),
                    "p95": float(np.percentile(walk_facility_snap["connector_m"], 95)),
                    "max": float(np.max(walk_facility_snap["connector_m"])),
                },
                "community_center_walk_to_road": {
                    "median": float(np.median(walk_center_snap["connector_m"])),
                    "p95": float(np.percentile(walk_center_snap["connector_m"], 95)),
                    "max": float(np.max(walk_center_snap["connector_m"])),
                },
                "resident_grid_center_walk_to_road": {
                    "median": float(np.median(walk_grid_snap["connector_m"])),
                    "p95": float(np.percentile(walk_grid_snap["connector_m"], 95)),
                    "max": float(np.max(walk_grid_snap["connector_m"])),
                },
            },
        },
        "candidate_relations": {
            "total": int(len(candidate_frame)),
            "community_center_count": int(n_communities),
            "facilities_without_candidate_community": int((facility_candidate_count == 0).sum()),
            "communities_without_facility_by_tier": {
                tier: int((facility_count[tier] == 0).sum()) for tier in TIER_ORDER
            },
        },
        "validation": {
            "output_rows": int(len(output)),
            "all_communities_present": bool(len(output) == 666),
            "nonnegative_main_indicators": bool(
                all((output[[f"RA_{tier}", f"SA_{tier}", f"TC_{tier}"]].fillna(0) >= 0).all().all() for tier in TIER_ORDER)
            ),
        },
    }
    result_path("calculation_audit_2018", "json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"result_dir": str(RESULT_DIR), "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main("mobile_users")
    main("census_expanded")

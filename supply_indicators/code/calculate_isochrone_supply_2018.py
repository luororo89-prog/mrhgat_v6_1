"""Calculate nine indicators using community CHC and hospital isochrones."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

import calculate_supply_indicators_2018 as base


OUT = base.ROOT / "26上实验" / "medical_identification_validation" / "results" / "isochrone_supply_2018"
D = {"CHC": 15.0, "SH": 25.0, "TH": 30.0}
HOSPITAL_MODE = {"SH": "drive", "TH": "drive"}


def routes(source_snap, source_id, target_snap, targets, adj, cutoff, mode):
    origins = ((int(source_snap["u"][source_id]), float(source_snap["cost_u"][source_id])),
               (int(source_snap["v"][source_id]), float(source_snap["cost_v"][source_id])))
    distance = base.dijkstra(adj, origins, cutoff)
    cost = base.route_values(distance, target_snap, targets, source_side=mode == "drive",
                             source_snap=source_snap, source_index=source_id, mode=mode)
    return distance, cost


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    centers = gpd.read_file(base.CENTER_PATH).to_crs(4326)
    centers["SQCODE"] = base.normalize_code(centers["SQCODE"])
    centers["community_id"] = pd.to_numeric(centers["OBJECTID"], errors="raise").astype(int)
    centers = centers.sort_values("community_id").reset_index(drop=True)
    centers["community_index"] = np.arange(len(centers))
    n = len(centers)
    center_points = centers.to_crs(base.PROJECTED_CRS).geometry
    communities = gpd.read_file(base.COMMUNITY_PATH).to_crs(base.PROJECTED_CRS)
    communities["SQCODE"] = base.normalize_code(communities["SQCODE"])
    communities["community_index"] = communities["SQCODE"].map(
        dict(zip(centers["SQCODE"], centers["community_index"])))
    communities = communities.loc[communities["community_index"].notna()].reset_index(drop=True)
    community_geometry = np.asarray(shapely.make_valid(communities.geometry.to_numpy()), dtype=object)
    community_numbers = communities["community_index"].to_numpy(dtype=int)
    community_tree = shapely.STRtree(community_geometry)

    grids = base.load_population_grids(centers).to_crs(base.PROJECTED_CRS)
    grid_points = grids.geometry.centroid
    grid_tree = shapely.STRtree(np.asarray(grid_points.to_list(), dtype=object))
    grid_population = grids["联通人"].to_numpy(dtype=float)
    all_grids = np.arange(len(grids))
    facilities = base.build_facilities()
    facility_points = facilities.to_crs(base.PROJECTED_CRS).geometry
    chc = facilities.loc[facilities["tier"] == "CHC"].reset_index(drop=True)
    hospitals = facilities.loc[facilities["tier"].isin(["SH", "TH"])].reset_index(drop=True)
    chc_points = facility_points.iloc[:len(chc)].reset_index(drop=True)
    hospital_points = facility_points.iloc[len(chc):].reset_index(drop=True)

    roads = gpd.read_file(base.ROAD_PATH).to_crs(base.PROJECTED_CRS).reset_index(drop=True)
    network = base.build_network(roads)
    print("Road network ready", flush=True)
    walk_center = base.snap_points(center_points, network["nonexpress_edges"], "walk")
    walk_chc = base.snap_points(chc_points, network["nonexpress_edges"], "walk")
    walk_grid = base.snap_points(grid_points, network["nonexpress_edges"], "walk")
    cycle_hospital = base.snap_points(hospital_points, network["cycle_edges"], "cycle")
    cycle_grid = base.snap_points(grid_points, network["cycle_edges"], "cycle")
    drive_hospital = base.snap_points(hospital_points, network["all_edges"], "drive")
    drive_grid = base.snap_points(grid_points, network["all_edges"], "drive")
    print("Road connections ready", flush=True)

    node_xy = network["node_xy"]
    node_points = shapely.points(node_xy[:, 0], node_xy[:, 1])
    node_idx, community_row = community_tree.query(node_points, predicate="intersects")
    node_community = community_numbers[community_row]
    print("Community road nodes ready", flush=True)
    data = {tier: {"RA": np.zeros(n), "SA": np.zeros(n), "Q": np.zeros((n, 3))} for tier in D}

    # CHC facilities belong to a community when they fall inside its 15-minute walking life circle.
    circles = gpd.read_file(base.ISOCHRONE_PATH).to_crs(base.PROJECTED_CRS)
    circles["community_index"] = base.normalize_code(circles["num"]).map(
        dict(zip(centers["SQCODE"], centers["community_index"])))
    circles = circles.loc[circles["community_index"].notna()].reset_index(drop=True)
    circles["geometry"] = shapely.make_valid(circles.geometry.to_numpy())
    members = gpd.sjoin(gpd.GeoDataFrame(geometry=chc_points.to_numpy(), crs=base.PROJECTED_CRS),
                        circles[["community_index", "geometry"]], how="inner", predicate="within")
    chc_members = {int(c): np.unique(group.index.to_numpy())
                   for c, group in members.groupby("community_index")}
    grid_inside = gpd.sjoin(gpd.GeoDataFrame(geometry=grid_points.to_numpy(), crs=base.PROJECTED_CRS),
                            circles[["community_index", "geometry"]], how="inner", predicate="within")
    grid_members = {int(c): np.unique(group.index.to_numpy())
                    for c, group in grid_inside.groupby("community_index")}
    chc_capacity = chc["capacity"].to_numpy(dtype=float)
    clipped_pairs = 0
    for c in range(n):
        origin = center_points.iloc[c]
        cutoff = D["CHC"] * base.WALK_M_PER_MIN
        ids = chc_members.get(c, np.zeros(0, dtype=int))
        grid_ids = grid_members.get(c, np.zeros(0, dtype=int))
        distance, facility_cost = routes(walk_center, c, walk_chc, ids,
                                         network["distance_adj"], cutoff, "walk")
        # Facilities inside the life circle beyond 15 road minutes are counted at 15 minutes.
        clipped_pairs += int(np.count_nonzero(facility_cost / base.WALK_M_PER_MIN > D["CHC"]))
        facility_time = np.minimum(facility_cost / base.WALK_M_PER_MIN, D["CHC"])
        chosen = facility_time <= D["CHC"]
        # CHC capacity and competition are plain sums without distance decay.
        ra = float(np.sum(chc_capacity[ids[chosen]]))
        competition = float(np.sum(grid_population[grid_ids]))
        data["CHC"]["RA"][c] = ra
        data["CHC"]["SA"][c] = 10000 * ra / competition if competition > 0 else 0
        for b in range(3):
            lo, hi = 5 * b, 5 * (b + 1)
            data["CHC"]["Q"][c, b] = np.count_nonzero(
                (facility_time >= lo if b == 0 else facility_time > lo)
                & (facility_time <= hi))
        if (c + 1) % 100 == 0:
            print(f"CHC communities {c + 1}/{n}", flush=True)

    polygons = []
    hospital_rows = []
    fallback_count = 0
    for h, hospital in hospitals.iterrows():
        tier = str(hospital["tier"])
        mode = HOSPITAL_MODE[tier]
        speed = base.CYCLE_M_PER_MIN if mode == "cycle" else 1.0
        source = cycle_hospital if mode == "cycle" else drive_hospital
        target = cycle_grid if mode == "cycle" else drive_grid
        adj = network["cycle_adj"] if mode == "cycle" else network["drive_reverse_adj"]
        distance, grid_cost = routes(source, h, target, all_grids, adj, D[tier] * speed, mode)
        grid_time = grid_cost / speed
        reached_grid = grid_time <= D[tier]
        competition = float(np.sum(grid_population[reached_grid]
                            * base.gaussian_decay(grid_time[reached_grid], D[tier])))
        beds = float(hospital["capacity"])
        ratio = beds / competition if competition > 0 else 0
        reached_nodes = np.fromiter(distance.keys(), dtype=int)
        node_time = np.full(len(node_xy), math.inf)
        node_time[reached_nodes] = np.fromiter(distance.values(), dtype=float) / speed
        reachable_xy = node_xy[reached_nodes]
        bins = np.floor(reachable_xy / 100.0).astype(np.int64)
        _, sampled = np.unique(bins, axis=0, return_index=True)
        polygon = shapely.concave_hull(shapely.multipoints(reachable_xy[np.sort(sampled)]),
                                       ratio=0.0, allow_holes=True)
        if polygon.geom_type not in ("Polygon", "MultiPolygon"):
            polygon = polygon.buffer(1.0)
        polygon = shapely.make_valid(polygon)
        polygons.append(polygon)
        candidates = community_tree.query(polygon, predicate="intersects")
        area = shapely.area(shapely.intersection(community_geometry[candidates], polygon))
        selected_rows = candidates[area > 0]
        covered = community_numbers[selected_rows]
        time = np.full(n, math.inf)
        np.minimum.at(time, node_community, node_time[node_idx])
        missing = covered[~np.isfinite(time[covered])]
        if len(missing):
            tree = cKDTree(node_xy[reached_nodes])
            for c in missing:
                row = selected_rows[np.flatnonzero(covered == c)[0]]
                overlap = shapely.intersection(polygon, community_geometry[row])
                point = shapely.point_on_surface(overlap)
                _, nearest = tree.query([point.x, point.y])
                time[c] = node_time[reached_nodes[nearest]]
            fallback_count += len(missing)
        weight = base.gaussian_decay(time[covered], D[tier])
        data[tier]["RA"][covered] += beds * weight
        data[tier]["SA"][covered] += 10000 * ratio * weight
        for b in range(3):
            lo, hi = b * D[tier] / 3, (b + 1) * D[tier] / 3
            chosen = (time[covered] >= lo if b == 0 else time[covered] > lo) & (time[covered] <= hi)
            data[tier]["Q"][covered[chosen], b] += 1
        hospital_rows.append({"facility_id": hospital["facility_id"],
                              "facility_name": hospital["facility_name"],
                              "tier": tier, "beds": beds, "mode": mode,
                              "minutes": D[tier], "covered_communities": len(covered),
                              "competition_population": competition})
        print(f"Hospitals {h + 1}/{len(hospitals)}", flush=True)

    output = centers[["community_id", "SQCODE", "SQNAME"]].rename(
        columns={"SQNAME": "community_name"}).copy()
    for tier in D:
        output[f"RA_{tier}"] = data[tier]["RA"]
        output[f"SA_{tier}_per_10000"] = data[tier]["SA"]
        output[f"TC_{tier}"] = base.topsis(data[tier]["Q"], base.TRAVEL_WEIGHTS)
        for b in range(3):
            output[f"Q_{tier}_{b + 1}"] = data[tier]["Q"][:, b].astype(int)
    output.to_csv(OUT / "nine_supply_indicators_2018.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(hospital_rows).to_csv(OUT / "included_hospitals_2018.csv", index=False, encoding="utf-8-sig")
    gpd.GeoDataFrame(pd.DataFrame(hospital_rows), geometry=polygons,
                     crs=base.PROJECTED_CRS).to_file(OUT / "hospital_isochrones_2018.gpkg", driver="GPKG")
    road_meta = json.loads(Path(base.ROAD_META_PATH).read_text(encoding="utf-8"))
    audit = {"created_at": datetime.now().isoformat(timespec="seconds"),
             "inputs": {"community": str(base.COMMUNITY_PATH), "centers": str(base.CENTER_PATH),
                        "population_grid": str(base.POPULATION_GRID_PATH), "CHC": str(base.CHC_PATH),
                        "hospital": str(base.HOSPITAL_PATH), "life_circle": str(base.ISOCHRONE_PATH),
                        "road": str(base.ROAD_PATH),
                        "road_date": road_meta["source_date"], "road_source": road_meta["source_url"]},
             "population_field": "联通人", "facility_counts": {"CHC": len(chc),
                 "SH": int(sum(hospitals["tier"] == "SH")), "TH": int(sum(hospitals["tier"] == "TH"))},
             "community_count": n, "grid_count": len(grids),
             "hospital_modes": HOSPITAL_MODE, "hospital_minutes": {"SH": D["SH"], "TH": D["TH"]},
             "hospital_isochrone": "最紧凹壳面包络按100米空间单元抽稀的可达OSM道路节点，保留孔洞",
             "hospital_coverage": "等时圈与行政社区正面积相交",
             "representative_time": "社区内可达道路节点至医院的最短道路时间",
             "representative_time_fallback_count": fallback_count,
             "CHC_scope": "落在社区15分钟步行生活圈面内的社康",
             "CHC_capacity": "生活圈面内社康服务能力之和，不做距离衰减",
             "CHC_time": "仅用于出行便利度分档；社区人口重心至社康的OSM步行最短路时间，生活圈面内超过15分钟的按15分钟计",
             "CHC_community_pairs": int(sum(len(v) for v in chc_members.values())),
             "CHC_pairs_counted_at_15_minutes": clipped_pairs,
             "CHC_outside_all_life_circles": int(len(chc) - len(members.index.unique())),
             "CHC_competition": "中心点落在社区15分钟步行生活圈面内的居住网格联通人口之和，不做距离衰减"}
    (OUT / "calculation_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), **audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

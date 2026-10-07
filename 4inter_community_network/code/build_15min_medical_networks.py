"""Build community medical flow from original OD positions and mode-specific adjacency."""

import argparse
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from run_validation import GRID, COMM, HOSP, RESULT


ROOT = Path(r"D:\多功能网络构建")
ISO_DIR = ROOT / "等时圈/等时圈wgs84拆分/等时圈wgs84拆分"
GRID250 = ROOT / "2018年11月非通勤出行/网格_人口_社区/网格_人口_社区/深圳250m网格_人口_所在社区.shp"
RA_FILE = Path(r"D:\博后三胎\input\联通人口_九项医疗供给指标_2018.xlsx")
CRS = 32650

# Authoritative hospital list (same file as the supply indicators).
HOSP_BEDS = ROOT / "26上实验/medical_identification_validation/results/hospital_beds_year/hospital_beds_amap_v3_all_2018_wgs84/hospital_beds.shp"

# Optional relocation of the data root (for example the copied data package); defaults are unchanged.
if os.environ.get("MEDNET_ROOT"):
    ROOT = Path(os.environ["MEDNET_ROOT"])
    ISO_DIR = ROOT / "等时圈/等时圈wgs84拆分/等时圈wgs84拆分"
    GRID250 = ROOT / "2018年11月非通勤出行/网格_人口_社区/网格_人口_社区/深圳250m网格_人口_所在社区.shp"
    GRID = ROOT / "SZ_grid_500m/SZ_grid_500m/SZ_grid_500m.shp"
    COMM = ROOT / "shp/深圳社区objectid.shp"
    HOSP = ROOT / "poi/poi/medical/医院_床位.shp"
    RA_FILE = Path(os.environ.get("MEDNET_RA_FILE", ROOT / "input/联通人口_九项医疗供给指标_2018.xlsx"))
    HOSP_BEDS = ROOT / "26上实验/medical_identification_validation/results/hospital_beds_year/hospital_beds_amap_v3_all_2018_wgs84/hospital_beds.shp"
if os.environ.get("MEDNET_HOSP_BEDS"):
    HOSP_BEDS = Path(os.environ["MEDNET_HOSP_BEDS"])


def polygonal(geometry):
    fixed = make_valid(geometry)
    if isinstance(fixed, (Polygon, MultiPolygon)):
        return fixed
    if isinstance(fixed, GeometryCollection):
        polygons = [part for part in fixed.geoms if isinstance(part, (Polygon, MultiPolygon))]
        return unary_union(polygons) if polygons else Polygon()
    raise ValueError(f"Unexpected isochrone geometry type: {fixed.geom_type}")


def read_circles(mode, ids):
    circles = gpd.read_file(ISO_DIR / f"{mode}等时圈.shp").to_crs(CRS)
    circles["objectid"] = pd.to_numeric(circles["objectid"], errors="raise").astype(int)
    assert circles.objectid.is_unique and set(circles.objectid).issubset(ids)
    circles.geometry = circles.geometry.apply(polygonal)
    assert circles.geometry.notna().all() and not circles.geometry.is_empty.any()
    return circles[["objectid", "geometry"]]


def assign_origin_communities(od, grids, nodes):
    grid250_polygons = gpd.read_file(GRID250).to_crs(CRS)[["Tid", "SQCODE", "geometry"]]
    grid250 = grid250_polygons.copy()
    assert grid250.Tid.is_unique
    grid250.geometry = grid250.geometry.centroid
    grid250["SQCODE"] = grid250.SQCODE.astype(str)
    joined = gpd.sjoin(grid250, grids, how="inner", predicate="within",
                       lsuffix="250", rsuffix="500")
    id_by_code = dict(zip(nodes.SQCODE.astype(str), nodes.OBJECTID.astype(int)))
    joined["community_id"] = joined.SQCODE.map(id_by_code)
    joined = joined.dropna(subset=["community_id"])
    joined["community_id"] = joined.community_id.astype(int)
    counts = (joined.groupby(["Tid_500", "community_id"]).size()
              .rename("n_250m").reset_index())
    counts["Tid_500"] = counts.Tid_500.astype(int)
    counts["probability"] = counts.n_250m / counts.groupby("Tid_500").n_250m.transform("sum")
    counts["assignment_basis"] = "matched_250m_grid_count"

    unmatched = sorted(set(od.home_tid) - set(counts.Tid_500))
    fallback_hits = gpd.sjoin(grid250_polygons, grids.loc[grids.Tid.isin(unmatched)],
                              how="inner", predicate="intersects", lsuffix="250", rsuffix="500")
    grid_shapes = grids.set_index("Tid").geometry
    fallback_hits = fallback_hits.loc[
        fallback_hits.apply(lambda row: row.geometry.intersection(
            grid_shapes.loc[int(row.Tid_500)]).area > 0, axis=1)].copy()
    fallback_hits["community_id"] = fallback_hits.SQCODE.astype(str).map(id_by_code)
    fallback_hits = fallback_hits.dropna(subset=["community_id"])
    fallback_hits["community_id"] = fallback_hits.community_id.astype(int)
    fallback = (fallback_hits.groupby(["Tid_500", "community_id"]).size()
                .rename("n_250m").reset_index())
    if not fallback.empty:
        fallback["Tid_500"] = fallback.Tid_500.astype(int)
        fallback["probability"] = (fallback.n_250m /
                                   fallback.groupby("Tid_500").n_250m.transform("sum"))
        fallback["assignment_basis"] = "intersecting_250m_grid_count"
    if not fallback.empty:
        counts = pd.concat([counts, fallback], ignore_index=True)
    weights = {int(tid): list(zip(group.community_id.astype(int), group.probability))
               for tid, group in counts.groupby("Tid_500", sort=False)}
    assert set(od.home_tid).issubset(weights)

    records = []
    allocations = []
    for row in od.itertuples(index=False):
        candidates = weights.get(int(row.home_tid))
        if not candidates:
            records.append((-1, int(row.end_tid), int(row.count)))
            continue
        ids = np.asarray([item[0] for item in candidates], dtype=int)
        shares = int(row.count) * np.asarray([item[1] for item in candidates], dtype=float)
        assigned = np.floor(shares).astype(int)
        remainder = int(row.count) - int(assigned.sum())
        order = sorted(range(len(ids)), key=lambda i: (-(shares[i] - assigned[i]), ids[i]))
        assigned[order[:remainder]] += 1
        for community_id, value in zip(ids, assigned):
            if value:
                records.append((int(community_id), int(row.end_tid), int(value)))
                allocations.append((int(row.home_tid), int(community_id), int(value)))
    resolved = pd.DataFrame(records, columns=["source_community_id", "end_tid", "count"])
    assert int(resolved["count"].sum()) == int(od["count"].sum())

    source_count = od.groupby("home_tid")["count"].sum().rename("home_grid_records")
    counts = counts.merge(source_count, left_on="Tid_500", right_index=True, how="left")
    allocation_count = (pd.DataFrame(allocations, columns=["Tid_500", "community_id", "allocated_records"])
                        .groupby(["Tid_500", "community_id"]).allocated_records.sum())
    counts = counts.merge(allocation_count, left_on=["Tid_500", "community_id"],
                          right_index=True, how="left")
    counts[["home_grid_records", "allocated_records"]] = counts[["home_grid_records", "allocated_records"]].fillna(0).astype(int)
    assert (counts.groupby("Tid_500").allocated_records.sum() ==
            counts.groupby("Tid_500").home_grid_records.first()).all()
    counts = counts.rename(columns={"Tid_500": "home_tid"})
    counts["n_communities"] = counts.groupby("home_tid").community_id.transform("size")
    counts = counts[["home_tid", "community_id", "n_250m", "n_communities",
                     "probability", "assignment_basis", "home_grid_records", "allocated_records"]]
    counts.to_csv(RESULT / "origin_500m_250m_assignment.csv", index=False, encoding="utf-8-sig")
    print("origin_audit", "mixed_grids", int(counts.loc[counts.n_communities.gt(1), "home_tid"].nunique()),
          "fallback_grids", fallback.Tid_500.nunique(), "unmapped_mass", int(resolved.loc[
              resolved.source_community_id.eq(-1), "count"].sum()), flush=True)
    return resolved, weights


def split_with_residential_destination(grouped, target, residence, ids, attraction):
    index = {int(community_id): position for position, community_id in enumerate(ids)}
    matrix = np.zeros((len(ids), len(ids)), dtype=np.int64)
    own_mass = residential_mass = medical_level_mass = equal_weight_mass = 0
    missing_source = missing_target = 0
    for row in grouped.itertuples(index=False):
        community_id = int(row.source_community_id)
        count = int(row.count)
        if community_id not in index:
            missing_source += count
            continue
        candidates = target.get(int(row.end_tid), [])
        if community_id in candidates:
            matrix[index[community_id], index[community_id]] += count
            own_mass += count
            continue
        residential = residence.get(int(row.end_tid))
        if residential:
            destinations = np.asarray([item[0] for item in residential], dtype=int)
            weights = np.asarray([item[1] for item in residential], dtype=float)
            residential_mass += count
        elif candidates:
            destinations = np.asarray(sorted(candidates), dtype=int)
            weights = np.asarray([attraction[item] for item in destinations], dtype=float)
            medical_level_mass += count
            if weights.sum() == 0:
                weights = np.ones(len(destinations), dtype=float)
                equal_weight_mass += count
        else:
            missing_target += count
            continue
        shares = count * weights / weights.sum()
        allocated = np.floor(shares).astype(np.int64)
        remainder = count - int(allocated.sum())
        order = sorted(range(len(destinations)),
                       key=lambda i: (-(shares[i] - allocated[i]), int(destinations[i])))
        allocated[order[:remainder]] += 1
        for destination_id, value in zip(destinations, allocated):
            matrix[index[community_id], index[int(destination_id)]] += value
    audit = {"assigned_mass": int(matrix.sum()), "own_circle_mass": own_mass,
             "residential_250m_mass": residential_mass,
             "medical_level_mass": medical_level_mass,
             "equal_weight_mass": equal_weight_mass,
             "missing_source_mass": missing_source, "missing_target_mass": missing_target}
    assert audit["assigned_mass"] + missing_source + missing_target == int(grouped["count"].sum())
    assert own_mass + residential_mass + medical_level_mass + missing_source + missing_target == int(grouped["count"].sum())
    return matrix, audit


def assign_grid_position_communities(grid_polygons, community, tids):
    points = grid_polygons.loc[grid_polygons.Tid.isin(tids)].copy()
    points.geometry = points.geometry.centroid
    hits = gpd.sjoin(points, community[["OBJECTID", "geometry"]],
                     how="left", predicate="intersects")
    assert hits.OBJECTID.notna().all(), "Original OD grid position has no community"
    assert hits.Tid.is_unique, "Original OD grid position has ambiguous community"
    assert set(hits.Tid) == set(tids), "An original OD grid is missing"
    assignments = hits[["Tid", "OBJECTID"]].rename(columns={"OBJECTID": "community_id"})
    assignments["community_id"] = assignments.community_id.astype(int)
    return assignments.set_index("Tid").community_id.to_dict(), assignments


def assign_hospital_destination_communities(grid_polygons, community, end_tids):
    hospitals = gpd.read_file(HOSP).to_crs(CRS)[["Uid", "Name", "geometry"]]
    hospitals = hospitals.drop_duplicates("Uid").copy()
    assert hospitals.Uid.notna().all() and hospitals.Uid.is_unique
    hospitals.geometry = hospitals.geometry.apply(polygonal)
    points = hospitals.copy()
    points.geometry = points.geometry.representative_point()
    owners = gpd.sjoin(points, community[["OBJECTID", "geometry"]],
                       how="left", predicate="intersects")
    assert owners.OBJECTID.notna().all(), "Hospital administrative community is missing"
    assert owners.Uid.is_unique, "Hospital administrative community is ambiguous"
    owners = owners[["Uid", "OBJECTID"]].rename(columns={"OBJECTID": "community_id"})
    owners["community_id"] = owners.community_id.astype(int)

    destination_points = grid_polygons.loc[grid_polygons.Tid.isin(end_tids)].copy()
    destination_points.geometry = destination_points.geometry.centroid
    hits = gpd.sjoin(destination_points, hospitals,
                     how="inner", predicate="intersects")
    hits = hits[["Tid", "Uid", "Name"]].drop_duplicates(["Tid", "Uid"])
    hits = hits.merge(owners, on="Uid", validate="many_to_one")
    counts = (hits.groupby(["Tid", "community_id"]).Uid.nunique()
              .rename("n_hospital_aoi").reset_index())
    counts["n_communities"] = counts.groupby("Tid").community_id.transform("size")
    # Overlapping hospitals with different owners do not establish a unique visit.
    counts = counts.loc[counts.n_communities.eq(1)].copy()
    counts["probability"] = 1.0
    counts["assignment_basis"] = "original_destination_centroid_inside_hospital_aoi"
    counts = counts.rename(columns={"Tid": "end_tid"})
    destinations = {int(tid): list(zip(group.community_id.astype(int), group.probability))
                    for tid, group in counts.groupby("end_tid", sort=False)}
    assert set(counts.community_id).issubset(set(community.OBJECTID))
    return destinations, counts


def split_with_hospital_destination(grouped, hospital, target, residence, ids, attraction):
    matrix = np.zeros((len(ids), len(ids)), dtype=np.int64)
    index = {int(community_id): position for position, community_id in enumerate(ids)}
    hospital_mass = original_position_mass = 0
    for row in grouped.itertuples(index=False):
        source = int(row.source_community_id)
        assert source in index, "Original OD origin is missing"
        tid = int(row.end_tid)
        if tid in hospital:
            candidates = hospital[tid]
            hospital_mass += int(row.count)
        else:
            assert tid in target and len(target[tid]) == 1, "Original OD destination is missing"
            candidates = [(target[tid][0], 1.0)]
            original_position_mass += int(row.count)
        assert len(candidates) == 1 and candidates[0][1] == 1.0
        destination = int(candidates[0][0])
        assert destination in index
        count = int(row.count)
        matrix[index[source], index[destination]] += count
    audit = {"assigned_mass": int(matrix.sum()),
             "original_grid_position_mass": original_position_mass,
             "hospital_aoi_community_mass": hospital_mass,
             "missing_source_mass": 0, "missing_target_mass": 0}
    assert audit["assigned_mass"] == int(grouped["count"].sum()), "OD person trips were lost"
    expected = grouped.groupby("source_community_id")["count"].sum().reindex(ids, fill_value=0)
    assert np.array_equal(matrix.sum(axis=1), expected.to_numpy(dtype=np.int64))
    assert original_position_mass + hospital_mass == audit["assigned_mass"]
    return matrix, audit


def overlap_weights(grid_polygons, community, tids):
    """Share of each 500 m grid falling in each community (area weights).

    A grid touching no community polygon goes to the community nearest to its centroid.
    """
    cells = grid_polygons.loc[grid_polygons.Tid.isin(tids), ["Tid", "geometry"]].copy()
    pieces = gpd.overlay(cells, community[["OBJECTID", "geometry"]], how="intersection", keep_geom_type=True)
    pieces["area"] = pieces.geometry.area
    pieces = pieces.loc[pieces.area > 0]
    pieces["weight"] = pieces.area / pieces.groupby("Tid").area.transform("sum")
    weights = {int(tid): list(zip(group.OBJECTID.astype(int), group.weight))
               for tid, group in pieces.groupby("Tid", sort=False)}
    missing = sorted(set(int(t) for t in tids) - set(weights))
    if missing:
        centres = cells.loc[cells.Tid.isin(missing)].copy()
        centres.geometry = centres.geometry.centroid
        nearest = gpd.sjoin_nearest(centres, community[["OBJECTID", "geometry"]], how="left")
        for tid, cid in zip(nearest.Tid, nearest.OBJECTID):
            weights.setdefault(int(tid), [(int(cid), 1.0)])
    return weights, missing


def population_origin_weights(grid_polygons, community, nodes, tids):
    """Home grids split by the resident population of the 250 m grids inside them.

    Grids without resident population fall back to area weights.
    """
    cells = gpd.read_file(GRID250).to_crs(CRS)[["Tid", "SQCODE", "联通人", "geometry"]]
    cells.geometry = cells.geometry.centroid
    cells["SQCODE"] = cells.SQCODE.astype(str)
    id_by_code = dict(zip(nodes.SQCODE.astype(str), nodes.OBJECTID.astype(int)))
    cells["community_id"] = cells.SQCODE.map(id_by_code)
    cells = cells.dropna(subset=["community_id"])
    big = grid_polygons.loc[grid_polygons.Tid.isin(tids), ["Tid", "geometry"]]
    joined = gpd.sjoin(cells, big, how="inner", predicate="within", lsuffix="250", rsuffix="500")
    pop = joined.groupby(["Tid_500", "community_id"])["联通人"].sum().rename("population").reset_index()
    pop = pop.loc[pop.population > 0]
    pop["weight"] = pop.population / pop.groupby("Tid_500").population.transform("sum")
    weights = {int(tid): list(zip(group.community_id.astype(int), group.weight))
               for tid, group in pop.groupby("Tid_500", sort=False)}
    rest = set(int(t) for t in tids) - set(weights)
    fallback, missing = overlap_weights(grid_polygons, community, rest) if rest else ({}, [])
    weights.update(fallback)
    return weights, {"population_grids": len(set(int(t) for t in tids)) - len(rest),
                     "area_fallback_grids": len(rest), "nearest_fallback_grids": len(missing)}


HOSPITAL_LEVELS = ("二级", "三级")
HOSPITAL_NAME_PATTERN = "中医院|中医医院|妇幼保健院|儿童医院"


def target_hospitals(hospitals):
    """Hospitals with an explicit level 2 or 3: general, TCM and children's hospitals, and maternal and
    child health hospitals by name. Entries whose name is not a hospital are left out."""
    level = hospitals["列1"].isin(HOSPITAL_LEVELS)
    kind = hospitals["AOI子类"].eq("综合医院") | hospitals["Name"].str.contains(HOSPITAL_NAME_PATTERN, regex=True)
    named = hospitals["Name"].str.contains("医院|保健院", regex=True)
    return hospitals.loc[level & kind & named]


def hospital_nearest_weights(grid_polygons, community, tids, max_distance=500.0):
    """Destination grids within max_distance of a target hospital go to the nearest hospital's community.

    The OD grid is only a location proxy; what matters is that a visit near a hospital is
    attributed to the life circle (community) the hospital belongs to.
    """
    hospitals = target_hospitals(gpd.read_file(HOSP).to_crs(CRS)).drop_duplicates("Uid").copy()
    hospitals.geometry = hospitals.geometry.apply(polygonal).representative_point()
    owners = gpd.sjoin(hospitals[["Uid", "Name", "geometry"]], community[["OBJECTID", "geometry"]],
                       how="left", predicate="intersects").drop_duplicates("Uid")
    owners = owners.rename(columns={"OBJECTID": "community_id"})
    cells = grid_polygons.loc[grid_polygons.Tid.isin(tids), ["Tid", "geometry"]].copy()
    near = gpd.sjoin_nearest(cells, owners[["Uid", "Name", "community_id", "geometry"]], how="inner",
                             max_distance=max(max_distance, 1e-6), distance_col="distance")
    near = near.sort_values(["Tid", "distance", "Uid"]).drop_duplicates("Tid")
    weights = {int(t): [(int(c), 1.0)] for t, c in zip(near.Tid, near.community_id)}
    return weights, near[["Tid", "Uid", "Name", "community_id", "distance"]]


def supply_hospitals():
    """The SH/TH hospitals of the supply indicators: hospital_beds with an explicit level 2 or 3 and
    institution type A100 (general), A210 (TCM) or A519 (children's), or a maternal and child health
    hospital by name. Returns points with Uid and Name."""
    beds = gpd.read_file(HOSP_BEDS).to_crs(CRS)
    allowed = beds["INST_TYPE"].isin(["A100", "A210", "A519"])
    maternal = beds["HOSP_NAME"].astype("string").str.contains("妇幼保健院", na=False)
    beds = beds.loc[beds["HOSP_GRD"].isin(["二级", "三级"]) & (allowed | maternal)].copy()
    beds["Uid"] = beds["ORG_CODE"].astype(str) + "_" + beds.index.astype(str)
    beds["Name"] = beds["HOSP_NAME"].astype(str)
    beds["level"] = beds["HOSP_GRD"].astype(str)
    return beds[["Uid", "Name", "level", "geometry"]]


def hospital_grid_weights(grid_polygons, community, tids, max_distance=500.0):
    """One destination grid per supply hospital (hospital_beds, SH/TH rule), attributed to the
    community holding the hospital point.

    The grid holding the hospital point is used; when that grid never appears as an OD
    destination, the nearest destination grid within max_distance stands in for it. Several
    hospitals sharing a grid split it equally. Other destination grids keep their area split.
    """
    hospitals = supply_hospitals()
    owners = gpd.sjoin(hospitals[["Uid", "Name", "geometry"]], community[["OBJECTID", "geometry"]],
                       how="inner", predicate="intersects").drop_duplicates("Uid")
    owners = owners.rename(columns={"OBJECTID": "community_id"})[["Uid", "Name", "community_id", "geometry"]]
    owners = owners.reset_index(drop=True)
    destinations = grid_polygons.loc[grid_polygons.Tid.isin(tids), ["Tid", "geometry"]].reset_index(drop=True)
    own_grid = gpd.sjoin(owners, destinations, how="inner", predicate="within")[["Uid", "Tid"]].assign(distance=0.0)
    rest = owners.loc[~owners.Uid.isin(own_grid.Uid)].reset_index(drop=True)
    nearest = gpd.sjoin_nearest(rest, destinations, how="inner", max_distance=max_distance,
                                distance_col="distance")[["Uid", "Tid", "distance"]]
    nearest = nearest.sort_values(["Uid", "distance", "Tid"]).drop_duplicates("Uid")
    pairs = pd.concat([own_grid, nearest], ignore_index=True).merge(owners[["Uid", "Name", "community_id"]], on="Uid")
    share = pairs.groupby(["Tid", "community_id"]).size().rename("hospitals").reset_index()
    share["weight"] = share.hospitals / share.groupby("Tid").hospitals.transform("sum")
    weights = {int(t): list(zip(g.community_id.astype(int), g.weight)) for t, g in share.groupby("Tid", sort=False)}
    unmatched = owners.loc[~owners.Uid.isin(pairs.Uid), "Name"].tolist()
    return weights, pairs, unmatched


def hospital_overlap_weights(grid_polygons, community, tids, rule="intersect", min_coverage=0.0):
    """Destination grids matched to hospitals go to the hospital's community.

    rule "point": the grid contains the hospital's point (AOI representative point).

    rule "intersect": the AOIs cover at least min_coverage of the grid area (any positive
    overlap when min_coverage is 0); several hospitals split the grid by their overlapping AOI
    area. rule "centroid": the grid centroid lies in one AOI only (the earlier rule).
    """
    hospitals = gpd.read_file(HOSP).to_crs(CRS)[["Uid", "Name", "geometry"]].drop_duplicates("Uid").copy()
    hospitals.geometry = hospitals.geometry.apply(polygonal)
    points = hospitals.copy()
    points.geometry = points.geometry.representative_point()
    owners = gpd.sjoin(points, community[["OBJECTID", "geometry"]], how="left", predicate="intersects")
    owners = owners.drop_duplicates("Uid")[["Uid", "OBJECTID"]].rename(columns={"OBJECTID": "community_id"})
    owners["community_id"] = owners.community_id.astype(int)
    cells = grid_polygons.loc[grid_polygons.Tid.isin(tids), ["Tid", "geometry"]].copy()
    if rule == "point":
        # Each hospital is one location: the destination grid holding the hospital point goes to the
        # hospital's community; several hospitals in one grid split it equally.
        hits = gpd.sjoin(points[["Uid", "geometry"]], cells, how="inner", predicate="within")
        hits = hits[["Tid", "Uid"]].drop_duplicates().merge(owners, on="Uid")
        share = hits.groupby(["Tid", "community_id"]).size().rename("area").reset_index()
        share["weight"] = share.area / share.groupby("Tid").area.transform("sum")
        weights = {int(tid): list(zip(group.community_id.astype(int), group.weight))
                   for tid, group in share.groupby("Tid", sort=False)}
        return weights, share
    if rule == "centroid":
        cells.geometry = cells.geometry.centroid
        hits = gpd.sjoin(cells, hospitals, how="inner", predicate="intersects")
        hits = hits[["Tid", "Uid"]].drop_duplicates().merge(owners, on="Uid")
        hits["area"] = 1.0
    else:
        hits = gpd.overlay(cells, hospitals, how="intersection", keep_geom_type=True)
        hits["area"] = hits.geometry.area
        hits = hits.loc[hits.area > 0, ["Tid", "Uid", "area"]].merge(owners, on="Uid")
    share = hits.groupby(["Tid", "community_id"]).area.sum().rename("area").reset_index()
    if rule != "centroid" and min_coverage > 0:
        grid_area = cells.set_index("Tid").geometry.area
        covered = share.groupby("Tid").area.sum() / grid_area.reindex(share.Tid.unique())
        share = share.loc[share.Tid.isin(covered.index[covered >= min_coverage])]
    if rule == "centroid":
        share = share.loc[share.groupby("Tid").community_id.transform("size").eq(1)]
    share["weight"] = share.area / share.groupby("Tid").area.transform("sum")
    weights = {int(tid): list(zip(group.community_id.astype(int), group.weight))
               for tid, group in share.groupby("Tid", sort=False)}
    return weights, share


def circle_members(grid_polygons, circles, tids):
    """(destination grid, community) pairs whose grid centroid lies inside the community's 15-minute circle."""
    centres = grid_polygons.loc[grid_polygons.Tid.isin(tids), ["Tid", "geometry"]].copy()
    centres.geometry = centres.geometry.centroid
    hits = gpd.sjoin(centres, circles, how="inner", predicate="within")
    return set(zip(hits.Tid.astype(int), hits.objectid.astype(int)))


def allocate_matrix(od, origin_weights, destination_weights, ids, inside=None):
    """Split each grid OD record over origin and destination communities, then round to integers.

    inside: optional set of (destination grid, community) pairs; a record whose destination
    grid lies inside its origin community's own circle stays internal (origin to itself).
    Rounding uses largest remainders, first on community row totals and then within rows,
    so the total and the rounded row totals are exact.
    """
    o = pd.DataFrame([(t, c, w) for t, items in origin_weights.items() for c, w in items],
                     columns=["home_tid", "source", "wo"])
    d = pd.DataFrame([(t, c, w) for t, items in destination_weights.items() for c, w in items],
                     columns=["end_tid", "target", "wd"])
    flows = od.merge(o, on="home_tid", how="left")
    assert flows.source.notna().all(), "A home grid has no community weight"
    flows["value"] = flows["count"] * flows.wo
    if inside:
        key = pd.MultiIndex.from_frame(flows[["end_tid", "source"]].astype(int))
        flows["inside"] = key.isin(list(inside))
    else:
        flows["inside"] = False
    internal = flows.loc[flows.inside].groupby("source").value.sum()
    outside = flows.loc[~flows.inside].merge(d, on="end_tid", how="left")
    assert outside.target.notna().all(), "A destination grid has no community weight"
    outside["value"] = outside.value * outside.wd
    frac = np.array(outside.groupby(["source", "target"]).value.sum().unstack(fill_value=0.0)
                    .reindex(index=ids, columns=ids, fill_value=0.0).to_numpy(), dtype=float)
    position = {int(c): i for i, c in enumerate(ids)}
    for c, v in internal.items():
        frac[position[int(c)], position[int(c)]] += v
    total = int(od["count"].sum())
    rows = largest_remainder(frac.sum(1), total)
    matrix = np.zeros_like(frac, dtype=np.int64)
    for i in range(len(ids)):
        if rows[i]:
            matrix[i] = largest_remainder(frac[i], int(rows[i]))
    assert int(matrix.sum()) == total, "OD person trips were lost"
    return matrix, frac


def largest_remainder(values, total):
    values = np.asarray(values, dtype=float)
    scaled = values * (total / values.sum()) if values.sum() > 0 else values
    base = np.floor(scaled).astype(np.int64)
    rest = int(total - base.sum())
    if rest > 0:
        base[np.argsort(-(scaled - base), kind="stable")[:rest]] += 1
    return base


def main(adjacency_only=False, allocation="overlap", hospital_rule="hospital-grid", min_coverage=0.10,
         internal="circle", hospital_distance=500.0):
    RESULT.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(RESULT / "node_order.csv", encoding="utf-8-sig")
    ids = nodes.OBJECTID.astype(int).to_numpy()
    assert len(ids) == len(set(ids)) == 666

    scores = pd.read_excel(RA_FILE)[["community_id", "RA_CHC", "RA_SH", "RA_TH"]]
    assert scores.community_id.is_unique and set(scores.community_id) == set(ids)
    ra = scores[["RA_CHC", "RA_SH", "RA_TH"]].to_numpy(dtype=float)
    assert np.isfinite(ra).all() and (ra >= 0).all()
    attraction = dict(zip(scores.community_id.astype(int), ra.sum(axis=1)))

    community = gpd.read_file(COMM).to_crs(CRS)[["SQCODE", "geometry"]]
    community["SQCODE"] = community.SQCODE.astype(str)
    community = community.merge(nodes[["SQCODE", "OBJECTID"]].assign(
        SQCODE=lambda d: d.SQCODE.astype(str)), on="SQCODE", validate="one_to_one")
    assert community.OBJECTID.is_unique and set(community.OBJECTID) == set(ids)
    community.geometry = community.geometry.apply(polygonal)
    borders = community.copy()
    borders.geometry = borders.geometry.boundary

    grid_polygons = gpd.read_file(GRID).to_crs(CRS)[["Tid", "geometry"]]
    grids = grid_polygons.copy()
    grids.geometry = grids.geometry.centroid
    assert grids.Tid.is_unique
    if not adjacency_only:
        od = pd.read_csv(RESULT / "outpatient_grid_OD_home_tid.csv")
        assert od[["home_tid", "end_tid"]].notna().all().all()
        assert (od["count"] >= 0).all() and np.equal(od["count"], od["count"].astype(int)).all()
        od = od[["home_tid", "end_tid", "count"]]
        if allocation == "overlap":
            origin_w, origin_audit = population_origin_weights(grid_polygons, community, nodes, set(od.home_tid))
            destination_w, missing_dest = overlap_weights(grid_polygons, community, set(od.end_tid))
            if hospital_rule == "none":
                hospital_w, hospital_share = {}, None
            elif hospital_rule == "hospital-grid":
                hospital_w, hospital_share, unmatched_hospitals = hospital_grid_weights(
                    grid_polygons, community, set(od.end_tid), hospital_distance)
                hospital_share.to_csv(RESULT / "hospital_destination_grid_match.csv", index=False,
                                      encoding="utf-8-sig")
            elif hospital_rule == "nearest":
                hospital_w, hospital_share = hospital_nearest_weights(grid_polygons, community, set(od.end_tid),
                                                                      hospital_distance)
                hospital_share.to_csv(RESULT / "hospital_destination_nearest_match.csv", index=False,
                                      encoding="utf-8-sig")
            else:
                hospital_w, hospital_share = hospital_overlap_weights(grid_polygons, community, set(od.end_tid),
                                                                      hospital_rule, min_coverage)
            destination_w.update(hospital_w)
            destination_mass = od.groupby("end_tid")["count"].sum()
            pd.DataFrame([(t, c, w) for t, items in origin_w.items() for c, w in items],
                         columns=["home_tid", "community_id", "weight"]).to_csv(
                RESULT / "origin_grid_population_weights.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame([(t, c, w, t in hospital_w) for t, items in destination_w.items() for c, w in items],
                         columns=["end_tid", "community_id", "weight", "hospital_aoi"]).to_csv(
                RESULT / "destination_grid_weights.csv", index=False, encoding="utf-8-sig")
            overlap_audit = {"origin_basis": "500m_grid_split_by_250m_resident_population", **origin_audit,
                             "destination_basis": "500m_grid_split_by_area" +
                                                  ("" if hospital_rule == "none" else f"_with_hospital_aoi_{hospital_rule}"),
                             "hospital_min_coverage": min_coverage if hospital_rule == "intersect" else None,
                             "hospital_max_distance_m": hospital_distance if hospital_rule in ("nearest", "hospital-grid") else None,
                             "hospital_set": ("hospital_beds SH/TH: level 2/3 and A100/A210/A519 or maternal and child health by name"
                                              if hospital_rule == "hospital-grid" else
                                              "医院_床位 level 2/3 general, TCM, maternal and child health, children's"
                                              if hospital_rule == "nearest" else "all hospitals in HOSP"),
                             "hospital_file": str(HOSP_BEDS) if hospital_rule == "hospital-grid" else str(HOSP),
                             "hospitals_without_destination_grid": (unmatched_hospitals
                                                                    if hospital_rule == "hospital-grid" else None),
                             "internal_basis": ("destination_grid_centroid_inside_origin_community_circle"
                                                if internal == "circle" else "destination_grid_overlap_with_origin_community"),
                             "destination_nearest_fallback_grids": len(missing_dest),
                             "hospital_destination_grids": len(hospital_w),
                             "hospital_destination_mass": int(destination_mass.reindex(list(hospital_w)).fillna(0).sum())}
        grid_communities, grid_assignment = assign_grid_position_communities(
            grid_polygons, community, set(od.home_tid) | set(od.end_tid))
        assigned_od = od[["end_tid", "count"]].copy()
        assigned_od.insert(0, "source_community_id", od.home_tid.map(grid_communities).astype(int))
        target = {int(tid): [int(grid_communities[tid])] for tid in set(od.end_tid)}
        residence = {}
        grid_assignment["is_origin"] = grid_assignment.Tid.isin(od.home_tid)
        grid_assignment["is_destination"] = grid_assignment.Tid.isin(od.end_tid)
        grid_assignment.to_csv(RESULT / "od_grid_community_assignment.csv",
                               index=False, encoding="utf-8-sig")
        hospital, hospital_assignment = assign_hospital_destination_communities(
            grid_polygons, community, set(od.end_tid))
        destination_mass = od.groupby("end_tid")["count"].sum()
        hospital_assignment["destination_grid_records"] = (
            hospital_assignment.end_tid.map(destination_mass).astype(int))
        hospital_assignment["original_grid_community_id"] = (
            hospital_assignment.end_tid.map(grid_communities).astype(int))
        hospital_assignment.to_csv(RESULT / "hospital_destination_community_assignment.csv",
                                   index=False, encoding="utf-8-sig")

    adjacency = {}
    flow_audit = {}
    for mode, label in (("foot", "walk"), ("bike", "bike")):
        circles = read_circles(mode, set(ids))
        if not adjacency_only and allocation == "overlap":
            inside = circle_members(grid_polygons, circles, set(od.end_tid)) if internal == "circle" else None
            matrix, _ = allocate_matrix(od, origin_w, destination_w, ids, inside)
            inflow = matrix.sum(0) - np.diag(matrix)
            audit = {"assigned_mass": int(matrix.sum()), "internal_mass": int(np.trace(matrix)),
                     "communities_zero_inflow": int((inflow == 0).sum()),
                     "communities_zero_outflow": int((matrix.sum(1) == 0).sum())}
            flow_audit[label] = {"input_od_mass": int(od["count"].sum()), **audit, **overlap_audit,
                                 "source_community_totals_conserved": "rounded_from_split_shares"}
            frame = pd.DataFrame(matrix, index=ids, columns=ids)
            frame.index.name = "source_community_id"
            frame.to_csv(RESULT / f"medical_network_15min_{label}.csv", encoding="utf-8-sig")
        elif not adjacency_only:
            matrix, audit = split_with_hospital_destination(
                assigned_od, hospital, target, residence, ids, attraction)
            assert np.issubdtype(matrix.dtype, np.integer) and (matrix >= 0).all()
            flow_audit[label] = {"input_od_mass": int(od["count"].sum()), **audit,
                                 "source_community_totals_conserved": True,
                                 "origin_basis": "original_500m_grid_centroid",
                                 "destination_basis": "original_500m_grid_centroid_with_hospital_aoi_correction",
                                 "hospital_destination_grids": len(hospital),
                                 "hospital_corrected_grids": int(hospital_assignment.loc[
                                     hospital_assignment.community_id.ne(
                                         hospital_assignment.original_grid_community_id), "end_tid"].nunique()),
                                 "hospital_corrected_mass": int(hospital_assignment.loc[
                                     hospital_assignment.community_id.ne(
                                         hospital_assignment.original_grid_community_id), "destination_grid_records"].sum())}
            frame = pd.DataFrame(matrix, index=ids, columns=ids)
            frame.index.name = "source_community_id"
            frame.to_csv(RESULT / f"medical_network_15min_{label}.csv", encoding="utf-8-sig")

        pairs = gpd.sjoin(circles, borders, how="inner", predicate="intersects")
        pairs = pairs.loc[pairs.objectid.ne(pairs.OBJECTID), ["objectid", "OBJECTID"]]
        adjacency[label] = set(map(tuple, pairs.drop_duplicates().astype(int).to_numpy()))
        if not adjacency_only:
            print(label, "circles", len(circles), "flow", int(matrix.sum()),
                  "diagonal", int(np.trace(matrix)), "edges", int((matrix > 0).sum()),
                  "adjacent", len(adjacency[label]), "audit", audit, flush=True)
        else:
            print(label, "adjacent", len(adjacency[label]), flush=True)

    all_pairs = sorted(adjacency["walk"] | adjacency["bike"])
    table = pd.DataFrame(all_pairs, columns=["source_community_id", "target_community_id"])
    table["walk_adjacent"] = [int(pair in adjacency["walk"]) for pair in all_pairs]
    table["bike_adjacent"] = [int(pair in adjacency["bike"]) for pair in all_pairs]
    metadata = nodes[["OBJECTID", "SQCODE", "SQNAME"]].copy()
    metadata.columns = ["source_community_id", "source_SQCODE", "source_name"]
    table = metadata.merge(table, on="source_community_id", how="left", validate="one_to_many")
    destination = nodes[["OBJECTID", "SQCODE", "SQNAME"]].copy()
    destination.columns = ["target_community_id", "target_SQCODE", "target_name"]
    table = table.merge(destination, on="target_community_id", how="left", validate="many_to_one")
    table["target_community_id"] = table["target_community_id"].astype("Int64")
    table["target_SQCODE"] = table["target_SQCODE"].astype("Int64")
    table[["walk_adjacent", "bike_adjacent"]] = table[["walk_adjacent", "bike_adjacent"]].fillna(0).astype(int)
    table = table[["source_community_id", "source_SQCODE", "source_name",
                   "target_community_id", "target_SQCODE", "target_name",
                   "walk_adjacent", "bike_adjacent"]]
    adjacency_path = RESULT / "adjacency_15min_walk_bike.csv"
    if adjacency_only or not adjacency_path.exists():
        table.to_csv(adjacency_path, index=False, encoding="utf-8-sig")
    print("adjacency_rows", len(table), "communities", table.source_community_id.nunique(), flush=True)
    if not adjacency_only:
        (RESULT / "medical_network_15min_flow_audit.json").write_text(
            json.dumps(flow_audit, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjacency-only", action="store_true")
    parser.add_argument("--allocation", choices=["overlap", "centroid"], default="overlap",
                        help="overlap: split grids by resident population (origins) and area (destinations); "
                             "centroid: assign each grid by its centroid (earlier rule)")
    parser.add_argument("--hospital-rule", choices=["hospital-grid", "nearest", "point", "intersect", "centroid", "none"],
                        default="hospital-grid",
                        help="hospital-grid: one destination grid per target hospital (its own grid, else the nearest "
                             "destination grid within --hospital-distance) goes to the hospital's community; "
                             "nearest: every destination grid within --hospital-distance of a target hospital")
    parser.add_argument("--hospital-distance", type=float, default=500.0, help="metres")
    parser.add_argument("--hospital-min-coverage", type=float, default=0.10,
                        help="intersect rule: share of the grid area the hospital AOIs must cover")
    parser.add_argument("--internal", choices=["circle", "overlap"], default="circle",
                        help="circle: destination grid centroid inside the origin community's own 15-minute circle "
                             "counts as internal flow (mode specific)")
    args = parser.parse_args()
    main(adjacency_only=args.adjacency_only, allocation=args.allocation, hospital_rule=args.hospital_rule,
         min_coverage=args.hospital_min_coverage, internal=args.internal, hospital_distance=args.hospital_distance)

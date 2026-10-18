"""把腾讯地图坐标拾取器（https://lbs.qq.com/getPoint）取得的社康点位写入 facility_location_overrides.csv。

输入 facility_points_tencent.csv：facility_id、tencent_lng、tencent_lat、坐标系（GCJ02 或 WGS84，空为 GCJ02）、依据（可空）。
GCJ02 坐标转为 WGS84（与供给表点位坐标系一致），WGS84 坐标直接使用，写入 overrides，处理记为“腾讯地图拾取点位”：
新点位所在 250 m 网格与设施原归属社区相交时仍归原归属社区，否则归新点位所在社区。
同一 facility_id 已在 overrides 中的行被替换。
"""
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "supply input"
TENCENT = HERE / "facility_points_tencent.csv"
OVERRIDE = HERE / "facility_location_overrides.csv"
FAC_COMM = HERE.parent / "step2_supply/results/facility_community.csv"
CRS = 4547
A, EE = 6378245.0, 0.00669342162296594323


def _lat(x, y):
    r = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    r += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    r += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    r += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return r


def _lng(x, y):
    r = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    r += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    r += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    r += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return r


def wgs_to_gcj(lng, lat):
    dlat, dlng = _lat(lng - 105.0, lat - 35.0), _lng(lng - 105.0, lat - 35.0)
    rad = lat / 180.0 * math.pi
    magic = 1 - EE * math.sin(rad) ** 2
    sq = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((A * (1 - EE)) / (magic * sq) * math.pi)
    dlng = (dlng * 180.0) / (A / sq * math.cos(rad) * math.pi)
    return lng + dlng, lat + dlat


def gcj_to_wgs(lng, lat):
    """迭代反解 GCJ-02 → WGS84，精度优于 0.1 m。"""
    x, y = lng, lat
    for _ in range(30):
        gx, gy = wgs_to_gcj(x, y)
        dx, dy = gx - lng, gy - lat
        x, y = x - dx, y - dy
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            break
    return x, y


t = pd.read_csv(TENCENT, encoding="utf-8-sig").dropna(subset=["tencent_lng", "tencent_lat"])
crs_in = t.get("坐标系", pd.Series("GCJ02", index=t.index)).fillna("GCJ02").str.upper()
w = [gcj_to_wgs(a, b) if c == "GCJ02" else (a, b) for a, b, c in zip(t.tencent_lng, t.tencent_lat, crs_in)]
t["坐标系"] = crs_in
t["longitude"], t["latitude"] = [p[0] for p in w], [p[1] for p in w]
fc = pd.read_csv(FAC_COMM, dtype={"SQCODE": str}).set_index("facility_id")
ovr = pd.read_csv(OVERRIDE, encoding="utf-8-sig", dtype={"SQCODE": str})
own = ovr.set_index("facility_id").SQCODE.combine_first(fc.SQCODE)  # 设施当前归属社区

comm = gpd.read_file(SRC / "深圳社区objectid.shp")[["SQCODE", "geometry"]].to_crs(CRS)
comm["SQCODE"] = comm.SQCODE.astype(str)
comm["geometry"] = comm.geometry.make_valid()
g250 = gpd.read_file(SRC / "深圳250m网格_人口_所在社区.shp")[["Tid", "geometry"]].to_crs(CRS)
pt = gpd.GeoDataFrame(t[["facility_id"]], geometry=gpd.points_from_xy(t.longitude, t.latitude), crs=4326).to_crs(CRS)
cell = gpd.sjoin(pt, g250, predicate="within").set_index("facility_id").Tid
in_comm = gpd.sjoin(pt, comm, predicate="within").set_index("facility_id").SQCODE
sqname = pd.read_csv(SRC / "node_order.csv", encoding="utf-8-sig", dtype={"SQCODE": str}).set_index("SQCODE").SQNAME
cg = comm.dissolve("SQCODE").geometry
rows = []
for r in t.itertuples():
    s0 = own[r.facility_id]
    touch = g250.set_index("Tid").geometry[cell[r.facility_id]].intersection(cg[s0]).area > 0 if r.facility_id in cell else False
    s = s0 if touch else in_comm.get(r.facility_id, s0)
    rows.append({"facility_id": r.facility_id, "facility_name": fc.facility_name.get(r.facility_id), "处理": "腾讯地图拾取点位",
                 "SQCODE": s, "SQNAME": sqname.get(s), "longitude": r.longitude, "latitude": r.latitude,
                 "依据": (r.依据 if isinstance(getattr(r, "依据", None), str) else "") +
                 f"（lbs.qq.com/getPoint {r.tencent_lng},{r.tencent_lat}，{'GCJ-02 转 WGS84' if r.坐标系 == 'GCJ02' else '按 WGS84 使用'}；"
                 + ("新点位网格与原归属社区相交，归属不变）" if touch else f"新点位在{sqname.get(s)}社区，归该社区）")})
new = pd.DataFrame(rows)
ovr = pd.concat([ovr[~ovr.facility_id.isin(new.facility_id)], new], ignore_index=True)
ovr.to_csv(OVERRIDE, index=False, encoding="utf-8-sig")
print(new[["facility_id", "facility_name", "SQNAME", "longitude", "latitude"]].to_string(index=False))

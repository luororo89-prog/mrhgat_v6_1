"""必要分配后没有到达人次的设施，核查点位准确性（OSM Nominatim）。

必要分配：到达 500 m 网格内 4 个 250 m 网格有设施时按吸引力分给这些设施；没有时分给周围一圈 12 个 250 m 网格的设施。
取 results/discount_<折扣>/grid250_destination_allocation.csv 中非补给的分配，所在 250 m 网格没有到达人次的设施逐个核查：
1. 社康按站点名称（去掉“深圳市”与举办医院前缀）查 OSM 的医疗设施；再按地址中的地物（小区、大厦、花园、村等）查；
   社康点位取 facility_grid250/facility_location_final.csv，地址与所在区取 2018年深圳CHC_912个点位_XY_WGS84.xlsx 的定位依据地址；
   医院按机构名称查，位置取 AOI 代表点。只取显示名称含设施所在区的结果。
2. 名称命中的医疗设施距点位超过 150 m，或地址地物距点位超过 300 m，列为待核；
   结果写入 results/check_zero_arrival_points/（zero_arrival_point_check.csv），查询结果缓存于 nominatim_cache.json。
第二个参数为 all 时核查全部社康（all_chc_point_check.csv、summary_all.json），用于估计社康点位整体准确性。
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
D = sys.argv[1] if len(sys.argv) > 1 else "0.5"
ALL = len(sys.argv) > 2 and sys.argv[2] == "all"
OUT = RES / "check_zero_arrival_points"
OUT.mkdir(exist_ok=True)
SRC = HERE.parent / "supply input"
CHC_XY = Path("/Users/luoxuan/Documents/博后三胎/outputs/coordinate_xy_wgs84_all_20261017/2018年深圳CHC_912个点位_XY_WGS84.xlsx")
CACHE = OUT / "nominatim_cache.json"
CRS = 4547
NAME_M, ADDR_M = 150, 300
UA = "mrhgat-research/1.0 (academic geocoding check)"
MED = {"clinic", "doctors", "hospital", "healthcare", "pharmacy"}

cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def search(q):
    if q not in cache:
        url = ("https://nominatim.openstreetmap.org/search?format=jsonv2&limit=8&countrycodes=cn"
               "&viewbox=113.7,22.85,114.65,22.4&bounded=1&q=" + urllib.parse.quote(q))
        try:
            cache[q] = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30))
        except Exception as e:  # 网络失败不写缓存
            print("查询失败", q, e)
            return []
        time.sleep(1.1)
    return cache[q]


fp = pd.read_csv(RES / f"discount_{D}/facility_grid250_attraction.csv", encoding="utf-8-sig", dtype={"SQCODE": str})
al = pd.read_csv(RES / f"discount_{D}/grid250_destination_allocation.csv", encoding="utf-8-sig")
nec = al[~al.rule.str.startswith("补给")]
got = set(fp.loc[fp.Tid_250.isin(nec.Tid_250), "facility_id"])
z = fp[~fp.facility_id.isin(got)].drop_duplicates("facility_id").copy()
if ALL:
    z = fp[fp.tier == "CHC"].drop_duplicates("facility_id").copy()
src = pd.read_excel(CHC_XY, header=None, usecols=[0, 2, 3, 6], names=["row", "所在区", "地址", "腾讯定位对象"])
src = src[pd.to_numeric(src.row, errors="coerce").notna()]
src["facility_id"] = "CHC_" + src.row.astype(int).astype(str)
z = z.merge(src[["facility_id", "所在区", "地址", "腾讯定位对象"]], on="facility_id", how="left")
xy = pd.read_csv(RES / "facility_grid250/facility_location_final.csv", encoding="utf-8-sig").set_index("facility_id")
aoi = gpd.read_file(SRC / "hospital_aoi_35.shp").to_crs(CRS)
rep = aoi.set_index("fac_name").geometry.make_valid().representative_point()
hv = pd.read_csv(HERE.parent / "step1_grid_community/results/hospital_aoi_added/hospital_external_validation.csv",
                 encoding="utf-8-sig")
haddr = hv.dropna(subset=["match_name"]).drop_duplicates("match_name").set_index("match_name")[["地址", "行政区划"]]
CJK = re.compile(r"[一-鿿]")


def stem(name):
    s = name.replace("深圳市", "")
    i = s.rfind("医院")
    s = s[i + 2:] if 0 <= i < len(s) - 4 else s
    s = re.sub(r"（.*?）|\(.*?\)", "", s)
    return s


def landmarks(addr):
    if not isinstance(addr, str):
        return []
    a = re.sub(r"^(广东省)?(深圳市)?([一-鿿]{1,4}区)?([一-鿿]{1,6}街道)?", "", addr)
    a = re.sub(r"（.*?）|\(.*?\)", " ", a)
    parts = re.split(r"[0-9０-９A-Za-z\-一二三四五六七八九十]+(?:栋|号|座|楼|层|铺|室|期|单元|区|幢|号楼)|[，,、；;\s]", a)
    keys = ("花园", "苑", "大厦", "村", "小区", "广场", "中心", "城", "园", "楼", "阁", "居", "府", "庭", "湾", "公寓", "新村", "工业")
    out = []
    for p in parts:
        p = p.strip("之的与和及")
        if len(CJK.findall(p)) >= 3 and any(k in p for k in keys) and p not in out:
            out.append(p)
    return out[:3]


rows = []
for r in z.itertuples():
    if r.tier == "CHC":
        pt = gpd.GeoSeries(gpd.points_from_xy([xy.loc[r.facility_id, "longitude"]], [xy.loc[r.facility_id, "latitude"]]),
                           crs=4326).to_crs(CRS).iloc[0]
        district, addr = r.所在区, r.地址
        qn = stem(r.facility_name)
    else:
        pt = rep[r.facility_name]
        district = haddr["行政区划"].get(r.facility_name, "")
        addr = haddr["地址"].get(r.facility_name, None)
        qn = r.facility_name.replace("深圳市", "")
    best = {"方式": "", "命中": "", "距点位_m": None, "lon": None, "lat": None}
    for kind, qs in (("名称", [qn]), ("地址地物", landmarks(addr))):
        for q in qs:
            for x in search(q):
                if isinstance(district, str) and district and district not in x["display_name"]:
                    continue
                if kind == "名称" and x.get("type") not in MED and x.get("category") not in ("amenity", "healthcare"):
                    continue
                g = gpd.GeoSeries(gpd.points_from_xy([float(x["lon"])], [float(x["lat"])]), crs=4326).to_crs(CRS).iloc[0]
                dist = round(g.distance(pt), 1)
                if best["方式"] == "" or (kind == best["方式"] and dist < best["距点位_m"]):
                    best = {"方式": kind, "命中": f"{q} → {x['display_name'][:60]}", "距点位_m": dist,
                            "lon": float(x["lon"]), "lat": float(x["lat"])}
            if best["方式"] == "名称":
                break
        if best["方式"]:
            break
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    flag = ("名称命中距点位远" if best["方式"] == "名称" and best["距点位_m"] > NAME_M else
            "地址地物距点位远" if best["方式"] == "地址地物" and best["距点位_m"] > ADDR_M else "")
    rows.append({"facility_id": r.facility_id, "facility_name": r.facility_name, "tier": r.tier, "SQCODE": r.SQCODE,
                 "Tid_250": r.Tid_250, "所在区": district, "地址": addr, "腾讯定位对象": getattr(r, "腾讯定位对象", None),
                 **best, "待核": flag,
                 "核查结论": "无法核实" if not best["方式"] and not flag else ("待核" if flag else "点位与OSM一致")})
res = pd.DataFrame(rows)
res.to_csv(OUT / ("all_chc_point_check.csv" if ALL else "zero_arrival_point_check.csv"), index=False, encoding="utf-8-sig")
summary = {"折扣档": D, ("核查的社康" if ALL else "必要分配后无到达人次的设施"): int(len(res)), "按层级": res.tier.value_counts().to_dict(),
           "核查结论": res["核查结论"].value_counts().to_dict(), "待核原因": res.loc[res["待核"] != "", "待核"].value_counts().to_dict()}
(OUT / ("summary_all.json" if ALL else "summary.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))

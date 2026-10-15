"""在深圳社区地图上显示新增 AOI 的 5 家医院：新增 AOI、供给表点位（高德 API 坐标）、所在社区，其余 35 个医院 AOI 作为参照。
输出 results/hospital_aoi_added/map_5_hospitals.html（Leaflet）。默认底图为社区边界与区界，可切换 Esri 街道地图与卫星影像。
地址栏 #1 至 #5 定位到对应医院，带 img 时切换卫星影像，例如 #2img。"""
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "supply input"
OUT = HERE / "results/hospital_aoi_added"
AOI = SRC / "hospital_aoi_35.shp"
FAC = SRC / "facility_supply_audit_2018_mobile_users.csv"
COMM = SRC / "深圳社区objectid.shp"
COMM_POP = SRC / "深圳各社区人口.shp"
DIST = SRC / "shenzhen_district_4547.shp"

INFO = {  # 2018 地址取外部门诊表，现状与核对结论取 2026-10-15 的查询
    "深圳市宝安区沙井人民医院": ("宝安区沙井街道沙井大街3号", "深圳市中西医结合医院（沙井大街3号、新沙路528号）",
                         "基本对应：AOI 为原沙井人民医院院区（OSM old_name），现地址仍含沙井大街3号；OSM 未标沙井大街，门牌未在图上核对"),
    "深圳市光明新区人民医院": ("宝安区公明镇松白路公明段339号", "西院区（原光明新区人民医院、公明人民医院，松白路4253号）",
                        "对应：西院区即原光明新区人民医院；2018 门牌（公明段339号）与现址（将石段4253号）不同，未能确认是否门牌调整"),
    "深圳市光明新区中心医院": ("光明新区光明街道华夏路39号", "东院区（华夏路39号）", "对应：2018 地址与东院区现址同为华夏路39号，AOI 紧贴华夏路"),
    "深圳市龙岗区第二人民医院": ("龙岗区布吉镇吉华路175号", "二院 2025-10-28 迁至创作路3号；原址现为第七人民医院（龙岭院区）",
                         "对应 2018 院址：AOI 内 OSM 门牌为吉华路175号"),
    "深圳市龙岗区第五人民医院": ("龙岗区平湖镇双拥街77号", "原址未变（1993 年起）", "对应：1993 年起在双拥街77号，AOI 距双拥街 20 m"),
}


def geo(gdf, tol=None):
    g = gdf.to_crs(4547)
    if tol:
        g["geometry"] = g.geometry.simplify(tol)
    return json.loads(g.to_crs(4326).to_json(drop_id=True))


def rnd(c):  # 坐标保留 6 位小数（约 0.1 m）
    return [rnd(x) for x in c] if isinstance(c[0], list) else [round(c[0], 6), round(c[1], 6)]


aoi = gpd.read_file(AOI)
fac = pd.read_csv(FAC).set_index("facility_name")
new = aoi[aoi.fac_name.isin(list(INFO))].copy()
new["no"] = range(1, len(new) + 1)
new["addr2018"] = new.fac_name.map(lambda n: INFO[n][0])
new["now"] = new.fac_name.map(lambda n: INFO[n][1])
new["verdict"] = new.fac_name.map(lambda n: INFO[n][2])
old = aoi[~aoi.fac_name.isin(list(INFO))][["fac_name", "tier", "geometry"]]
pts = new[["no", "fac_name"]].copy()
pts["lon"] = pts.fac_name.map(fac.longitude)
pts["lat"] = pts.fac_name.map(fac.latitude)

comm = gpd.read_file(COMM)[["SQCODE", "geometry"]]
comm["geometry"] = comm.geometry.make_valid()
comm = comm.merge(gpd.read_file(COMM_POP)[["SQCODE", "SQNAME"]], on="SQCODE", how="left")
host = new.groupby("sqcode").no.apply(lambda s: "、".join(map(str, s)))
comm["host"] = comm.SQCODE.map(host).fillna("")
dist = gpd.read_file(DIST)[["dt_name", "geometry"]]
lab = dist.to_crs(4547).explode(index_parts=False)
lab = lab.assign(a=lab.area).sort_values("a").groupby("dt_name").tail(1)
lab = gpd.GeoDataFrame(lab[["dt_name"]], geometry=lab.representative_point(), crs=4547).to_crs(4326)
dlab = [{"name": n, "lat": g.y, "lon": g.x} for n, g in zip(lab.dt_name, lab.geometry)]

cols = ["no", "fac_name", "aoi_src", "aoi_name", "outpat", "beds", "area_m2", "sqname", "addr2018", "now", "verdict",
        "geometry"]
data = {"new": geo(new[cols]), "old": geo(old), "pts": pts.drop(columns="fac_name").to_dict("records"),
        "comm": geo(comm, 5), "dist": geo(dist, 20), "dlab": dlab}
for layer in ("comm", "dist"):
    for f in data[layer]["features"]:
        f["geometry"]["coordinates"] = rnd(f["geometry"]["coordinates"])

html = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>新增医院 AOI</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
:root { color-scheme: light; --surface: #fcfcfb; --surface-2: #f1f0ec; --ink: #0b0b0b; --ink-2: #52514e;
  --line: #d9d8d2; --aoi: #2a78d6; --pt: #eb6834; --old: #8a8984; --comm: #b5b4ad; --dist: #52514e; --host: #e4e2da; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --surface: #1a1a19;
  --surface-2: #262624; --ink: #ffffff; --ink-2: #c3c2b7; --line: #3a3a37; --aoi: #3987e5; --pt: #d95926;
  --old: #9a998f; --comm: #55544f; --dist: #c3c2b7; --host: #34332f; } }
:root[data-theme="dark"] { color-scheme: dark; --surface: #1a1a19; --surface-2: #262624; --ink: #ffffff;
  --ink-2: #c3c2b7; --line: #3a3a37; --aoi: #3987e5; --pt: #d95926; --old: #9a998f; --comm: #55544f;
  --dist: #c3c2b7; --host: #34332f; }
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; background: var(--surface); color: var(--ink);
  font: 14px/1.5 "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif; }
.wrap { display: flex; height: 100%; }
aside { width: 360px; flex: none; overflow-y: auto; padding: 16px; border-right: 1px solid var(--line); }
#map { flex: 1; min-height: 360px; background: var(--surface); }
h1 { font-size: 17px; margin: 0 0 4px; }
.sub { color: var(--ink-2); font-size: 12.5px; margin: 0 0 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 12.5px; color: var(--ink-2); margin-bottom: 12px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 14px; height: 10px; border-radius: 2px; display: inline-block; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; box-shadow: 0 0 0 2px var(--surface); }
.card { border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; margin-bottom: 8px; cursor: pointer;
  background: var(--surface); }
.card:hover, .card.on { background: var(--surface-2); }
.card b { font-size: 14px; }
.num { display: inline-flex; width: 20px; height: 20px; border-radius: 50%; align-items: center; justify-content: center;
  background: var(--aoi); color: #fff; font-size: 12px; font-weight: 600; margin-right: 6px; }
.kv { font-size: 12.5px; color: var(--ink-2); margin-top: 4px; }
.kv em { font-style: normal; color: var(--ink); }
.lbl { background: var(--aoi); color: #fff; border: 0; border-radius: 10px; padding: 0 7px; font-weight: 600;
  box-shadow: 0 0 0 2px #fff; }
.lbl::before, .dlbl::before { display: none; }
.dlbl { background: transparent; border: 0; box-shadow: none; color: var(--ink-2); font-size: 12px; font-weight: 600;
  text-shadow: 0 0 3px var(--surface), 0 0 3px var(--surface); }
.leaflet-popup-content { font-size: 13px; line-height: 1.5; }
@media (max-width: 760px) { .wrap { flex-direction: column; } aside { width: 100%; max-height: 45%;
  border-right: 0; border-bottom: 1px solid var(--line); } }
</style>
</head>
<body>
<div class="wrap">
<aside>
  <h1>新增 AOI 的 5 家医院</h1>
  <p class="sub">底图为深圳 666 个社区与区界。点击卡片定位，右上角可换成街道地图或卫星影像。</p>
  <div class="legend">
    <span><i class="sw" style="background:var(--aoi);opacity:.55"></i>新增 AOI</span>
    <span><i class="dot" style="background:var(--pt)"></i>供给表点位（高德坐标，取坐标时间不详）</span>
    <span><i class="sw" style="background:var(--host);border:1px solid var(--comm)"></i>医院所在社区</span>
    <span><i class="sw" style="border:1.5px dashed var(--old)"></i>其余 35 个医院 AOI</span>
  </div>
  <div id="list"></div>
</aside>
<div id="map"></div>
</div>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const map = L.map("map", { zoomControl: true, zoomSnap: 0.25 });
map.fitBounds(L.geoJSON(D.dist).getBounds(), { padding: [20, 20] });
const blank = L.layerGroup();
const osm = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
  { maxZoom: 19, attribution: "Map &copy; Esri" });
const img = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  { maxZoom: 19, attribution: "Imagery &copy; Esri" });
blank.addTo(map);

const comm = L.geoJSON(D.comm, {
  style: f => ({ color: css("--comm"), weight: 0.7, fillColor: css("--host"), fillOpacity: f.properties.host ? 0.9 : 0 }),
  onEachFeature: (f, l) => l.bindTooltip(f.properties.SQNAME + (f.properties.host ? `（医院 ${f.properties.host}）` : ""),
    { sticky: true })
}).addTo(map);
const dist = L.geoJSON(D.dist, {
  style: () => ({ color: css("--dist"), weight: 1.6, fill: false }), interactive: false
}).addTo(map);
D.dlab.forEach(d => L.tooltip({ permanent: true, direction: "center", className: "dlbl" })
  .setLatLng([d.lat, d.lon]).setContent(d.name).addTo(map));
L.control.layers({ "社区地图": blank, "街道地图": osm, "卫星影像": img }, { "社区边界": comm, "区界": dist },
  { position: "topright" }).addTo(map);

L.geoJSON(D.old, { style: () => ({ color: css("--old"), weight: 1.5, fill: false, dashArray: "4 3" }),
  onEachFeature: (f, l) => l.bindTooltip(f.properties.fac_name, { sticky: true }) }).addTo(map);

const fmt = v => Number(v).toLocaleString("zh-CN");
const layers = {};
L.geoJSON(D.new, {
  style: () => ({ color: css("--aoi"), weight: 2.5, fillColor: css("--aoi"), fillOpacity: 0.28 }),
  onEachFeature: (f, l) => {
    const p = f.properties;
    l.bindPopup(`<b>${p.no}. ${p.fac_name}</b><br>AOI：${p.aoi_name}（${p.aoi_src}，${fmt(p.area_m2)} m²）`
      + `<br>所在社区：${p.sqname}<br>2018 地址：${p.addr2018}<br>现状：${p.now}<br>核对：${p.verdict}`);
    l.bindTooltip(String(p.no), { permanent: true, direction: "center", className: "lbl" });
    layers[p.no] = l;
  }
}).addTo(map);
D.pts.forEach(p => L.circleMarker([p.lat, p.lon], { radius: 6, color: "#fff", weight: 2,
  fillColor: css("--pt"), fillOpacity: 1 }).bindTooltip("供给表点位（高德坐标）").addTo(map));

const list = document.getElementById("list");
D.new.features.forEach(f => {
  const p = f.properties, c = document.createElement("div");
  c.className = "card";
  c.innerHTML = `<b><span class="num">${p.no}</span>${p.fac_name}</b>`
    + `<div class="kv">所在社区 <em>${p.sqname}</em> · 2018 门诊 <em>${fmt(p.outpat)}</em> 人次</div>`
    + `<div class="kv">2018 地址 <em>${p.addr2018}</em></div>`
    + `<div class="kv">AOI <em>${p.aoi_name}</em>（${p.aoi_src}）</div>`
    + `<div class="kv">现状 <em>${p.now}</em></div>`
    + `<div class="kv">核对 <em>${p.verdict}</em></div>`;
  c.onclick = () => { document.querySelectorAll(".card").forEach(x => x.classList.remove("on")); c.classList.add("on");
    map.fitBounds(layers[p.no].getBounds(), { maxZoom: 16, padding: [60, 60] }); layers[p.no].openPopup(); };
  list.appendChild(c);
});
const h = location.hash.slice(1);
if (h.includes("img")) { map.removeLayer(blank); img.addTo(map); }
const n = parseInt(h, 10);
if (layers[n]) list.children[n - 1].click();
</script>
</body>
</html>
"""
out = OUT / "map_5_hospitals.html"
out.write_text(html.replace("__DATA__", json.dumps(data, ensure_ascii=False)), encoding="utf-8")
print(out, round(out.stat().st_size / 1e6, 2), "MB")

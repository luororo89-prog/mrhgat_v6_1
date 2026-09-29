"""Read-only inventory of candidate MR-HGAT inputs; never changes source data."""
from pathlib import Path
import json
import pandas as pd
import geopandas as gpd

BASE = Path(r"D:\多功能网络构建")
ROOT = Path(__file__).resolve().parents[2]
SOURCES = [
    r"shp\深圳社区objectid.shp",
    r"等时圈\等时圈wgs84拆分\等时圈wgs84拆分\foot等时圈.shp",
    r"等时圈\等时圈wgs84拆分\Shenzhen_人口重心\深圳社区_联通人口重心result.shp",
    r"2018年11月非通勤出行\网格_人口_社区\网格_人口_社区\深圳各社区人口.shp",
    r"2018年11月非通勤出行\网格_人口_社区\网格_人口_社区\深圳250m网格_人口_所在社区.shp",
    r"poi\poi\medical\医院_床位.shp",
    r"poi\poi\medical\社康带权重.shp",
    r"poi\poi\medical\医疗设施set\TH.shp",
    r"poi\poi\medical\医疗设施set\SH.shp",
    r"poi\poi\medical\GRID\就医网格.shp",
    r"poi\poi\medical\GRID\居住网格.shp",
    r"26上实验\api\res_route_od_aligned_all.xlsx",
    r"26上实验\api\od.xlsx",
    r"服务能力分级\医院数据\医院名单_分级0721.xlsx",
    r"poi\poi\medical\社区服务中心服务能力.xlsx",
    r"服务能力分级\五次嵌入\result\merged_all_data.csv",
    r"服务能力分级\五次嵌入\result\medical_flow_classification.csv",
    r"服务能力分级\五次嵌入\result\medical_flow.csv",
    r"代码\出行轨迹\result\medical_flow_result1224.csv",
    r"代码\出行轨迹\result\medical_flow_result12241.csv",
    r"代码\出行轨迹\result\medical_flow_split_result.csv",
    r"26上实验\gat\data\feature_matrix_ini.csv",
    r"poi\poi\medical\四次统计\社区医疗数据_多表合并汇总_改-99.csv",
    r"2018年11月非通勤出行\网格人口.csv",
]


def summarize(df):
    out = {"rows": len(df), "columns": list(df.columns),
           "nulls": {str(c): int(df[c].isna().sum()) for c in df.columns}}
    if isinstance(df, gpd.GeoDataFrame):
        out["crs"] = str(df.crs)
        out["geometry_types"] = df.geom_type.value_counts().to_dict()
    out["capacity_and_tier_values"] = {}
    for c in df.columns:
        if any(k in str(c).lower() for k in ("bed", "床位", "等级", "tier", "type", "weight", "权重")):
            out["capacity_and_tier_values"][str(c)] = {
                str(k): int(v) for k, v in df[c].value_counts(dropna=False).head(15).items()}
    return out


def main():
    entries = []
    for rel in SOURCES:
        path = BASE / rel
        entry = {"path": str(path), "exists": path.exists()}
        try:
            if path.suffix == ".shp":
                entry.update(summarize(gpd.read_file(path)))
            elif path.suffix == ".csv":
                for encoding in ("utf-8-sig", "gb18030"):
                    try:
                        df = pd.read_csv(path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                entry.update(summarize(df))
                entry["encoding"] = encoding
            else:
                book = pd.ExcelFile(path)
                entry["sheets"] = {name: summarize(pd.read_excel(book, sheet_name=name))
                                   for name in book.sheet_names}
        except Exception as exc:
            entry["inspection_error"] = str(exc)
        entries.append(entry)
        print(json.dumps(entry, ensure_ascii=False, default=str), flush=True)
    output = ROOT / "reports" / "A_source_inventory.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(entries, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

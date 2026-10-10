# 标准化资源与直接供需比资料说明

本资料包集中保存标准化资源与直接供需比方法的计算代码、六类输入数据、两种人口口径的供给结果及配套空间图件。归档共复制八十五个文件，逐文件 SHA256 校验与来源文件一致。

主要查看联通人口结果。七普扩展人口结果作为另一种人口口径单独保留。社区、设施和网格输出按同一次计算的记录对应。

## 一 文件夹内容

| 位置 | 内容 | 文件数 |
| --- | --- | --- |
| 代码 | 九项指标计算、空间分布绘图、高值社区标注和绘图辅助模块 | 4 |
| 输入数据 | 社区边界、人口重心、人口网格、医院、社康和道路及配套文件 | 30 |
| 输出结果 | 工作簿、指标表、空间数据、设施与网格统计、计算记录和图件 | 51 |
| 文件清单.json | 来源路径、归档路径、文件大小及内容校验值 | 1 |

代码和输入数据在各自目录内保留项目中的相对目录结构及文件名。输出结果保留 `supply_indicators_2018_osm_190101` 目录及其子目录。

## 二 主要结果入口

| 文件 | 用途 |
| --- | --- |
| [联通人口九项指标工作簿](<D:/多功能网络构建/标准化资源与直接供需比/输出结果/supply_indicators_2018_osm_190101/联通人口_九项医疗供给指标_2018.xlsx>) | 查看九项供给指标 |
| [联通人口社区结果](<D:/多功能网络构建/标准化资源与直接供需比/输出结果/supply_indicators_2018_osm_190101/community_supply_indicators_2018_mobile_users.csv>) | 666 个行政社区的完整指标及辅助字段 |
| [联通人口空间结果](<D:/多功能网络构建/标准化资源与直接供需比/输出结果/supply_indicators_2018_osm_190101/community_supply_indicators_2018_mobile_users.gpkg>) | 在 GIS 中查看社区指标 |
| [联通人口计算记录](<D:/多功能网络构建/标准化资源与直接供需比/输出结果/supply_indicators_2018_osm_190101/calculation_audit_2018_mobile_users.json>) | 查看输入、参数、设施筛选和计算定义 |
| [设施统计](<D:/多功能网络构建/标准化资源与直接供需比/输出结果/supply_indicators_2018_osm_190101/facility_supply_audit_2018_mobile_users.csv>) | 查看设施资源、中位数、资源得分和竞争人口 |
| [网格统计](<D:/多功能网络构建/标准化资源与直接供需比/输出结果/supply_indicators_2018_osm_190101/grid_supply_diagnostics_2018_mobile_users.csv>) | 查看居住网格人口、可达性和时间档设施数量 |
| [设施归属表](<D:/多功能网络构建/标准化资源与直接供需比/输出结果/supply_indicators_2018_osm_190101/facility_community_15min.csv>) | 查看设施与社区的十五分钟范围对应关系 |
| [联通人口空间分布图](<D:/多功能网络构建/标准化资源与直接供需比/输出结果/supply_indicators_2018_osm_190101/spatial_visualization_mobile_users/supply_spatial_distribution.png>) | 查看九项指标空间分布 |
| [联通人口高值标注图](<D:/多功能网络构建/标准化资源与直接供需比/输出结果/supply_indicators_2018_osm_190101/spatial_visualization_mobile_users/supply_spatial_distribution_highlights.png>) | 查看高值社区位置 |

七普扩展人口结果使用无 `_mobile_users` 后缀的社区、设施、网格和计算记录文件。其空间图件位于 `spatial_visualization` 子目录。图件保留已有 PNG、PDF、SVG 和 TIFF 格式。

## 三 输入数据

| 输入 | 文件名 | 使用内容 |
| --- | --- | --- |
| 行政社区边界 | 深圳社区objectid.shp | 行政社区评价单元和输出空间边界 |
| 社区人口重心 | 深圳社区_联通人口重心result.shp | 十五分钟设施集合的社区起点 |
| 二百五十米人口网格 | 深圳250m网格_人口_所在社区.shp | 联通人和七普加字段及网格所属社区 |
| 医院点位 | hospital_beds.shp | HOSP_GRD 医院等级及 FINAL_BED 床位数等字段 |
| 社康点位 | 2018年深圳CHC.csv | 点位及服务能力评分 |
| 道路网络 | shenzhen_osm_roads_190101.gpkg | 二〇一九年一月一日 OpenStreetMap 快照的深圳道路 |
| 道路元数据 | shenzhen_osm_roads_190101.json | 道路来源、提取设定和通行参数 |

四组 Shapefile 均保留已有 SHP、SHX、DBF、PRJ、CPG 及配套索引文件。读取时使用对应 SHP 文件，并保持同组文件在同一目录。

输入数据对应的保存结果包含 666 个社区、25,496 个居住网格、912 家 CHC、17 家 SH 和23家 TH。医院来源文件包含139条记录，指标计算按所定义的医院等级和类型筛选其中的40家。

## 四 计算方法

### RA 资源可用性

各层级分别使用资源正值中位数作为参照。每家设施的得分为零点五加零点五乘以该设施资源除以同层级中位数，社区 RA 为所属设施得分之和。

医院资源使用床位数，社康使用服务能力评分。保存结果中的 CHC、SH 和 TH 中位数分别为 1.345、400 和 810。

### SA 空间可达性

每家设施的床位数或社康服务能力评分除以自身服务范围内人口总量，得到设施供需比。社区将其设施集合内的同层级供需比直接相加。每万人指标在该结果基础上乘以一万。

SA 的分子使用设施资源原值。服务范围内人口按人数计入分母。道路时间用于确定范围，范围内的供需比计算和社区汇总采用直接计入方式。

### TC 出行便利性

统计居民在零至五、五至十、十至十五分钟时间档内能够到达的设施数量，按社区人口加权，再分层进行 TOPSIS。三个档次的权重分别为 0.5、0.3 和 0.2。

### 九项核心字段

| 维度 | CHC | SH | TH |
| --- | --- | --- | --- |
| RA | RA_CHC | RA_SH | RA_TH |
| 每万人 SA | SA_CHC_per_10000 | SA_SH_per_10000 | SA_TH_per_10000 |
| TC | TC_CHC | TC_SH | TC_TH |

RA 为相对资源得分，医院 SA 保留每万人床位供给机会的单位，社康 SA 保留每万人服务能力评分的单位，TC 为零至一的相对贴近度。三个层级分别解释。

## 五 代码入口及运行口径

核心指标代码为 [calculate_supply_indicators_2018.py](<D:/多功能网络构建/标准化资源与直接供需比/代码/26上实验/medical_identification_validation/src/calculate_supply_indicators_2018.py>)。主函数分别处理联通人口和七普扩展人口。

空间分布绘图代码为 [visualize_supply_spatial_distribution.py](<D:/多功能网络构建/标准化资源与直接供需比/代码/26上实验/medical_identification_validation/src/visualize_supply_spatial_distribution.py>)，高值标注代码为 [highlight_supply_high_value_communities.py](<D:/多功能网络构建/标准化资源与直接供需比/代码/26上实验/medical_identification_validation/src/highlight_supply_high_value_communities.py>)。绘图辅助模块 `audit_panel_alignment.py` 一并归档。

计算使用 Python 及 GeoPandas、NumPy、pandas、Shapely 和 SciPy。绘图另使用 Matplotlib。

代码保留源文件内容，包括原有输入路径、输出路径和通行方式配置。直接运行时，读写位置由代码中的路径配置决定。本资料包的来源与归档位置对应关系保存在 [文件清单.json](<D:/多功能网络构建/标准化资源与直接供需比/文件清单.json>)。

### 保存结果的参数

| 参数 | CHC | SH | TH |
| --- | --- | --- | --- |
| 设施归属范围 | 十五分钟步行 | 十五分钟步行 | 十五分钟步行 |
| 竞争人口范围 | 二十五分钟骑行 | 四十分钟驾车 | 七十分钟驾车 |

上述参数来自配套 `calculation_audit_2018_mobile_users.json` 和 `calculation_audit_2018.json`，用于解释归档的保存结果。

### 归档代码的配置

| 参数 | CHC | SH | TH |
| --- | --- | --- | --- |
| 设施归属范围 | 十五分钟步行 | 十五分钟骑行 | 十五分钟驾车 |
| 竞争人口范围 | 二十五分钟步行 | 四十分钟骑行 | 七十分钟驾车 |

归档代码与保存结果的通行方式配置不同。此处分别列明两种口径，代码与结果各自保留原有内容。现有保存结果应按其配套计算记录解释。

# mech_no_pair 聚合向量供给变化还原

`reconstruct_mech_no_pair_supply_v13.py` 读取目的地预测实验保存的 `mech_no_pair/seed*/arrays.npz`。其中 `h` 是本地基线、补偿和竞争三个通道拼接后的聚合向量。脚本以该向量拟合九维供给解码器，并在同一解码器下将补偿和竞争通道置零。供给变化定义为

```text
D(h_base, h_comp, h_compete) - D(h_base, 0, 0)
```

线性解码器下，补偿和竞争两部分严格相加。截距和本地基线项在差分中抵消。该量是模型表征干预下的供给变化，不是独立观测的互动后供给。

运行示例

```bash
python 5lineb/v13_geoflow/reconstruct_mech_no_pair_supply_v13.py \
  --mechanism-results /path/to/outpatient_d0.25_od_mechanism_v13 \
  --baseline-table /path/to/node_results.csv \
  --folds /path/to/node_spatial_folds.csv \
  --ruler /path/to/ruler.json \
  --output /path/to/local_supply_change
```

四面板图使用固定 OD 矩阵、有向功能社区成员表和社区 GeoJSON。

```bash
PYTHONPATH=/path/to/alignment/helper \
python 5lineb/v13_geoflow/plot_mech_no_pair_supply_change_abcd_v13.py \
  --reconstruction /path/to/local_supply_change \
  --fixed-split /path/to/outpatient_d0.25_od_mechanism_v13/fixed_split.npz \
  --cluster-membership /path/to/cluster_membership.csv \
  --geojson /path/to/supply_od_clusters.geojson \
  --seed 2 \
  --output /path/to/fig_mech_no_pair_supply_change_seed2
```

数值结果与图文件应保存在仓库外，不提交模型权重、聚合向量、表格或图片。

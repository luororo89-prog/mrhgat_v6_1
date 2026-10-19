# mech_no_pair 聚合向量相对供给偏移还原

`reconstruct_mech_no_pair_supply_v13.py` 读取目的地预测实验保存的 `mech_no_pair/seed*/arrays.npz`。其中 `h` 是基线、补偿和竞争三个通道拼接后的聚合向量。脚本不再把外部通道置零，因为零值几乎不在外部聚合的经验分布中，会产生系统性外推偏移。

默认方法复用既有空间折。对于每个留出社区，只在其他空间折的同一基线配置类型中寻找 `h_base` 最相似的 20 个社区。`h_base` 距离在训练折标准化后使用欧氏距离计算；这 20 个邻居的补偿与竞争聚合向量逐维中位数构成典型外部聚合。另一项 RidgeCV 将完整聚合向量解码到九项固定标尺供给指标。相对偏移定义为

```text
D^(-fold)(h_base, h_ext) - D^(-fold)(h_base, median_20NN^(-fold)[h_ext | h_base, T])
```

两项模型都不使用被评价空间折的节点。线性解码器下，补偿和竞争残差贡献严格相加，截距和基线通道在差分中抵消。

这一简单参照把同类型、且本地供给表示相近的社区普遍具有的外部聚合吸收到典型项中。它不再单独拟合医院服务范围，也不显式使用三甲、二级或 CHC 目的地指标。常见的医院跨社区流动可以通过同类近邻的共同模式进入参照，但当前聚合 OD 不能识别患者实际访问的机构层级，所以剩余偏移不能直接等同于不合理社康流动，也不是独立观测的绝对供给变化。

供给解码器仍为线性 RidgeCV，以保留补偿与竞争通道严格可加的分解。脚本还保留 `--reference-method extra_trees` 作为敏感性分析；不指定时默认使用 `type_knn_median`。

默认方法在每个 `seed*_decoder.npz` 中保存每个社区的 20 个邻居索引与标准化距离，`node_supply_change.csv` 同时保存平均和最大邻居距离。Extra Trees 敏感性分析才会额外输出 `configuration_service_range_reference.csv`。

方法沿用空间交叉拟合和条件参照的思路（Chernozhukov et al., 2018, https://doi.org/10.1111/ectj.12097；Apley and Zhu, 2020, https://doi.org/10.1111/rssb.12377；Aas et al., 2021, https://doi.org/10.1016/j.artint.2021.103502）。默认实现使用同类型局部邻域的逐维中位数，不把它表述为参数化条件均值；Extra Trees 条件均值仅作为敏感性分析。

运行示例

```bash
python 5lineb/v13_geoflow/reconstruct_mech_no_pair_supply_v13.py \
  --mechanism-results /path/to/outpatient_d0.25_od_mechanism_v13 \
  --baseline-table /path/to/node_results.csv \
  --folds /path/to/node_spatial_folds.csv \
  --ruler /path/to/ruler.json \
  --output /path/to/local_relative_supply_shift
```

四面板图使用固定 OD 矩阵、有向功能社区成员表和社区 GeoJSON。

```bash
PYTHONPATH=/path/to/alignment/helper \
python 5lineb/v13_geoflow/plot_mech_no_pair_supply_change_abcd_v13.py \
  --reconstruction /path/to/local_relative_supply_shift \
  --fixed-split /path/to/outpatient_d0.25_od_mechanism_v13/fixed_split.npz \
  --cluster-membership /path/to/cluster_membership.csv \
  --geojson /path/to/supply_od_clusters.geojson \
  --seed 2 \
  --output /path/to/fig_mech_no_pair_relative_supply_shift_seed2
```

数值结果与图文件应保存在仓库外，不提交模型权重、聚合向量、表格或图片。

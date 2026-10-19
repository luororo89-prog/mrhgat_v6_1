# mech_no_pair 聚合向量相对供给偏移还原

## 论文方法表述

我们在每个空间留出折内，利用其余社区的九项基线供给指标拟合 RidgeCV，以估计相同本地供给条件下的典型外部聚合，并将实际聚合向量与保留相同 `h_base` 的典型参照向量经同一交叉拟合解码器所得结果之差定义为相对供给表征偏移。这样处理旨在将医院较宽服务范围内合理且具有规律性的跨社区流动吸收到参照项中，而把原则上应就近提供服务的社康跨社区流动所形成的剩余偏移视为潜在的不合理流动信号；但由于聚合 OD 不能识别实际就诊机构层级，该信号不能被解释为社康不合理流动的直接证据。

`reconstruct_mech_no_pair_supply_v13.py` 读取目的地预测实验保存的 `mech_no_pair/seed*/arrays.npz`。其中 `h` 是基线、补偿和竞争三个通道拼接后的聚合向量。脚本不再把外部通道置零，因为零值几乎不在外部聚合的经验分布中，会产生系统性外推偏移。

脚本复用既有空间折。在每个训练折内，一项 RidgeCV 从九项基线供给指标预测补偿与竞争通道，得到留出折的条件典型外部聚合；另一项 RidgeCV 将完整聚合向量解码到九项固定标尺供给指标。相对偏移定义为

```text
D^(-fold)(h_base, h_ext) - D^(-fold)(h_base, E^(-fold)[h_ext | X])
```

两项模型都不使用被评价空间折的节点。线性解码器下，补偿和竞争残差贡献严格相加，截距和基线通道在差分中抵消。

这一参照把同等基线供给条件下反复出现的常规跨社区联系吸收到典型项中，因此可以近似容纳医院较宽服务范围内的合理流动。社康原则上以邻近基层服务为主，偏离典型项的非本地联系可作为进一步检验社康流动错配的候选信号。但是当前聚合 OD 不能识别患者实际访问的机构层级，所以相对偏移不能直接等同于不合理社康流动，也不是独立观测的绝对供给变化。

方法依据包括条件均值残差化与交叉拟合（Robinson, 1988, https://doi.org/10.2307/1912705；Chernozhukov et al., 2018, https://doi.org/10.1111/ectj.12097），以及相关特征下避免分布外替换的条件参照（Apley and Zhu, 2020, https://doi.org/10.1111/rssb.12377；Aas et al., 2021, https://doi.org/10.1016/j.artint.2021.103502）。

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

# MR-HGAT 机制识别稳定性实验

以执行方案 v6.1 为主，论文方法和作者代码用于解决通道学习不稳定问题。

## 当前口径

使用原九项设施评价、聚合人流及原拓扑。通道学习与机制判定分开，供给差和低供给来源流入等判断指标仅用于训练后的检验。模型代码中的通道标识不表示已经确认相应机制。

训练时两个潜在通道保持匿名。当前补丁保留原通道公式，对比原损失、表示去相关正则，以及联合因素图判别辅助项，并落实空间五折选择训练轮数。辅助判别器学习匿名因素编号。机制命名由训练后的补偿和竞争检验确定。两个检验均通过或证据不足的通道保留待判定。

跨运行匹配只用于稳定性诊断。模拟真值匹配只用于评价分解能否恢复，不能用于真实数据的机制命名。原通道编号评分不再等同于事后机制判定结果。逐通道方向筛查保存在 `anonymous_hypothesis_screen.csv`，其中未完成零模型检验的通道均不命名。

## 文件位置

| 内容 | 相对位置 |
|---|---|
| 模型及实验代码 | `src/D_mrhgat` |
| 本次实验结果 | 原 `results/runs/20260928_md_stability` 已删除——用了向 vmap 批量训练，被发现与逐个训练结果在真实规模下分歧巨大（见 `reports/D7_batching_divergence.md`），需用去掉批量训练后的代码重新跑 |
| 方法依据与论文代码链接 | `reports/D6_literature_method_basis.md` |
| 六篇论文提取文本 | `reports/method_sources` |
| 作者代码版本和文件校验记录 | `reports/method_sources/official_code/source_manifest.json` |
| 文件复制校验记录 | `reports/location_manifest.json` |
| 稳定性补丁验证 | `tests/test_stability.py` |
| 事后机制判定实现 | `src/D_mrhgat/mechanism_judgment.py` |
| 多起点独立验证 | `src/D_mrhgat/run_restart_validation.py` |

`development_cv.json` 保存逐折轮数和留出误差，`selection.json` 保存配置选择。各次拟合的模型参数、后验、训练记录和评分保存在对应结果子目录。以实际生成的文件为准。

## 复现入口

当前使用已有Python运行环境

`D:\多功能网络构建\26上实验\mrhgat_v6_1\.venv\Scripts\python.exe`

在本目录下运行

```powershell
& 'D:\多功能网络构建\26上实验\mrhgat_v6_1\.venv\Scripts\python.exe' .\src\D_mrhgat\run_stability.py --stage development --directory .\results\runs\20260928_md_stability
```

此前执行过五折验证、55次独立复核拟合和25次多起点补充拟合，但该批结果用的是 vmap 批量训练，已发现与逐个训练在真实规模下分歧巨大（结果不可信），已连同结果目录一并删除，详见 `reports/D7_batching_divergence.md`。`src/D_mrhgat/stability.py`、`batched_train.py` 已改为逐个训练每个种子（可选多进程并行，不再向量化打包），需用现在的代码重新跑一遍才能得到有效结果。机制名称仍保留待判定，不构成真实补偿或竞争的因果证据。

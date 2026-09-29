# D7 批量(vmap)训练与逐个训练的分歧核查

## 结论

`stability.py`(`fit_group`/`fit_baselines`)与 `batched_train.py`(`fit_batch`/`baseline_batch`)此前用 `torch.func.vmap` + `stack_module_state` 把同一组的多个种子打包成一次向量化计算来训练,以加快170次拟合的速度。核查发现,这种批量计算在真实训练轮数下,跟"老老实实逐个训练"并不等价,已将其移除,改为逐个训练(可选多进程并行,但不再向量化打包)。

## 核查过程

1. 在小规模合成数据、4个epoch时,批量训练与逐个训练的最大绝对偏差约在10⁻⁸量级,在现有测试容差(`rtol=2e-4, atol=2e-6`)内。
2. 拉长到接近真实使用的训练轮数(60节点合成数据,60~300轮)后,批量与逐个训练的最终结果最大相对偏差可达70%~85%,且改动前后的模型架构均出现类似量级的偏差,证明这与具体架构无关。
3. 双精度(float64)重跑同一流程,偏差立刻降到机器精度量级(~10⁻¹⁶),确认根因是float32浮点舍入误差经过数百步Adam训练被链式放大,不是批量实现里的逻辑bug(没有跨种子串数据)。
4. 用真实666节点图数据、`variant='none'`、139轮(`selection.json` 里 `audit_fixed_epochs` 对应的轮数)重复验证:两个随机种子上,批量与逐个训练之间,目的地预测(argmax)有9%~54%的社区不一致,补偿/竞争通道归属有4%~26%的社区不一致(不同代码版本下具体数字有波动,但量级持续偏大)。
5. 排查六篇方法依据论文的官方代码(`reports/method_sources/official_code/` 及 `code/`),确认没有一篇使用类似的"多模型向量化打包训练"技术;这是本项目自行为提速添加的工程手段,没有外部先例可以佐证其可靠性。

## 决定

移除 `fit_group`/`fit_baselines`/`fit_batch`/`baseline_batch` 内部的 vmap 向量化批量训练,改为每个种子独立、依次训练(数学上与 `train.py` 里单独调用的训练循环完全一致)。新增 `n_workers` 参数,`n_workers>1` 时改用独立的操作系统进程并行跑多个种子(每个进程完全独立,不共享计算图),用于弥补速度损失,而不是重新引入跨种子的向量化打包。已验证 `n_workers=2` 与 `n_workers=1`(纯顺序)结果逐位一致。

## 影响范围

- `src/D_mrhgat/batched_train.py`:重写 `fit_batch`、`baseline_batch`。
- `src/D_mrhgat/stability.py`:重写 `fit_group`、`fit_baselines`。
- 调用方(`run_stability.py`、`run_restart_validation.py`、`run_null_validation.py`)未改动:函数签名与返回值结构保持一致。
- `tests/test_stability.py`、`tests/test_nulls.py`:新增针对 `fit_group`/`fit_baselines`/`fit_batch` 的回归测试(此前这两个函数没有专门的单元测试覆盖),并验证并行与顺序结果一致。

## 尚未处理的部分

本次只解决"批量训练是否忠实于逐个训练"这个实现层面的问题。`results/runs/20260928_md_stability` 下已有的170次拟合结果,是用旧的批量实现跑出来的,尚未用新的逐个训练重新验证——这些结果里"跨种子不一致"的具体数值,可能部分来自已被移除的批量实现偏差,需要重新跑一遍才能确认现在的数字。

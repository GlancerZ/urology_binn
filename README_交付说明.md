# 离散时间 MAPS Teacher–Student 建模代码交付包

本代码包用于代码审阅与在获授权的数据环境中复现。当前冻结方案为 6 层 Reactome-informed BINN、10 个一年期离散生存区间、Teacher-MAPS 与配对蒸馏 Student-MAPS（训练 seed 7）。

## 包含内容

- `2_internal_python_package/`：内部 Python 包源码、训练/预测/评价命令、绘图脚本与单元测试。
- `3_frozen_configs/`：冻结的 TOML 配置与模型登记。
- `BINN.main/binn/model/`：当前建模所依赖的 BINN/通路网络实现。
- `BINN.main/binn/data/downloads/`：建模所用 Reactome 映射表与通路关系表。
- `feature_lists/`：NMR 与蛋白特征名称清单。

## 不包含内容

不包含患者级原始数据、冻结分割表（含受试者 ID）、训练权重 `.pt`、逐人预测、实验结果与结果报告、模型选择记录、结果图及 Python 虚拟环境。仅凭本代码包**不能**复现论文数值；需在合规环境中接入原始队列、分割表和权重，或使用原始数据重新训练。冻结配置中的数据与权重路径均为相对路径，以配置文件所在目录 `3_frozen_configs/` 为基准解析。

## 核心代码位置

- 网络与10个时间区间输出：`2_internal_python_package/src/maps_discrete/modeling.py`
- 时间分箱及删失掩码：`2_internal_python_package/src/maps_discrete/endpoint.py`
- 教师/学生训练与蒸馏：`2_internal_python_package/src/maps_discrete/training.py`
- 数据读入、Reactome映射与预处理：`2_internal_python_package/src/maps_discrete/data.py`、`preprocessing.py`
- 预测与累计10年风险：`2_internal_python_package/src/maps_discrete/prediction.py`
- 完整单 seed 流程和多 seed 运行：`2_internal_python_package/src/maps_discrete/pipeline.py`、`batch_runner.py`
- Wales 外部验证：`2_internal_python_package/src/maps_discrete/wales_external_validation.py`

## 使用提示

配置已指向本目录 `raw_inputs/` 下的数据（相对路径）；seed 7 权重如需使用，放在本目录的 `5_teacher_student_multiseed/seed_007/` 下。在 Python 3.11+ 环境中安装 `2_internal_python_package`（并安装 `networkx`、`scipy`；传统 ML 还需 `xgboost`、`lightgbm`；实际验证过的版本见 `2_internal_python_package/requirements_tested.txt`），然后运行：

```powershell
maps-research audit --config <analysis_config.toml>
maps-research run-maps --config <analysis_config.toml> --seed 7 --output <output_dir>
```

该项目是内部研究框架，不是脱离数据即可运行的公开成品库。Wales 验证的 Teacher 和 Student 实际均使用 seed 7 权重；旧版文字中“Student seed 98/87”是标签错误，不是实际加载的权重。

## 2026-09 新增：改进方案与评估脚本

在冻结 seed 7 方案之外，新增以下改进方案及其评估脚本（冻结配置与冻结模型行为不变）。评估结果见内部报告 `16_optimization_report_20260926.md`，不随公开仓库发布。

- **改进版 Teacher**：BINN 结构不变，训练目标改为 10 年内是否发病，加两阶段稀疏直连路径，5 个训练种子取平均。只输出 10 年风险，逐年风险仍用冻结 Teacher。
- **改进版 Student**（NMR + PRS）：BINN，10 年目标，不蒸馏，5 个训练种子取平均。

复现脚本（在 `2_internal_python_package/scripts/` 下运行，结果写入本目录的 `14_repeated_split_validation/`、`15_external_validation_wales/`）：

1. `repeated_split_validation.py`：冻结 Teacher 在 20 次重复划分上的评估，以及 KLK3 / KLK3+PRS 参照模型；
2. `repeated_split_ml_baselines.py`、`repeated_split_xgboost_tuned.py`、`repeated_split_deep_baselines.py`：传统 ML、调参 XGBoost、通用深度学习基线；
3. 先运行 `repeated_split_binn_improvements.py --variants binary_direct_l1_1e-2` 生成第一阶段特征选择表 `14_repeated_split_validation/binn_improvements/binary_direct_l1_1e-2_seed7_by_split.csv`，再运行 `--variants binary_relaxed --training-seeds 7 1 2 3 4` 得到改进版 Teacher（`repeated_split_students.py` 也读取这张表）；
4. `repeated_split_students.py` 与 `summarize_students.py`：Student 各变体与比较；
5. `external_validation_wales.py`（`--part teacher` / `--part student`）：Wales 外部验证。

包内新增选项均默认关闭：BINN `residual_width`（默认 1）、训练设置 `decoupled_weight_decay`（默认 False）。`tests/` 中除需要分割表数据的一项外，其余测试可在无数据环境下运行。

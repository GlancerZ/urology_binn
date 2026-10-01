# urology_binn

离散时间 MAPS Teacher–Student 风险预测模型代码。编码器是基于 Reactome 通路的 BINN；Teacher 输入 NMR 代谢组 + Olink 蛋白组 + PRS，Student 输入 NMR + PRS。

## 目录

- `2_internal_python_package/`：Python 包 `maps_discrete`（数据读入、模型、训练与蒸馏、评价）、实验脚本 `scripts/`、单元测试 `tests/`
- `3_frozen_configs/`：冻结配置（数据与权重路径均为相对路径）和模型登记
- `BINN.main/`：BINN 通路网络实现和 Reactome 映射表
- `feature_lists/`：NMR 与蛋白特征名称清单
- `README_交付说明.md`：详细说明与复现步骤

## 数据与结果

仓库只包含代码和使用说明，不含任何受试者级数据（原始矩阵、分割表、逐人预测）、模型权重和实验结果。在获授权的环境中把 `raw_inputs/` 放到仓库根目录即可运行。

## 安装与测试

```bash
cd 2_internal_python_package
pip install -r requirements_tested.txt
pip install -e .
python -m unittest discover -s tests
```

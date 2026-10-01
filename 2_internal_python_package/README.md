# maps-discrete-research

这是离散时间MAPS正式分析的内部研究包，不是面向公众发布的完整软件库。

版本：0.2.0（内部研究框架）

已实现：

- TOML冻结配置读取和阻断项检查；
- Python/NumPy/PyTorch统一seed设置；
- 固定split文件的唯一性、分组和SHA256校验；
- 365天/年的10区间离散时间结局；
- Simple dual-path Teacher、蒸馏Student和无蒸馏Student；
- train80拟合、validation10早停、test10评价的统一预处理与训练；
- NMR-only、PRS-only、NMR+PRS、NMR+Olink+PRS四个ML模态组；
- XGBoost、LightGBM、Random Forest、Logistic、Elastic Net、SVM-RBF；
- Harrell C-index、固定10年ROC-AUC、PR-AUC、连续NRI、IDI和DCA；
- 受试者层bootstrap接口；
- 100-seed运行计划、结果收集和探索性Top/Bottom seed选择保护；
- 每次运行的配置、split哈希、预测、训练历史、checkpoint和指标清单；
- test-based seed选择保护；
- 配置和工具函数单元测试。

## 使用的Python环境

Python 3.11+，实际验证过的依赖版本见 `requirements_tested.txt`：

```bash
pip install -r requirements_tested.txt
pip install -e .
python -m unittest discover -s tests   # 单元测试（test_framework 中的分割表检查需要 raw_inputs 数据）
```

## 常用命令

配置文件：`../3_frozen_configs/analysis_config.toml`（其中的数据与权重路径均为相对路径，以配置文件所在目录为基准）

```powershell
maps-research audit --config <config.toml>
maps-research plan-batch --config <config.toml> --maps-root <dir> --ml-root <dir> --output <jobs.csv>
maps-research run-maps --config <config.toml> --seed 1 --output <dir> --quick-smoke
maps-research run-maps --config <config.toml> --seed 1 --output <dir>
maps-research run-ml --config <config.toml> --seed 1 --output <dir>
```

`run-maps`正式运行和`run-ml`会检查`protocol.approved=true`；`--quick-smoke`只允许用于技术验证，结果不得进入正式分析。

## 混合治愈 Teacher（探索性分析，主分析未采用）

主分析继续使用离散时间比例风险 Teacher（`run-maps`），混合治愈模型仅作为探索性分析保留。

`mixture_cure.py` 在同一 Reactome BINN 编码器上接两个输出头：π = 10 年内发病概率，f(t | 发病) = 发病年份分布（`--timing-head ordinal` 为有序 logit，`softmax` 为自由 softmax）。训练使用考虑删失的混合似然：发病者 π·f(t)，第 c 年删失者 (1−π) + π·(1−F(c))；10 年内无删失时等价于全体的发病 BCE 加病例的年份交叉熵。

```powershell
maps-research run-mixture --config <config.toml> --seed 1 --output <dir>                  # 开发：train80 的 90% 训练、10% 早停、val10 评价
maps-research run-mixture --config <config.toml> --seed 1 --output <dir> --final-test     # 冻结后仅运行一次：train80 训练、val10 早停、test10 评价
python scripts/compare_mixture_cure_development.py --seeds 1 2 3 4 5                      # 比例风险 Teacher 与两种混合治愈 Teacher 的 val10 对比
```

开发模式不评价、也不保存 test10 的预测。

## 重复划分评估、改进方案与外部验证（2026-09）

`scripts/` 下的 `repeated_split_*.py`、`summarize_teacher_improvements.py`、`summarize_students.py`、`external_validation_wales.py` 用于 20 次重复随机划分评估、ML/深度学习基线、改进版 Teacher/Student（“10 年内是否发病”的二分类，不是离散时间模型）及 Wales 外部验证，运行顺序见上级目录 `README_交付说明.md`。

## 尚待后续数据到位后执行

- Base/Lifestyle/Clinical变量模型。

该项已预留在总分析目录，临床变量表尚未提供，因此不属于当前包的可运行数据模块。

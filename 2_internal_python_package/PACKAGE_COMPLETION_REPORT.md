# 内部Python包完成报告

版本：0.2.0  
完成日期：2026-09-10  
包名：`maps-discrete-research`

## 已完成模块

1. 固定配置、协议审批保护、seed统一设置和split哈希审计。
2. 一年365天、0-1至9-10年共10区间的离散时间结局和删失mask。
3. Simple dual-path离散时间Teacher、蒸馏Student、无蒸馏Student。
4. NMR半最小值填补、蛋白中位数填补、PRS/组学标准化；参数只由train80拟合。
5. 固定validation10早停和固定test10评价；training seed不会重新划分患者。
6. NMR-only、PRS-only、NMR+PRS、NMR+Olink+PRS四个机器学习模态组。
7. XGBoost、LightGBM、Random Forest、Logistic regression、Elastic Net和SVM-RBF。
8. Harrell C-index、10年ROC-AUC、PR-AUC、连续NRI、IDI、DCA和成对bootstrap 95%CI。
9. 100个training seed的200项运行计划、指标汇总和探索性Top/Bottom seed选择。
10. checkpoint、训练历史、患者级预测、DCA数据、通路节点数、路由表和run manifest自动保存。

## 验证结果

- 单元测试：9/9通过。
- GPU：NVIDIA RTX 5080，PyTorch CUDA可用。
- Teacher真实前向传播：3,016个输入；H1-H6节点数1327、1239、884、466、175、30；输出10个离散时间hazard logits。
- Student节点数：98、96、93、74、45、20。
- 完整GPU技术冒烟：Teacher、蒸馏Student、无蒸馏Student均完成2 epochs并生成checkpoint、预测和测试指标。
- ML技术冒烟：NMR-only Logistic regression完成训练、预测和统一评价。
- 成对bootstrap技术冒烟：C-index/AUC/PR-AUC差值、NRI和IDI均可从患者级预测文件计算。

技术冒烟结果仅验证流程，不进入正式性能报告。

## 关键位置

- 冻结配置：`3_frozen_configs/analysis_config.toml`
- 统计定义：`1_analysis_protocol/STATISTICAL_METHODS_LOCKED.md`
- 100-seed清单：`4_cohort_and_split_freeze/100_seed_job_plan.csv`
- GPU冒烟输出：`12_run_logs/quick_smoke_seed001`和`12_run_logs/quick_smoke_all_seed002`
- ML冒烟输出：`12_run_logs/quick_smoke_ml_seed001`

正式批量分析仍由配置中的`protocol.approved`开关保护；在研究流程最终批准前不会误启动正式100-seed任务。

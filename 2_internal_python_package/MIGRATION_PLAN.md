# 最终训练代码迁移完成记录

状态：核心迁移已完成（2026-09-10）。

## Phase 1：冻结公共接口（完成）

- `CohortBundle`：样本、结局、随访时间、固定split。
- `PreprocessingBundle`：仅由train80拟合的填补和标准化参数。
- `TeacherMAPS`、`StudentMAPS`：统一`fit/predict/save/load`接口。
- `PredictionFrame`：每名患者、模型、training seed的标准化预测长表。

## Phase 2：迁移训练逻辑（完成）

- 从既有离散时间脚本迁移模型类、离散时间targets、教师训练、学生训练和预测。
- 移除全局`SEED`、固定输出路径及`seed10`文件名。
- training seed通过函数参数传递，split seed只记录、不用于重分割。
- 根据审核结论统一365天和学生蒸馏对象。

## Phase 3：迁移普通ML（完成）

- 所有算法读取固定split manifest。
- 模态组合和超参数搜索空间写入冻结配置。
- validation完成调参与阈值选择；test预测单独导出。

## Phase 4：统一评价（核心统计完成）

- 读取标准PredictionFrame计算C-index、AUC、PR-AUC、NRI、IDI、DCA和bootstrap CI。
- 同一患者上进行配对模型比较。
- 生成5-seed ensemble和全seed稳定性分布。

## Phase 5：外部验证（待正式分析阶段）

- Wales只调用已冻结模型的`predict`。
- 校验特征、单位、编码、缺失处理和Reactome映射一致性。

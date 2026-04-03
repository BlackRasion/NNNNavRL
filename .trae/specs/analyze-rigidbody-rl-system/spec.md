# 简化刚体强化学习系统审计与优化 Spec

## Why
当前项目已完成从无人机到简化 Go2 刚体机器人的迁移，但训练链路、状态与奖励定义、动作控制安全层、可复现性与代码质量仍缺少系统性审计与统一规范。需要通过一次端到端分析与定向改造，提升可解释性、可复现性与训练稳定性。

## What Changes
- 建立 `isaac-training/training/scripts/` 全景模块图、调用拓扑图与训练时序图（PlantUML）。
- 产出状态空间、奖励函数、动作空间、速度控制相关配置项与默认值清单（Markdown 表格）。
- 审计并修正 `env.py` 状态拼接逻辑，补充维度一致性检查、NaN/Inf 防护与归一化策略说明。
- 审计并结构化 `env.py` 奖励分量，输出权重、冲突分析与梯度敏感性结果。
- 审计 `ppo.py -> go2_velocity_controller.py -> go2_robot.py` 动作链路。
- 对速度控制器进行测试与审查。
- 对 `env.py`、`go2_robot.py`、`go2_velocity_controller.py`、`ppo.py` 做代码质量优化（冗余注释、死代码、未使用导入、函数拆分、类型注解、格式化）。
- 输出优化 diff 与新增/删除行统计，以及关键错误与性能隐患清单。

## Impact
- Affected specs: 环境观测与奖励定义、PPO 训练流程、动作执行安全、工程可复现与可维护性
- Affected code:
  - `isaac-training/training/scripts/env.py`
  - `isaac-training/training/scripts/ppo.py`
  - `isaac-training/training/scripts/go2_robot.py`
  - `isaac-training/training/scripts/go2_velocity_controller.py`
  - `isaac-training/training/scripts/train.py`（如需补充复现性信息导出）
  - `isaac-training/training/scripts/eval.py`（如需补充一致性检查）
  - `isaac-training/training/scripts/` 下分析文档与测试相关文件

## ADDED Requirements
### Requirement: 系统级结构与流程可视化
系统 SHALL 提供训练工程的模块拓扑图与训练时序图，覆盖主入口到 PPO 更新闭环。

#### Scenario: 生成模块与时序图
- **WHEN** 执行系统分析任务
- **THEN** 输出目录结构、模块依赖拓扑图与 PlantUML 训练时序图
- **AND** 明确数据流阶段：仿真环境 → 状态提取 → 奖励计算 → 策略网络 → 速度指令 → 刚体执行 → 下一帧状态

### Requirement: 配置与可复现信息清单
系统 SHALL 汇总状态、奖励、动作与速度控制相关配置项默认值，并标注随机种子、超参数与缓存路径。

#### Scenario: 生成配置表
- **WHEN** 解析配置与代码默认参数
- **THEN** 产出 Markdown 表格，字段至少包含“配置项、默认值、来源文件、作用阶段、备注”

### Requirement: 状态空间一致性与鲁棒性审计
系统 SHALL 对状态拼接逻辑执行维度一致性检查，并防止 NaN/Inf 进入策略网络。

#### Scenario: 状态审计通过
- **WHEN** 采集并拼接观测状态
- **THEN** 状态维度与文档声明一致
- **AND** 对非法值进行拦截/替换/告警
- **AND** 输出状态变更日志（字段名｜旧维度｜新维度｜归一化方式｜备注）

### Requirement: 奖励函数可解释与可观测
系统 SHALL 列出全部奖励分量与权重，支持冲突分析、敏感性评估与实时可视化。

#### Scenario: 奖励可视化与调参
- **WHEN** 训练运行
- **THEN** 可实时查看分量奖励曲线与总奖励
- **AND** 提供权重调优脚本模板以支持快速实验

### Requirement: 动作链路安全约束
系统 SHALL 在策略动作到刚体执行链路中保证维度匹配、限幅生效。

#### Scenario: 随机动作输入安全执行
- **WHEN** 输入任意合法/边界动作
- **THEN** 速度命令保持在物理限制范围内
- **AND** 急停触发时输出被安全置零

### Requirement: 代码质量与测试保障
系统 SHALL 对关键脚本完成静态质量优化和单元测试覆盖，并输出可核验变更统计。

#### Scenario: 质量门禁通过
- **WHEN** 执行格式化、静态检查与单元测试
- **THEN** 无新增关键错误
- **AND** 提供 diff 与新增/删除行统计

## MODIFIED Requirements
### Requirement: 训练链路文档化要求
训练系统的文档要求从“可运行”升级为“可追踪、可解释、可复现”，必须同时包含结构图、时序图、配置表与风险清单。

## REMOVED Requirements
### Requirement: 仅依赖内联注释理解系统
**Reason**: 内联注释难以维护且无法覆盖跨模块流程，不能满足系统审计与复现实验要求。  
**Migration**: 使用结构化分析文档、表格和图示替代零散注释，注释仅保留关键设计动机（why）。

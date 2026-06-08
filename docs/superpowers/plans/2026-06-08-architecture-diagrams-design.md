# NPU灵衢KVCache架构图设计文档

## 文档信息
- 日期: 2026-06-08
- 关联立项报告: `docs/superpowers/plans/2026-06-08-npu-lingqu-kvcache-technical-proposal.md`

## 图表清单

### 图1: AS IS架构现状与痛点
- **文件**: `docs/superpowers/assets/asis-architecture.puml` → `asis-architecture-pain-points.svg`
- **内容**: Store模式4跳路径（红色）+ P2P模式卡侧RoCE（橙色）+ 缺失能力（深红）
- **要点**:
  - Store模式: NPU HBM → Host DRAM → RDMA → Host DRAM → NPU HBM，4跳15-25μs
  - P2P模式: NPU HBM → 卡侧RoCE → NPU HBM，1跳5-10μs，有灵衢优化空间
  - 缺失: GVA统一编址、灵衢HCCP、多级缓存、智能路径选择

### 图2: TO BE总体优化架构
- **文件**: `docs/superpowers/assets/tobe-architecture.puml` → `tobe-architecture-solution.svg`
- **内容**: 完整TO BE架构，包含所有优化点
- **层次**:
  - 应用层: vLLM / SGLang / 自研
  - API层: 标准API + GVA扩展 + 场景感知
  - Tiered-Cache数据平面: L0-L4五级
  - Dual-Path传输引擎: HCCP/HCOM/RDMA/TCP + 智能选择器
  - GVA统一编址层: 256TB全局地址
  - 硬件层: A2(HCCS+RDMA) / A3/A5(灵衢UB+RDMA)
- **痛点解决标注**:
  - Store 4跳 → L0↔L1 HCCP 2跳
  - 缓存2层 → L0-L4五级，命中率+20-30%
  - 静态路径 → 智能选择器+并发
  - 地址转换 → GVA零转换
  - P2P RoCE → 灵衢HCCP <1μs

### 图3: Tiered-Cache L0-L4数据流详图
- **文件**: `docs/superpowers/assets/tiered-cache-dataflow.puml` → `tiered-cache-dataflow.svg`
- **内容**: 五级缓存层次 + 两种场景数据流 + DataCopier迁移引擎
- **场景路径**:
  - 路径A (蓝色): 多轮对话 - Soft Pin → HCCP预取 → 降级
  - 路径B (绿色): Coding Agent - System Prompt Pin → 文件切换 → ContextGroup共享 → 持久化
  - DataCopier (橙色): 热度评分 → 层级决策 → 异步迁移

## 设计决策
- 采用方案B: 3张图覆盖全部场景（AS IS / TO BE总体 / Tiered-Cache详图）
- AS IS区分Store模式（痛点）和P2P模式（优化空间）
- TO BE为单一总体视图，包含所有优化点和痛点解决对照
- PlantUML + SVG渲染，便于版本管理和文档嵌入

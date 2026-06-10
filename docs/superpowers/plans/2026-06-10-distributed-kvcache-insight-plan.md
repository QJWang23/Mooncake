# 分布式 KVCache 技术趋势洞察与 openFuyao 规划 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 基于已批准的设计文档，完成一篇 10-15 页的分布式 KVCache 技术趋势洞察报告，覆盖 Mooncake V1-V3/HiCache/HiSparse/MemCache/LMCache/openFuyao 等六大系统的技术分析，并输出 openFuyao 双线落地规划。

**Architecture:** 按 Section 逐节撰写，每节独立研究 + 撰写 + 审校。最终输出为中文 Markdown 文档，含数据来源引用、架构对比表和路线图。

**Tech Stack:** Markdown, Web research, 架构图 (Mermaid/PlantUML)

**Design Doc:** `docs/superpowers/plans/2026-06-10-distributed-kvcache-insight-design.md`

**Output:** `docs/superpowers/insights/2026-06-10-distributed-kvcache-technology-insight.md`

---

### Task 1: 创建文档骨架和 Section 1（引言与核心洞察摘要）

**Files:**
- Create: `docs/superpowers/insights/2026-06-10-distributed-kvcache-technology-insight.md`

**Step 1: 创建 insights 目录**

Run: `mkdir -p docs/superpowers/insights`

**Step 2: 撰写文档骨架和 Section 1**

按照设计文档 Section 1 的结构撰写，包括：
- 文档标题、摘要、目录结构
- 背景概述（2-3 句 LLM 推理 KVCache 瓶颈）
- 核心论点：分布式 KVCache 从传输管道到智能存储系统的演进
- 关键数据点表格（含来源引用 URL）：
  - Mooncake Store + vLLM 3.8x 吞吐 (vLLM Blog 2026-05-06)
  - HiCache 6x 吞吐 (SGLang Blog 2025-09-10)
  - HiSparse 5x 吞吐 (SGLang Blog 2026-04-10)
  - LMCache CacheBlend ~100% 命中率 (EuroSys 2025)
  - LMCache+Mooncake TTFT-69.1% (LMCache Blog)
  - openFuyao InferNex 22.08% E2EL 改善 (openFuyao v26.03)
- 一句话结论

**Step 3: 验证**

- 检查所有数据点的来源引用是否完整
- 检查文档结构是否与设计文档一致
- 确认篇幅约 1-1.5 页（executive summary 层级）

**Step 4: Commit**

```bash
git add docs/superpowers/insights/2026-06-10-distributed-kvcache-technology-insight.md
git commit -m "docs(insight): add Section 1 - executive summary with data sources"
```

---

### Task 2: 撰写 Section 2（技术演进趋势）

**Files:**
- Modify: `docs/superpowers/insights/2026-06-10-distributed-kvcache-technology-insight.md`

**Step 1: 深度研究四大趋势**

针对每个趋势，从以下来源补充技术细节：

**趋势 1（存储层级深化）：**
- 阅读 Mooncake V1 论文 (arXiv 2407.00079) 的架构设计
- 阅读 HiCache 博客的 GPU 辅助 I/O 内核设计
- 阅读 LMCache 架构文档的 NUMA 感知分配
- 关键问题：各系统如何定义存储层级？层间迁移策略有何差异？

**趋势 2（注意力机制多样化）：**
- 阅读 Mooncake Store 的布局处理器代码（kvcache_layout.h, mha/gqa/mla/hybrid_layout_handler.*）
- 阅读 HiSparse 博客的稀疏注意力优化策略
- 关键问题：MLA 压缩 4-8x 的技术原理？稀疏注意力如何影响存储格式？

**趋势 3（异构硬件生态）：**
- 阅读 Mooncake TE 的传输后端代码（RDMA/HIP/Ascend/MUSA transport）
- 研究 MemCache 的 Ascend 原生互连设计
- 关键问题：异构集群的 KVCache 格式兼容性挑战？

**趋势 4（生态集成深化）：**
- 研究 vLLM KV Connector 和 SGLang RadixAttention 的集成模式
- 研究 HiCache 的插件式后端接口（3 函数设计）
- 关键问题：集成深度如何影响缓存命中率？

**Step 2: 撰写 Section 2**

每个趋势按以下结构撰写（约 1-1.5 页/趋势）：
- 演进脉络（时间线 + 关键事件）
- 关键洞察（技术原理 + 数据支撑）
- 趋势判断（未来方向 + 1-2 年预测）
- 来源引用

**Step 3: 验证**

- 检查每个趋势是否有至少 3 个具体技术细节
- 检查来源引用完整性
- 确认总篇幅约 4-6 页

**Step 4: Commit**

```bash
git add docs/superpowers/insights/2026-06-10-distributed-kvcache-technology-insight.md
git commit -m "docs(insight): add Section 2 - technology evolution trends"
```

---

### Task 3: 撰写 Section 3（生态格局与竞合分析）

**Files:**
- Modify: `docs/superpowers/insights/2026-06-10-distributed-kvcache-technology-insight.md`

**Step 1: 补充生态研究**

- Mooncake 社区：GitHub stars, contributors, PyTorch 生态地位, FAST 2025 Best Paper
- SGLang/HiCache 社区：LMSYS/UC Berkeley 背景, 企业用户（蚂蚁/Novita/阿里云 Tair）
- LMCache 社区：Tensormesh 公司背景, EuroSys 2025 Best Paper, 与 Mooncake 战略合作
- MemCache：华为 Ascend 内部定位, vLLM-Ascend RFC #6410
- openFuyao：华为/中国移动/中国联通联盟, InferNex v26.03 发布数据

**Step 2: 撰写定位矩阵**

从以下维度对比六大系统（表格格式）：
- 核心定位（一句话）
- 技术栈层级（底层/中层/上层）
- 推理引擎支持（vLLM/SGLang/其他）
- 硬件生态（NVIDIA/AMD/Ascend/其他）
- 存储层级定义
- 开源协议
- 社区活跃度（定性评估）

**Step 3: 绘制竞合关系图**

用 Mermaid 语法绘制竞合关系图，标注：
- 合作关系（Mooncake↔LMCache, Mooncake↔HiCache）
- 竞争关系（HiCache↔LMCache, Mooncake↔MemCache）
- 上下游关系（openFuyao→Mooncake）

**Step 4: 撰写关键判断**

3 个关键判断，每个包含：
- 判断结论（1 句）
- 证据支撑（2-3 句）
- 对 openFuyao 的启示（1 句）

**Step 5: 验证**

- 检查定位矩阵每个维度是否有来源依据
- 检查竞合关系图的准确性
- 确认篇幅约 2-3 页

**Step 6: Commit**

```bash
git add docs/superpowers/insights/2026-06-10-distributed-kvcache-technology-insight.md
git commit -m "docs(insight): add Section 3 - ecosystem landscape and competitive analysis"
```

---

### Task 4: 撰写 Section 4（架构深度对比）

**Files:**
- Modify: `docs/superpowers/insights/2026-06-10-distributed-kvcache-technology-insight.md`

**Step 1: 深入代码和文档进行架构对比**

需要深入研究的源码/文档：

**存储层级：**
- `mooncake-store/src/` 目录结构
- HiCache 设计文档 (`docs.sglang.ai/advanced_features/hicache_design.html`)
- LMCache 架构文档 (`docs.lmcache.ai/developer_guide/architecture.html`)

**传输引擎：**
- `mooncake-transfer-engine/src/` 传输后端列表
- MemCache MemFabric 设计

**注意力机制：**
- `mooncake-store/include/*_layout_handler.h` 布局处理器
- HiSparse CUDA 内核设计

**推理引擎集成：**
- vLLM KV Connector 接口
- SGLang HiRadixTree 设计

**Step 2: 撰写四个维度对比表**

每个维度一个对比表 + 关键洞察段落：
- 4.1 存储层级设计对比（层级数、定义、淘汰策略、层间迁移）
- 4.2 传输引擎设计对比（支持后端、关键能力）
- 4.3 注意力机制适配对比（MHA/GQA/MLA/Hybrid/DSA 支持矩阵）
- 4.4 推理引擎集成深度对比（vLLM/SGLang/其他）

**Step 3: 验证**

- 检查每个表格数据的准确性（与源码/文档对照）
- 检查关键洞察是否有数据支撑
- 确认篇幅约 2-3 页

**Step 4: Commit**

```bash
git add docs/superpowers/insights/2026-06-10-distributed-kvcache-technology-insight.md
git commit -m "docs(insight): add Section 4 - architecture deep comparison"
```

---

### Task 5: 撰写 Section 5（openFuyao 差异化定位与突破方向）

**Files:**
- Modify: `docs/superpowers/insights/2026-06-10-distributed-kvcache-technology-insight.md`

**Step 1: 研究 openFuyao 现状**

- 阅读 openFuyao 官网和 v26.03 发布公告
- 阅读 InferNex 组件文档（Hermes-router、弹性扩展器、Eagle-eye）
- 阅读 `docs/superpowers/plans/2026-06-08-npu-lingqu-kvcache-technical-proposal.md`（NPU KVCache 提案）
- 阅读 `docs/superpowers/plans/2026-04-13-mooncake-technical-advantages.md`（Mooncake 技术优势分析）
- 检查 git log 中 openFuyao 相关的已有贡献

**Step 2: 撰写现状诊断**

优势（4 点）和差距（4 点），每点需要：
- 具体事实/数据支撑
- 与竞品的对比参照
- 来源引用

**Step 3: 撰写差异化定位**

- 核心论点："异构推理的云原生编排层"而非"另一个 Mooncake"
- 定位公式：异构硬件编排层 + 云原生治理层 + KVCache 存储优化贡献者
- 与 Mooncake/MemCache 的分工边界

**Step 4: 撰写四大突破方向**

每个方向包括：
- 方向名称和定位（1 句）
- 技术路径（3-5 句，具体技术方案）
- 可参考的现有系统（对标）
- 预期成果（1-2 句）
- 优先级（P0/P1/P2）

**Step 5: 验证**

- 检查诊断结论是否有数据支撑
- 检查突破方向是否与前文趋势分析一致
- 检查与 MemCache 的分工是否清晰
- 确认篇幅约 2-3 页

**Step 6: Commit**

```bash
git add docs/superpowers/insights/2026-06-10-distributed-kvcache-technology-insight.md
git commit -m "docs(insight): add Section 5 - openFuyao positioning and breakthroughs"
```

---

### Task 6: 撰写 Section 6（双线规划路线图）

**Files:**
- Modify: `docs/superpowers/insights/2026-06-10-distributed-kvcache-technology-insight.md`

**Step 1: 制定具体任务分解**

**上游贡献线（Q3-Q4 2026）：**
- Q3: NPU 布局处理器贡献 → 具体文件和接口定义
- Q3: 热缓存优化增强 → 具体优化方向
- Q4: 异构传输测试与优化 → 具体测试场景
- Q4: 稀疏注意力适配 → 具体布局处理器设计

**自研体系线（Q3 2026 - Q2 2027）：**
- Q3-Q4: InferNex KVCache 增强 → Hermes-router 扩展点
- Q3-Q4: 云原生 KVCache Operator → K8s CRD 设计
- Q1 2027: 异构集群互通 → 格式转换层设计
- Q1-Q2 2027: 智能缓存调度 → 预测模型选型

**Step 2: 撰写路线图**

每个任务包括：
- 时间段
- 任务名称和描述
- 验收标准（可量化）
- 依赖关系
- 关联的技术趋势

**Step 3: 撰写关键里程碑**

4 个里程碑（M1-M4），每个包括：
- 时间点
- 里程碑描述
- 验收标准
- 前置依赖

**Step 4: 撰写风险与依赖**

- 技术风险（2-3 项）+ 缓解策略
- 社区风险（1-2 项）+ 缓解策略
- 竞争风险（1-2 项）+ 缓解策略
- 生态风险（1-2 项）+ 缓解策略

**Step 5: 用 Mermaid 绘制时间线图**

用 Mermaid gantt 或 timeline 语法绘制双线并进的可视化路线图。

**Step 6: 验证**

- 检查任务之间的依赖关系是否合理
- 检查时间线是否可行（考虑资源约束）
- 检查风险缓解策略是否具体可执行
- 确认篇幅约 2-3 页

**Step 7: Commit**

```bash
git add docs/superpowers/insights/2026-06-10-distributed-kvcache-technology-insight.md
git commit -m "docs(insight): add Section 6 - dual-track planning roadmap"
```

---

### Task 7: 整体审校与定稿

**Files:**
- Modify: `docs/superpowers/insights/2026-06-10-distributed-kvcache-technology-insight.md`

**Step 1: 全文审校**

检查以下方面：
- **数据一致性**: Section 1 的数据点是否与后文详细分析一致
- **来源完整性**: 所有数据点是否有来源引用
- **逻辑连贯性**: 6 个 Section 之间的逻辑链条是否完整
- **格式统一性**: 表格、标题、引用格式是否一致
- **术语一致性**: 同一概念是否使用统一术语
- **篇幅控制**: 总篇幅 10-15 页（不含附录）

**Step 2: 添加附录**

- 术语表（已有设计文档中的术语）
- 参考文献列表（所有引用的 URL 和论文）
- 可选：架构图列表

**Step 3: 最终验证**

- 运行 `wc -w` 检查总字数（中文约 8000-15000 字）
- 确认所有链接可访问
- 检查 markdown 渲染效果

**Step 4: Commit**

```bash
git add docs/superpowers/insights/2026-06-10-distributed-kvcache-technology-insight.md
git commit -m "docs(insight): finalize distributed KVCache technology insight report"
```

---
title: 分布式 KVCache 技术趋势洞察与 openFuyao 规划
date: 2026-06-10
status: approved
audience: 技术团队 + 管理层
scope: 上游 Mooncake 贡献 + openFuyao 自研体系
---

# 分布式 KVCache 技术趋势洞察与 openFuyao 规划 — 设计文档

## 文档目标

形成一篇面向技术团队和管理层的分布式 KVCache 技术趋势洞察报告，结合 Mooncake 上游社区（V1/V2/V3 架构、HiCache+SGLang、HiSparse、MemCache、LMCache 等组件）的技术演进，以及 openFuyao 在 KVCache 上的当前工作和差异化定位，输出双线（上游贡献 + 自研体系）落地规划。

## 文档架构：方案 B（问题域驱动）

```
Section 1: 引言与核心洞察摘要（1页 executive summary）
Section 2: 技术演进趋势（四大关键趋势）
Section 3: 生态格局与竞合分析（定位矩阵 + 竞合关系图）
Section 4: 架构深度对比（四维度横向对比）
Section 5: openFuyao 差异化定位与突破方向
Section 6: 双线规划路线图（上游贡献 + 自研体系）
```

---

## Section 1: 引言与核心洞察摘要

### 背景（2-3 句）

LLM 推理中 KVCache 管理已成为核心瓶颈——它占用 GPU HBM 的 60-80%，长上下文场景下可达数十 GB。分布式 KVCache 通过分离存储和传输 KVCache，显著降低重计算开销和推理延迟。

### 核心论点

**分布式 KVCache 正从"PD 分离的传输管道"演变为"多层级、多注意力机制、异构硬件的智能存储系统"。** 这一演进体现在存储层级深化、注意力机制多样化、异构硬件支持和生态集成深化四个维度。

### 关键数据点（含来源）

| 数据点 | 来源 |
|--------|------|
| Mooncake Store + vLLM 实现 3.8x 吞吐提升、46x TTFT 降低 | [vLLM Blog, 2026-05-06](https://vllm.ai/blog/2026-05-06-mooncake-store) |
| Mooncake 为 Kimi K2 在 128xH200 上实现 224k/288k tokens/sec (prefill/decode) | [Mooncake GitHub README](https://github.com/kvcache-ai/Mooncake/) |
| HiCache 实现最高 6x 吞吐提升、80% TTFT 降低 | [SGLang Blog, 2025-09-10](https://lmsys.org/blog/2025-09-10-sglang-hicache/) |
| 蚂蚁集团使用 DeepSeek-R1-671B + Mooncake Store 后端 TTFT 降低 84% | [SGLang HiCache Blog](https://lmsys.org/blog/2025-09-10-sglang-hicache/) |
| HiSparse 在 GLM-5.1 长上下文场景实现 5x 吞吐提升 | [SGLang Blog, 2026-04-10](https://lmsys.org/blog/2026-04-10-sglang-hisparse/) |
| LMCache CacheBlend 在 RAG 场景接近 100% KVCache 命中率，获 EuroSys 2025 Best Paper | [EuroSys 2025](https://dl.acm.org/doi/10.1145/3693.comfortable), LMCache GitHub |
| LMCache + Mooncake 在 8xH800 Qwen2.5-72B 上 TTFT 降低 69.1%、吞吐提升 191% | [LMCache Blog](https://blog.lmcache.ai) |
| openFuyao InferNex PD KVCache 感知路由实现 22.08% E2EL 改善 | [openFuyao v26.03 Release](https://www.openfuyao.cn/zh/blogs/blogsList/openFuyao-26-03-released/) |

### 一句话结论

KVCache 生态正从单一项目竞争走向分层协作，openFuyao 在异构 NPU 场景有独特定位，应聚焦"异构推理的云原生编排层"而非重复造轮子。

---

## Section 2: 技术演进趋势 — 四大关键趋势

### 趋势 1：从 PD 分离到 Tiered Cache（存储层级深化）

**演进脉络:**
- Mooncake V1（2024.06）: 纯 PD 分离架构，KVCache 通过 RDMA 直接传输
- Mooncake V2（2024.11）: Transfer Engine 抽象传输层，支持 TCP/RDMA/CXL/NVMe-oF
- Mooncake V3（2025.03）: Mooncake Store 多级存储引擎，GPU→DRAM→SSD 层级
- HiCache（2025.09）: 标准化三层模型（GPU HBM / CPU DRAM / 远程存储），GPU 辅助 I/O 内核达 3x cudaMemcpy 吞吐
- LMCache（2025+）: 四层扩展（+本地 NVMe GDS），NUMA 感知分配

**关键洞察:** HiCache 的 GPU 辅助 I/O 内核是独特创新；LMCache 的 NVMe GDS 支持适合大规模持久化。未来 CXL 内存、持久内存等新介质将加入层级体系。

### 趋势 2：从全量注意力到稀疏/混合注意力适配

**演进脉络:**
- 早期: MHA 统一格式（每 head 独立 K/V）
- GQA（GLM-4/Qwen）: KV 组共享，减少内存
- MLA（DeepSeek V3）: 压缩潜在向量，4-8x 存储缩减
- Hybrid（Qwen3.5+）: 滑动窗口 + 全局注意力混合
- DSA 稀疏注意力（DeepSeek V3.2/GLM-5.1）: 仅关注 KV 子集

**关键洞察:** HiSparse 仅保留稀疏注意力的活跃 KV 子集在 GPU 上，实现 5x 吞吐提升。KVCache 系统需要可插拔的布局适配层（Mooncake Store 已有 MHA/GQA/MLA/Hybrid 四种布局处理器）。

### 趋势 3：从同构 GPU 到异构硬件生态

**演进脉络:**
- NVIDIA GPU（主力）
- AMD GPU（ROCm/HIP 支持）
- 华为 NPU（HCCL/ADXL 支持）
- 异构集群（Ascend Prefill + H100 Decode 可行）
- Moore Threads GPU（MUSA 支持）

**关键洞察:** MemCache 针对 Ascend 原生互连（device_rdma/sdma/host_urma）优化；Mooncake TE 已覆盖 6+ 传输协议。异构推理在中国市场是刚需，NPU 生态需要与 GPU 互通。

### 趋势 4：从独立组件到生态集成

**演进脉络:**
- 独立 KVCache 存储（早期 Mooncake）
- 推理引擎集成（vLLM KV Connector、SGLang HiRadixTree）
- 全栈协作（LMCache 作为 vLLM-Mooncake 桥接层）
- 插件式后端（HiCache 仅需 3 个函数即可集成新后端）

**关键洞察:** 集成深度正从"put/get 接口"向"注意力感知决策"演进。LMCache 定位为"知识交付网络（KDN）"，HiCache 与推理引擎的 RadixAttention 深度绑定。

---

## Section 3: 生态格局与竞合分析

### 3.1 定位矩阵

| 维度 | Mooncake | HiCache+SGLang | LMCache | MemCache | openFuyao/InferNex |
|------|----------|----------------|---------|----------|---------------------|
| 核心定位 | 分布式KVCache存储引擎 + 传输引擎 | 分层KV缓存系统 | KVCache管理层(KDN) | Ascend分布式KVCache引擎 | 云原生AI推理基础设施 |
| 技术栈层级 | 底层传输+存储 | 推理引擎内层 | 推理引擎与存储之间 | 底层传输+存储(Ascend) | 上层编排+调度+存储 |
| 推理引擎 | vLLM/SGLang/TRT-LLM/LMDeploy | SGLang | vLLM | vLLM-Ascend | vLLM/vLLM-Ascend |
| 硬件生态 | NVIDIA/AMD/Ascend/MooreThreads | NVIDIA(主力) | NVIDIA | Ascend NPU | x86/ARM/GPU/NPU |
| 存储层级 | GPU→DRAM→SSD(RDMA) | GPU→CPU→远程存储 | GPU→CPU→NVMe→远程 | 设备→主机→远程(Ascend RDMA) | 分布式池化存储 |
| 开源协议 | MIT | Apache 2.0 | Apache 2.0 | 华为内部 | Apache 2.0 |
| 社区活跃度 | PyTorch生态, FAST 2025 Best Paper | LMSYS/UC Berkeley 背书 | Tensormesh 公司运营 | 华为内部驱动 | openFuyao社区 |

### 3.2 竞合关系

- **Mooncake ↔ LMCache**: 战略合作（LMCache 作为 vLLM-Mooncake 连接层，2025.05 宣布合作）
- **Mooncake ↔ HiCache**: 互补（Mooncake Store 是 HiCache 的远程存储后端之一）
- **Mooncake ↔ MemCache**: 竞争（同类底层存储引擎，不同硬件平台）
- **openFuyao ↔ Mooncake**: 上游贡献 + 下游集成（热缓存优化已合并上游）
- **HiCache ↔ LMCache**: 竞争（都做分层缓存，分别绑定 SGLang 和 vLLM）
- **HiSparse → HiCache**: 承继关系（相同分层理念，应用于稀疏注意力场景）

### 3.3 关键判断

1. **底层存储引擎趋于收敛**: Mooncake Store 成为主流（PyTorch 生态、多硬件支持），MemCache 在 Ascend 生态有独立价值
2. **上层管理层继续竞争**: HiCache（SGLang 绑定）与 LMCache（vLLM 绑定）的竞争本质是推理引擎生态竞争
3. **异构硬件是中国市场独特变量**: NPU 生态需要独立但与 GPU 互通的方案

---

## Section 4: 架构深度对比

### 4.1 存储层级设计对比

| 系统 | 层级数 | 层级定义 | 淘汰策略 | 层间迁移 |
|------|--------|----------|----------|----------|
| Mooncake Store | 3 | HBM→DRAM→SSD | 应用控制 | RDMA 传输 |
| HiCache | 3 | GPU→CPU→远程 | 分层重叠+预取 | GPU辅助内核(3x cudaMemcpy) |
| LMCache | 4 | GPU→CPU→本地NVMe→远程 | LRU+分块(256 tokens) | 异步+NUMA感知 |
| MemCache | 3 | 设备→主机→远程 | 多副本 | Ascend互连(SDMA/RDMA) |

**关键洞察:** HiCache 的 GPU 辅助 I/O 内核是独特创新，吞吐达标准 cudaMemcpy 的 3 倍；LMCache 的 NVMe GDS 和 NUMA 感知适合大规模持久化场景。

### 4.2 传输引擎设计对比

| 系统 | 支持后端 | 关键能力 |
|------|----------|----------|
| Mooncake TE | TCP/RDMA/NVLink/CXL/NVMe-oF + 异构(HCCL/ADXL) | 拓扑感知、多NIC聚合、零拷贝 |
| HiCache | 插件式（3函数接口） | Mooncake/3FS/NIXL/文件后端 |
| LMCache | NIXL/Redis/Mooncake | 点对点通道 |
| MemCache | MemFabric(device_rdma/sdma/host_urma) | Ascend原生 |

**关键洞察:** Mooncake TE 覆盖 6+ 传输协议，是通用性最强的传输引擎；MemCache 的 MemFabric 在 Ascend 硬件上有原生优势。

### 4.3 注意力机制适配对比

| 系统 | MHA | GQA | MLA | Hybrid(滑动窗口) | 稀疏注意力(DSA) |
|------|-----|-----|-----|--------|------------|
| Mooncake Store | ✅ | ✅ | ✅ | ✅ | 规划中 |
| HiCache | ✅ | ✅ | 有限 | ❌ | ❌ |
| LMCache | ✅ | ✅ | 有限 | ❌ | ❌ |
| HiSparse | ❌ | ❌ | ❌ | ❌ | ✅(DSA专项) |

**关键洞察:** Mooncake Store 在多注意力机制适配方面领先，已有 MHA/GQA/MLA/Hybrid 四种布局处理器，这是核心差异化优势。稀疏注意力适配是下一个竞争焦点。

### 4.4 推理引擎集成深度对比

| 系统 | vLLM | SGLang | 其他 |
|------|------|--------|------|
| Mooncake Store | KV Connector | HiCache后端 | TRT-LLM/LMDeploy |
| HiCache | 间接 | 原生(RadixAttention) | - |
| LMCache | 原生(Connector) | 间接 | - |
| openFuyao | vLLM-Ascend | 间接 | InferNex套件 |

**关键洞察:** 推理引擎集成正从"put/get 接口"向"注意力感知决策"演进。SGLang 的 RadixAttention 和 vLLM 的 KV Connector 是两种不同的集成哲学。

---

## Section 5: openFuyao 差异化定位与突破方向

### 5.1 现状诊断

**优势:**
- 异构硬件原生支持：InferNex 原生适配 Ascend NPU + Kunpeng CPU
- 云原生编排能力：Hermes-router 智能路由、弹性扩展器、Eagle-eye 可观测性
- 超大规模实战：10,000+ 节点调度，运营商级生产经验
- 已贡献上游：热缓存优化等特性已合并 Mooncake 上游

**差距:**
- 存储引擎层依赖：底层 KVCache 存储仍依赖 Mooncake Store（GPU 优化为主）
- 注意力机制适配：MLA/稀疏注意力等新机制适配深度不足
- 社区影响力：相比 Mooncake（PyTorch 生态）和 SGLang（LMSYS 背书）有差距
- 生态绑定：强绑定 Ascend 硬件，跨厂商互通能力需加强

### 5.2 差异化定位

**核心论点：openFuyao 不应成为"另一个 Mooncake"，而应成为"异构推理的云原生编排层"。**

定位公式：
```
openFuyao/InferNex = 异构硬件编排层 + 云原生治理层 + KVCache 存储优化贡献者
```

### 5.3 四大突破方向

**方向 1：NPU 原生 KVCache 优化（差异化护城河）**
- 针对 Ascend NPU 的 HBM/DRAM 访问模式优化
- 借鉴 HiSparse"活跃子集驻留"思路，针对 NPU 稀疏注意力优化
- 目标：成为 Ascend 生态的 KVCache 标准实现

**方向 2：异构集群跨厂商 KVCache 互通（生态桥梁）**
- 实现 Ascend Prefill + NVIDIA Decode 的 KVCache 格式转换与高效传输
- 在 Mooncake TE 异构传输基础上补充格式适配层
- 目标：成为异构推理的事实标准

**方向 3：云原生 KVCache 治理（管理层突破）**
- KVCache 生命周期管理 K8s Operator（预热/淘汰/迁移/压缩）
- 基于流量预测的主动缓存调度（Hermes-router 扩展）
- 目标：从"组件提供者"升级为"治理平台"

**方向 4：上游贡献战略（生态共建）**
- NPU 优化、热缓存、异构适配等核心贡献回流 Mooncake
- 在 Mooncake Store 布局处理器框架中贡献 NPU 专用处理器
- 目标：成为 Mooncake 核心 Maintainer 之一

---

## Section 6: 双线规划路线图

### 6.1 上游贡献线（短期 3-6 个月）

| 阶段 | 任务 | 目标 | 关联趋势 |
|------|------|------|----------|
| Q3 2026 | NPU 布局处理器贡献 | 将 Ascend 专用 KVCache 格式适配器贡献到 Mooncake Store | 趋势 2/3 |
| Q3 2026 | 热缓存优化增强 | 扩展已有热缓存特性，增加 NPU 场景优化 | 趋势 1 |
| Q4 2026 | 异构传输测试与优化 | 补充 Ascend+NVIDIA 异构场景集成测试和性能基准 | 趋势 3 |
| Q4 2026 | 稀疏注意力适配 | 参考 HiSparse，为 Mooncake Store 贡献稀疏注意力布局处理器 | 趋势 2 |

### 6.2 自研体系线（中期 6-12 个月）

| 阶段 | 任务 | 目标 | 关联趋势 |
|------|------|------|----------|
| Q3-Q4 2026 | InferNex 分布式 KVCache 增强 | Hermes-router 增加 KVCache 感知调度策略 | 趋势 4 |
| Q3-Q4 2026 | 云原生 KVCache Operator | K8s Operator 管理 KVCache 生命周期 | 趋势 4 |
| Q1 2027 | 异构集群 KVCache 互通 | Ascend↔NVIDIA KVCache 格式转换与传输 | 趋势 3 |
| Q1-Q2 2027 | 智能缓存调度 | 基于流量预测的主动缓存调度 | 趋势 1/4 |

### 6.3 关键里程碑

- **M1（2026 Q3）**: 成为 Mooncake 社区活跃贡献者，NPU 适配器合并
- **M2（2026 Q4）**: InferNex KVCache 增强版发布，性能对标 Mooncake Store GPU 版
- **M3（2027 Q1）**: 异构集群互通 PoC 验证
- **M4（2027 Q2）**: 完整的云原生 KVCache 治理平台发布

### 6.4 风险与依赖

- **技术风险**: Ascend 硬件更新可能导致适配工作需要持续投入
- **社区风险**: Mooncake 社区贡献审核周期可能较长
- **竞争风险**: MemCache 作为 Ascend 原生方案可能与 openFuyao 定位冲突 → 应对策略：与 MemCache 团队明确分工（存储引擎 vs 编排层）
- **生态风险**: vLLM-Ascend 与 SGLang 对 Ascend 的支持成熟度

---

## 附录：术语表

| 术语 | 全称 | 说明 |
|------|------|------|
| PD Disaggregation | Prefill-Decode Disaggregation | 预填充-解码分离架构 |
| TE | Transfer Engine | Mooncake 高性能数据传输引擎 |
| MHA | Multi-Head Attention | 多头注意力 |
| GQA | Grouped Query Attention | 分组查询注意力 |
| MLA | Multi-Head Latent Attention | 多头潜在注意力（DeepSeek） |
| DSA | DeepSeek Sparse Attention | DeepSeek 稀疏注意力 |
| KDN | Knowledge Delivery Network | 知识交付网络（LMCache 概念） |
| HBM | High Bandwidth Memory | 高带宽内存（GPU/NPU） |
| CXL | Compute Express Link | 计算互连协议 |
| GDS | GPUDirect Storage | GPU 直接存储访问 |

# Mooncake 技术分析与竞争优势

> 日期：2026-04-13
> 版本：v1.0

---

## 1. 架构全景

![Mooncake Architecture](assets/mooncake-architecture.svg)

Mooncake 是一个以 KVCache 为中心的分离式 LLM 推理架构，核心设计目标是将 Prefill 和 Decode 集群解耦，同时利用分布式存储引擎在 CPU DRAM、SSD 等低成本资源上构建大规模 KVCache 池，通过高性能传输引擎实现零拷贝、带宽饱和的跨节点数据传输。

架构由四层组成：

- **推理引擎层**：vLLM、SGLang、TensorRT-LLM、LMCache、LMDeploy 等主流引擎的原生集成
- **Mooncake Store 层**：分布式 KVCache 存储引擎，包含 Master Service（元数据管理、副本协调、HA 演进中）和 Store Client（Put/Get/Remove 语义、Lease 保护、Soft Pin）
- **Transfer Engine 层**：统一传输抽象，支持 RDMA、NVLink、Ascend Direct、HIP、CXL、NVMe-oF、灵衢/UB（进行中）等多种传输后端
- **硬件平台层**：覆盖 NVIDIA GPU、华为 Ascend NPU、AMD GPU、摩尔线程 MTHREADS 等多厂商异构硬件

源文件：[`assets/mooncake-architecture.puml`](assets/mooncake-architecture.puml)

---

## 2. 关键技术点

### 2.1 分层存储与内存管理

Mooncake 采用三级存储分层架构：

| 层级 | 存储 | 角色 | 带宽特征 |
|------|------|------|----------|
| **L1** | GPU VRAM / NPU HBM | 热数据，活跃 KVCache | 1.6-4 TB/s |
| **L2** | CPU DRAM | 温数据，KVCache 池 | ~100 GB/s |
| **L3** | SSD / NVMe-oF | 冷数据，容量扩展 | ~10 GB/s |

**OffsetBufferAllocator**：针对 LLM 推理中 KVCache 块尺寸高度统一的特征，采用 Bin-based 分配策略，在 power-of-2 场景下达到 **>99% 内存利用率**，分配延迟仅 100-500ns。

**驱逐策略**：近似 LRU + Lease 保护 + Soft Pin 三重机制。正在读写的对象不会被驱逐（Lease），系统提示词等重要对象可标记为 Soft Pin 优先保留。高水位线（默认 95%）触发自动驱逐。

**不可变对象语义**：Put 成功后对象不可修改，简化了一致性保证，消除了读写冲突的复杂度。

### 2.2 高性能传输引擎

Transfer Engine 是 Mooncake 的核心竞争力之一：

**拓扑感知路径选择**：自动检测 NUMA/GPU/NPU 拓扑，基于拓扑矩阵选择最优传输路径，避免跨 NUMA 访问的带宽损失。

**多 NIC 聚合**：跨多个网卡并行传输，突破单 NIC 带宽上限。实测 RDMA 聚合带宽达 **19.87 GiB/s**，超过单网卡容量。

**Segment 统一抽象**：将 DRAM、VRAM、HBM、NVMe 统一表示为连续地址空间，BatchTransfer 支持异步 scatter/gather 操作。

**连接池管理**：SIEVE 算法管理 Endpoint 池，高效复用连接资源。

### 2.3 去中心化元数据演进

当前 Master Service 采用中心化架构，存在单点风险。**社区正在积极推进 HA 架构演进**，核心方向包括：

- **多 Master 故障切换**：主备 Master 自动切换，消除单点故障
- **元数据复制**：Master 状态实时同步到备用节点
- **水平扩展**：多 Master 分片管理不同 Object Space

这一演进完成后，中心化架构的可扩展性顾虑将被消除，同时保持现有实现的简洁性。

### 2.4 多厂商异构传输

Transfer Engine 的模块化 Transport 接口设计使其成为当前**唯一同时覆盖四类 GPU/NPU 平台的分布式传输引擎**：

| 传输后端 | 硬件 | 协议 | 状态 |
|----------|------|------|------|
| RDMA Transport | NVIDIA GPU | IB/RoCEv2/eRDMA/GPUDirect | 生产就绪 |
| NVLink Transport | NVIDIA GPU | NVLink MNNVL | 生产就绪 |
| HIP Transport | AMD GPU | ROCm/HIP | 生产就绪 |
| Ascend HCCL | 华为 NPU | HCCL 集合通信 | 生产就绪 |
| Ascend Direct | 华为 NPU | ADXL/HIXL 单边通信 | 生产就绪 |
| 异构 Transport | NPU+GPU | Ascend ↔ NVIDIA | 生产就绪 |
| MUSA Transport | 摩尔线程 GPU | MUSA | 生产就绪 |
| UB/URMA Transport | 昆鹏 CPU | 灵衢 UB（RFC #1773） | 开发中 |
| CXL Transport | 通用 | CXL 协议 | 开发中 |

---

## 3. 竞争优势分析

### 3.1 多厂商硬件兼容性

Mooncake 的 Transfer Engine 抽象层实现了一套代码覆盖四类异构硬件，这在分布式 KVCache 领域是**独一无二的能力**。

**实际影响**：
- 同一套 KVCache 管理逻辑可在 NVIDIA、Ascend、AMD、MTHREADS 上运行
- 混合 GPU/NPU 集群（如 910B Prefill + H100 Decode）的异构传输已部分实现
- 新硬件只需实现 Transport 接口即可接入整个生态
- 保护用户硬件投资，避免被单一厂商锁定

### 3.2 推理引擎生态覆盖

Mooncake 原生集成了 5+ 主流推理引擎：

| 推理引擎 | 集成方式 | 场景 |
|----------|---------|------|
| **vLLM** | KV Connector | Prefill-Decode 分离 |
| **SGLang** | HiCache Backend | 分层 KV Cache |
| **TensorRT-LLM** | KV 传输后端 | 高性能推理 |
| **LMCache** | 增强 KV 管理 | KV Cache 管理 |
| **LMDeploy** | PD 分离后端 | Prefill-Decode 分离 |
| **veRL** | TransferQueue 后端 | RL 训练数据传输 |
| **InferNex (openFuyao)** | 核心组件 | 云原生昇腾推理 |

这种广覆盖意味着用户无需更换推理引擎即可采用 Mooncake，降低了迁移成本。

### 3.3 生产级成熟度

Mooncake 是当前分布式 KVCache 领域**经过最大规模生产验证**的开源方案：

- **支撑 Kimi（月之暗面）**：日均亿级请求的生产负载
- **SLA 保证**：Production 级别的可靠性和可用性
- **长期运行验证**：在真实用户流量下持续优化

生产环境的持续打磨使得 Mooncake 在边界条件处理、故障恢复、长尾延迟控制等方面积累了大量实战经验。

### 3.4 量化性能收益

基于官方 benchmark 数据：

**Prefill-Decode 分离推理（vLLM + RDMA，A10 GPU，Qwen2.5-7B）**：

| 指标 | Mooncake RDMA | 基线 | 提升 |
|------|---------------|------|------|
| TTFT（单节点） | 250-600ms | 650-850ms | **25-65% 降低** |
| TTFT（多节点） | 250-320ms | 350-380ms | **15-30% 降低** |
| P99 TTFT | — | — | **40-50% 降低** |
| ITL（单节点） | 8-12ms | 11-17ms | **30-40% 降低** |
| ITL（多节点） | 8-8.5ms | 8-10ms | **10-20% 降低** |

**SGLang P/D 分离**：

| 配置 | 吞吐 | ITL | ITL 改善 |
|------|------|-----|----------|
| 1P1D 分离 | 6,160-7,080 tok/s | 7-17ms | **30% 降低** |
| 2 Regular 实例 | 6,160-7,080 tok/s | 10-25ms | 基线 |

**传输引擎带宽**：

| 指标 | 数值 |
|------|------|
| RDMA 多线程聚合带宽 | **19.87 GiB/s** |
| 多 NIC 聚合效果 | 超越单 NIC 线速 |
| 内存分配延迟 | **100-500ns** |
| 内存利用率 | **>99%**（power-of-2 场景） |

### 3.5 云原生生态

通过 openFuyao 社区的 InferNex/xPyD 框架，Mooncake 获得了完整的云原生部署能力：

- **K8s 原生调度**：基于 DRA（Dynamic Resource Allocation）的 NPU 设备管理
- **弹性伸缩**：v26.03 版本新增弹性伸缩与决策系统
- **近实时可观测性**：推理指标实时监控
- **v26.03 优化**：平均首 Token 延迟降低 30%，端到端延迟降低 10%

---

## 4. 昇腾代际硬件构建趋势

### 4.1 当前昇腾适配现状

Mooncake 在昇腾 NPU 上的适配已达到**生产可用**水平：

**三种传输后端已完成**：
- **Ascend HCCL Transport**：基于 HCCL 集合通信，适用于 NPU 集群内通信
- **Ascend Direct Transport（HIXL/ADXL）**：基于 CANN HIXL 库的零拷贝 NPU-to-NPU 传输，支持批量传输，已适配 Mooncake Store 的 "dummy real mode"
- **异构 Transport**：910B NPU Prefill + GPU Decode 混合架构，8MB 数据块聚合优化 HBM-DRAM 传输

**部署验证**：
- vllm-ascend 官方提供 Mooncake Store 和 Mooncake Connector 的完整部署指南
- openFuyao 社区 xPyD 分离推理引擎直接集成 Mooncake KVCache
- 华为 HIXL + Mooncake + vLLM KV Cache Pooling 方案已有公开部署实践

### 4.2 A3 代际（910C / CloudMatrix384）

| 参数 | 规格 |
|------|------|
| 芯片 | Ascend 910C（双 Chiplet） |
| HBM 容量 | **128 GB** |
| HBM 带宽 | **~3.2 TB/s** |
| FP16 算力 | ~800 TFLOPS |
| 集群规模 | 384 NPU + 192 Kunpeng CPU |
| 互联 | UB 1.0（196 GB/s）+ RDMA（200 GB/s，KVCache 专用平面） |

**CloudMatrix384 的双平面架构对 KVCache 的意义**：
- UB 平面负责集群内通用通信
- RDMA 平面**物理隔离专用于 KVCache 传输**，避免带宽争抢
- Mooncake 的 RDMA Transport 可直接利用 RDMA 专用平面

**学术验证**：CloudMatrix384 上的 PDC（Prefill-Decode-Caching）分离推理方案已在 arXiv 公开发表，xDeepServe 论文进一步验证了异构 NPU 混合推理的可行性。

### 4.3 A5 代际（950DT / Atlas 950 SuperPoD）

| 参数 | Ascend 950PR | Ascend 950DT |
|------|-------------|-------------|
| 自研 HBM | HiBL 1.0 | HiZQ 2.0 |
| HBM 容量 | 128 GB | **144 GB** |
| HBM 带宽 | 1.6 TB/s | **4.0 TB/s** |
| FP8 算力 | 1 PFLOPS | 1 PFLOPS |
| FP4 算力 | 2 PFLOPS | 2 PFLOPS |
| 集群规模 | — | **8192 卡** |
| 集群总 HBM | — | **1152 TB** |
| 集群总带宽 | — | **16.3 PB/s** |
| 互联协议 | 灵衢 UB | 灵衢 UB 全互联 |

**A5 对 Mooncake 的战略意义**：

1. **144 GB HBM → 更多 KVCache 驻留本地**：单卡可存储更多请求的 KVCache，减少跨节点传输需求，TTFT 和 ITL 进一步降低。

2. **4 TB/s HBM 带宽 → KVCache 读写不再是瓶颈**：当前 L2（DRAM）到 L1（HBM）的数据加载在 Mooncake 的实测中已经占比很小（<5% TTFT），A5 代际将进一步压缩这一比例。

3. **8192 卡 + 16.3 PB/s → 超大规模推理**：Mooncake 的 HA 架构演进完成后，配合灵衢 UB 全互联，可支撑 8192 卡规模的分布式 KVCache 管理。

4. **灵衢 UB 全互联 → 消除传输瓶颈**：RFC #1773 完成后，Mooncake TE 可通过 URMA API 直接利用灵衢 UB 的 ~50 GB/s 传输能力。

### 4.4 Mooncake 适配灵衢的演进路径

```
Phase 1 (已完成)          Phase 2 (2026 H2)         Phase 3 (2027+)
─────────────────         ──────────────────         ──────────────────
ADXL/HIXL                 UB Transport               灵衢 GVA 全局地址
Direct Transport          via URMA API               空间融合
                          (RFC #1773)

┌──────────────┐         ┌──────────────┐           ┌──────────────┐
│ NPU-NPU      │         │ NPU-NPU      │           │ NPU-NPU      │
│ HIXL 单边    │   →     │ UB 高带宽    │     →     │ GVA 全局     │
│ ~20 GB/s     │         │ ~50 GB/s     │           │ 统一寻址     │
│              │         │              │           │ >100 GB/s    │
│ HCCL 集合    │         │ RDMA 平面    │           │ UB 2.0       │
│ 通信         │         │ 隔离         │           │ 全互联       │
└──────────────┘         └──────────────┘           └──────────────┘
```

### 4.5 TieredCache 与昇腾分层的天然映射

Mooncake 的 TieredCache 架构与昇腾硬件分层存在直接对应关系，无需架构层面的重大修改：

```
Mooncake TieredCache              昇腾硬件
────────────────────              ────────
L1: GPU VRAM / NPU HBM     ←→    Ascend 910C/950DT HBM
L2: CPU DRAM                ←→    Kunpeng DDR
L3: SSD / NVMe-oF           ←→    SSD / NVMe

Transfer Engine 抽象               昇腾传输
────────────────────              ────────
RDMA Transport              ←→    RoCE/RDMA 平面
Ascend Direct Transport     ←→    HIXL/ADXL D2D
UB Transport (RFC #1773)    ←→    灵衢/UB 互联
NVMe-oF Transport           ←→    SSD Fabric
```

这种天然映射意味着 Mooncake 的**所有分层策略、驱逐策略、Lease 保护机制、Soft Pin 等核心功能可直接在昇腾硬件上复用**，无需重新设计。

---

## 5. 总结

Mooncake 的核心竞争力可以归纳为四个维度：

1. **唯一的多厂商统一 TE**：一套代码覆盖 NVIDIA + Ascend + AMD + MTHREADS，保护硬件投资
2. **最广的推理引擎生态**：5+ 主流引擎原生集成，零迁移成本
3. **最高的生产成熟度**：Kimi 亿级日活验证，边界条件和故障恢复经过实战打磨
4. **清晰的昇腾演进路径**：三种传输后端已完成，灵衢 UB RFC 推进中，TieredCache 与昇腾分层天然匹配

**在昇腾 A5（950DT）代际**，144 GB HBM + 4 TB/s 带宽 + 8192 卡 + 16.3 PB/s 集群互联的组合，将使 Mooncake 的分布式 KVCache 管理能力达到新的性能边界。配合社区正在推进的 HA 架构演进，Mooncake 有望在万卡规模下同时保持架构简洁性和高可用性。

---

## 数据来源

- [Mooncake GitHub](https://github.com/kvcache-ai/Mooncake) — 源码分析
- [Mooncake Issue #719](https://github.com/kvcache-ai/Mooncake/issues/719) — Ascend NPU 支持
- [Mooncake Issue #1017](https://github.com/kvcache-ai/Mooncake/issues/1017) — 异构传输
- [Mooncake Issue #1058](https://github.com/kvcache-ai/Mooncake/issues/1058) — RoadMap
- [Mooncake Issue #1773](https://github.com/kvcache-ai/Mooncake/issues/1773) — UB Transport RFC
- [openFuyao 官方](https://www.openfuyao.cn/)
- [HIXL+Mooncake+vLLM 指南](https://zhuanlan.zhihu.com/p/2000213428279740325)
- [Serving LLMs on CloudMatrix384](https://arxiv.org/html/2506.12708v1)
- [xDeepServe Paper](https://arxiv.org/html/2508.02520v1)
- [华为 HC 2025 Keynote](https://www.huawei.com/cn/news/2025/9/hc-xu-keynote-speech)
- [UnifiedBus 官方](https://www.unifiedbus.com/)
- [openEuler UB Service Core 白皮书](https://www.openeuler.org/projects/ub-service-core/white-paper/UB-Service-Core-SW-Arch-RD-2.0-en.pdf)

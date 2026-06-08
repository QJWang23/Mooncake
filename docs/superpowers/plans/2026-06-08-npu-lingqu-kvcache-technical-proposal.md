# 技术立项报告：基于昇腾NPU灵衢高速互联的Mooncake KVCache多级传输优化及推理加速

## 文档信息

| 项目 | 内容 |
|------|------|
| 项目名称 | 基于昇腾NPU灵衢高速互联的KVCache多级传输优化及推理加速 |
| 版本 | v1.0 |
| 日期 | 2026年6月 |
| 状态 | 立项评审 |

---

## 一、项目背景与目标

### 1.1 项目背景

随着大语言模型（LLM）在企业和工业场景中的深度应用，推理性能已成为决定业务效率和成本的核心瓶颈。当前企业级LLM推理面临两大高价值场景的严峻挑战：

- **企业级多轮对话**：以客服系统、知识问答、智能助理为代表，用户会话涉及数轮至数十轮交互，KVCache随对话轮次线性增长，单次会话KVCache可达数GB至数十GB，对缓存容量和访问延迟提出极高要求。
- **Coding Agent**：以代码补全、代码审查、自动化编程为代表，Agent需在长时间任务中维护大量上下文（代码仓库、API文档、历史操作），KVCache生命周期长、访问模式复杂、需频繁跨节点迁移。

现有Mooncake架构在GPU（NVIDIA）+ RDMA网络上已展现出显著优势，但在昇腾NPU（A2/A3/A5）平台及灵衢高速互联总线场景下，仍存在传输路径未充分优化、多级缓存架构未适配、硬件特性未被充分利用等问题。

### 1.2 项目目标

基于昇腾A2、A3/A5 NPU平台及灵衢（LingQu/UB）高速互联总线技术，对Mooncake现有架构进行深度优化和扩展，实现：

1. **Tiered-Cache架构升级**：构建L0-L4五级缓存层次，充分利用灵衢统一编址（GVA）实现跨节点透明内存访问
2. **Dual-Path传输路径优化**：基于灵衢HCCP/HCOM双协议栈实现智能路径选择，超节点内亚微秒级延迟、100GB/s+带宽
3. **企业级场景性能加速**：多轮对话场景TTFT降低30%+，Coding Agent场景KVCache复用率提升50%+
4. **上游社区贡献**：将核心优化回馈Mooncake上游社区，建立技术影响力

### 1.3 TO BE关键技术总结

| # | 关键技术 | 核心说明 |
|---|---------|---------|
| 1 | **五级Tiered-Cache** | 在现有DRAM+SSD二级存储之上，新增L0(NPU HBM)、L1(灵衢HBM池，GVA统一编址超节点共享)、L2(Host DRAM)三层，将缓存命中率从60-70%提升至>90% |
| 2 | **灵衢HCCP设备直连** | 利用A3/A5灵衢UB交换机HCCP协议实现超节点内NPU-to-NPU亚微秒直传，绕过Host DRAM中转，将Store模式4跳简化为L0↔L1的2跳 |
| 3 | **Dual-Path智能传输** | 基于HCCP/HCOM/RDMA三协议栈构建运行时自适应路径选择器，大块传输(>64MB)时HCCP+RDMA双路径并发聚合带宽，替代现有静态协议选择 |
| 4 | **GVA 256TB统一编址** | 通过MemFabric Hybrid BM API实现跨节点内存Import/Export，消除多次地址转换开销，NPU可通过GVA直接访问远端HBM/DRAM |
| 5 | **场景感知缓存策略** | 多轮对话场景采用L1 Soft Pin + HCCP异步预取将TTFT降低30%，Coding Agent场景采用L1 System Prompt永久Pin + ContextGroup跨Agent共享将复用率从40%提升至>80% |
| 6 | **DataCopier层间迁移引擎** | 基于访问热度评分的L0-L4异步迁移引擎，Miss事件触发升级、定期扫描触发降级、显式Pin/Unpin API控制，迁移过程不阻塞推理 |

> 架构图详见：[AS IS痛点图](../assets/asis-architecture-pain-points.svg) | [TO BE总体架构图](../assets/tobe-architecture-solution.svg) | [Tiered-Cache数据流图](../assets/tiered-cache-dataflow.svg)

---

## 二、AS IS：当前架构与技术现状

### 2.1 现有Mooncake架构概览

Mooncake采用KVCache中心化的分离式推理架构（Prefill-Decode Disaggregation），核心组件包括：

```
┌──────────────────────────────────────────────────────────────────┐
│                    Mooncake 现有架构                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐                    ┌─────────────┐             │
│  │ Prefill集群  │                    │ Decode集群   │             │
│  │ (GPU/NPU)    │                    │ (GPU/NPU)    │             │
│  │  ┌────────┐  │   Transfer Engine  │  ┌────────┐  │             │
│  │  │ LLM    │  │ ═════════════════► │  │ LLM    │  │             │
│  │  │ Engine │  │   RDMA/TCP         │  │ Engine │  │             │
│  │  └───┬────┘  │                    │  └───┬────┘  │             │
│  │      │       │                    │      │       │             │
│  │  ┌───▼────┐  │                    │  ┌───▼────┐  │             │
│  │  │KVCache │  │                    │  │KVCache │  │             │
│  │  │ (HBM)  │  │                    │  │ (HBM)  │  │             │
│  │  └────────┘  │                    │  └────────┘  │             │
│  └──────┬───────┘                    └──────┬───────┘             │
│         │                                   │                     │
│         └─────────────┬─────────────────────┘                     │
│                       │                                           │
│              ┌────────▼────────┐                                 │
│              │  Mooncake Store  │                                 │
│              │  (Master+Worker) │                                 │
│              │  - DRAM缓存池    │                                 │
│              │  - SSD持久化     │                                 │
│              │  - 多副本管理    │                                 │
│              └─────────────────┘                                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 现有架构的关键技术特征

| 维度 | 当前实现 |
|------|---------|
| **传输层** | RDMA/TCP双协议，Topology-aware路径选择，多NIC聚合 |
| **存储层** | DRAM/SSD二级存储，OffsetAllocator内存管理 |
| **缓存层** | 内存池 + SSD持久化（DFS），近似LRU淘汰策略 |
| **元数据** | etcd/Redis/HTTP三选一，Master集中管理 |
| **NPU支持** | HCCL Transport、Ascend Direct Transport、Heterogeneous RDMA |

### 2.3 当前架构在企业级场景下的痛点

#### 痛点1：KVCache传输路径冗长，延迟不满足实时交互需求

**现状（AS IS）**：

```
现有数据路径（多跳，高延迟）：

跨节点KVCache传输：
远程NPU HBM → 本地Host DRAM → 本地NPU HBM
     ↓              ↓              ↓
   RDMA Read     PCIe DMA       PCIe DMA
   (~5-10μs)     (~2μs)         (~2μs)
                               总延迟: ~9-14μs

跨节点（超节点外）：
远程NPU HBM → RDMA网络 → 本地Host DRAM → 本地NPU HBM
                               总延迟: ~20-50μs
```

**痛点表现**：

- 企业级多轮对话中，每轮对话需重新加载历史KVCache，加载延迟直接影响TTFT
- 典型场景：32K上下文，KVCache约2-4GB，传输需0.5-2ms，占TTFT比例达20-40%
- Coding Agent场景中，代码上下文频繁切换，KVCache迁移路径开销累积严重

#### 痛点2：多级缓存层次不足，缓存命中率低

**现状（AS IS）**：

```
当前缓存层次（仅DRAM+SSD二级）：

L0: NPU HBM（推理实例私有，容量受限，32-64GB/NPU）
     ↓ Miss
L1: DRAM缓存池（Mooncake Store管理，~100GB-1TB/集群）
     ↓ Miss
L2: SSD持久化（DFS，容量大但延迟高，100-500μs）

问题：
- 缺少超节点级别的共享缓存层
- NPU HBM与DRAM之间无灵衢加速的快速通道
- 无GVA统一编址，跨节点访问需多次地址转换
```

**痛点表现**：

- 多轮对话场景中，跨节点缓存命中率仅60-70%，大量请求回退到SSD或重新计算
- Coding Agent长时间任务中，KVCache容量超限后被淘汰，再次访问需重新Prefill
- 超节点内NPU间无共享缓存池，跨卡/跨节点KVCache共享效率低

#### 痛点3：传输路径静态，无法自适应硬件拓扑

**现状（AS IS）**：

```
当前路径选择逻辑（mooncake-transfer-engine/src/multi_transport.cpp）：

selectTransport(entry) {
    auto proto = target_segment_desc->protocol;
    transport = transport_map_[proto];  // 静态协议选择
    return transport;
}

局限：
- 传输协议在Segment注册时确定，运行时无法动态切换
- 不识别灵衢HCCP/HCOM双协议栈
- 无法根据数据大小、延迟敏感度自适应选择路径
- 无Dual-Path并发传输能力
```

**痛点表现**：

- A3超节点内，NPU间通信仍走RDMA而非灵衢HCCP，延迟损失5-10倍
- 大块KVCache传输无法同时利用灵衢+RDMA双路径聚合带宽
- 网络拥塞或故障时无自动降级/切换机制

#### 痛点4：元数据管理集中化，超节点规模下成为瓶颈

**现状（AS IS）**：

```
当前元数据管理：
- Master Service集中管理所有对象元数据
- 1024分片减少锁竞争
- etcd持久化元数据

瓶颈：
- 单Master处理所有PutStart/PutEnd/GetReplicaList请求
- 超节点规模（64+ NPU）下，元数据操作QPS成为瓶颈
- 无GVA感知，无法利用灵衢统一地址空间优化
```

**痛点表现**：

- 高并发推理请求下，Master RPC延迟抖动，影响SLO
- Coding Agent场景中，大量小粒度KVCache对象的元数据管理开销大
- 缓存淘汰决策缺乏全局视角，淘汰热点数据导致性能回退

---

## 三、TO BE：目标架构与技术方案

### 3.1 目标架构概览

```
┌──────────────────────────────────────────────────────────────────────┐
│            Mooncake x 灵衢 优化后架构 (TO BE)                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                      应用层（推理框架）                          │ │
│  │  vLLM (KV Connector) │ SGLang (HiCache) │ 自研推理引擎           │ │
│  └────────────────────────────┬────────────────────────────────────┘ │
│                               │                                      │
│  ┌────────────────────────────▼────────────────────────────────────┐ │
│  │              Mooncake Store API (扩展)                           │ │
│  │  Put / Get / Remove / Query + GVA-aware 接口                    │ │
│  └────────────────────────────┬────────────────────────────────────┘ │
│                               │                                      │
│  ┌────────────────────────────▼────────────────────────────────────┐ │
│  │           Tiered-Cache 数据平面（L0-L4 五级）                    │ │
│  │                                                                  │ │
│  │  L0: 本地NPU HBM    (32-64GB, <100ns)    ← 推理实例私有        │ │
│  │  L1: 灵衢HBM池      (256GB-1TB, <1μs)   ← 超节点内共享 (NEW)  │ │
│  │  L2: 本地Host DRAM  (512GB-2TB, 1-2μs)  ← 节点内共享          │ │
│  │  L3: 分布式KVCache  (PB级, 5-50μs)      ← 集群共享 (RDMA)     │ │
│  │  L4: SSD持久化      (PB级, 100-500μs)   ← 长期存储            │ │
│  └────────────────────────────┬────────────────────────────────────┘ │
│                               │                                      │
│  ┌────────────────────────────▼────────────────────────────────────┐ │
│  │           Dual-Path 传输引擎（智能路径选择）                     │ │
│  │                                                                  │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │ │
│  │  │灵衢 HCCP     │  │灵衢 HCOM     │  │传统 RDMA     │          │ │
│  │  │(设备直连)    │  │(主机RoCE)    │  │(跨超节点)    │          │ │
│  │  │<1μs/>100GB/s │  │~2μs/~80GB/s  │  │~10μs/~50GB/s │          │ │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │ │
│  │         └─────────┬───────┘                 │                   │ │
│  │                   │         ┌───────────────┘                   │ │
│  │         ┌─────────▼─────────▼───────────┐                       │ │
│  │         │  智能路径选择器 (Path Selector) │                       │ │
│  │         │  延迟/带宽/可用性自适应决策     │                       │ │
│  │         └────────────────────────────────┘                       │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                               │                                      │
│  ┌────────────────────────────▼────────────────────────────────────┐ │
│  │              GVA 统一编址管理层                                   │ │
│  │  256TB全局地址空间 │ 跨节点透明访问 │ 2MB大页对齐                │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                               │                                      │
│  ┌────────────────────────────▼────────────────────────────────────┐ │
│  │              灵衢硬件层                                          │ │
│  │  A2: PCIe + RDMA    A3/A5: 灵衢UB + HCCP/HCOM + RDMA          │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 核心技术创新点

#### 创新点1：五级Tiered-Cache架构

**设计目标**：将Mooncake Store现有DRAM+SSD二级存储扩展为L0-L4五级缓存层次，新增灵衢HBM池化共享层。

```
┌──────────────────────────────────────────────────────────────────┐
│                    五级Tiered-Cache架构                           │
├──────────┬──────────────┬──────────────┬─────────────────────────┤
│ 层级     │ 容量(典型)   │ 访问延迟     │ 技术                    │
├──────────┼──────────────┼──────────────┼─────────────────────────┤
│ L0: 本地 │ 32-64GB/NPU  │ <100ns       │ NPU HBM本地直接访问     │
│   HBM    │              │              │                         │
├──────────┼──────────────┼──────────────┼─────────────────────────┤
│ L1: 灵衢 │ 256GB-1TB    │ <1μs         │ 灵衢HCCP设备直连       │
│   HBM池  │ (多NPU聚合)  │              │ GVA统一编址 (NEW)       │
├──────────┼──────────────┼──────────────┼─────────────────────────┤
│ L2: 本地 │ 512GB-2TB    │ 1-2μs        │ PCIe DMA               │
│   DRAM   │ /节点        │              │ Host-Device传输         │
├──────────┼──────────────┼──────────────┼─────────────────────────┤
│ L3: 分布 │ PB级         │ 5-50μs       │ RoCE/RDMA               │
│   式缓存 │ (集群聚合)   │              │ Mooncake Store          │
├──────────┼──────────────┼──────────────┼─────────────────────────┤
│ L4: SSD │ PB级         │ 100-500μs    │ NVMe-oF / DFS           │
│   持久化 │              │              │ 异步持久化              │
└──────────┴──────────────┴──────────────┴─────────────────────────┘
```

**关键实现**：

1. **CacheTier抽象接口扩展**（参考上游PR #1212）：
```cpp
class CacheTier {
public:
    virtual bool Init(TieredBackend* backend, TransferEngine* engine) = 0;
    virtual std::optional<TieredLocation> Allocate(size_t size) = 0;
    virtual bool Free(uint64_t offset, size_t size) = 0;
    virtual bool WriteAt(uint64_t offset, const DataSource& source) = 0;
    virtual std::optional<DataSource> AsDataSource(const std::string& key) = 0;
    virtual void BindKey(const std::string& key, uint64_t offset, size_t size) = 0;
    virtual void Delete(const std::string& key) = 0;
};
```

2. **灵衢CacheTier实现**（L1层核心）：
   - 通过MemFabric Hybrid BM API管理GVA内存
   - RH2D/GD2D双向数据传输优化
   - 2MB大页对齐内存分配与注册

3. **DataCopier数据拷贝矩阵**：支持L0-L4任意层间的数据迁移
```
           → L0(HBM)  → L1(灵衢)  → L2(DRAM)  → L3(分布式)  → L4(SSD)
L0(HBM)    │ 本地     │ HCCP      │ PCIe DMA  │ RDMA       │ 异步写
L1(灵衢)   │ HCCP     │ GVA直连   │ PCIe DMA  │ RDMA       │ 异步写
L2(DRAM)   │ PCIe DMA │ PCIe DMA  │ 本地      │ RDMA       │ 异步写
L3(分布式) │ RDMA     │ RDMA      │ RDMA      │ 本地       │ 异步写
L4(SSD)    │ 异步读   │ 异步读    │ 异步读    │ 异步读     │ 本地
```

**参考来源**：上游Mooncake Store PR #1212 Tiered Backend设计 + P2P分支的分布式对象共享模式。

#### 创新点2：Dual-Path智能传输路径优化

**设计目标**：基于灵衢HCCP（设备直连）和HCOM（主机RoCE）双协议栈，实现运行时自适应路径选择和并发传输。

```
┌──────────────────────────────────────────────────────────────────┐
│              Dual-Path 智能传输架构                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  传输请求 ──► 智能路径选择器 ──► 并行传输执行                    │
│               │                  │                               │
│               ├─ 拓扑感知        ├─ Path A: 灵衢HCCP            │
│               ├─ 数据大小感知    ├─ Path B: 灵衢HCOM/RDMA       │
│               ├─ 延迟/带宽感知   └─ 结果聚合                    │
│               └─ 可用性感知                                     │
│                                                                  │
│  决策矩阵：                                                      │
│  ┌──────────────┬────────────────┬─────────────────────┐        │
│  │ 场景         │ 首选路径       │ 备选路径             │        │
│  ├──────────────┼────────────────┼─────────────────────┤        │
│  │ 超节点内     │ 灵衢HCCP      │ 灵衢HCOM             │        │
│  │ 设备→设备    │ (亚微秒延迟)  │ (RoCE回退)           │        │
│  ├──────────────┼────────────────┼─────────────────────┤        │
│  │ 超节点内     │ 灵衢HCOM      │ 传统RDMA             │        │
│  │ 主机→设备    │ (RDMA+DMA)    │                      │        │
│  ├──────────────┼────────────────┼─────────────────────┤        │
│  │ 跨超节点     │ 传统RDMA      │ TCP                   │        │
│  │              │ (多NIC聚合)   │ (兼容性回退)          │        │
│  ├──────────────┼────────────────┼─────────────────────┤        │
│  │ 大块传输     │ Dual-Path并发 │ 单路径                │        │
│  │ (>64MB)      │ (HCCP+RDMA)   │ (带宽聚合)           │        │
│  └──────────────┴────────────────┴─────────────────────┘        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**关键实现**：

1. **智能路径选择器**：
```cpp
enum TransportPath {
    PATH_LOCAL,           // 本地内存直接访问
    PATH_LINGQU_HCCP,     // 灵衢HCCP (设备直连, <1μs)
    PATH_LINGQU_HCOM,     // 灵衢HCOM (主机RoCE, ~2μs)
    PATH_RDMA,            // 传统RDMA (跨超节点, ~10μs)
    PATH_TCP,             // TCP回退
};

TransportPath SelectOptimalPath(const SelectionContext& ctx) {
    if (IsInSameSuperPod(ctx.local_rank, ctx.remote_rank)) {
        if (IsDeviceDirectAvailable(ctx)) return PATH_LINGQU_HCCP;
        if (HCOMAvailable(ctx))           return PATH_LINGQU_HCOM;
    }
    if (RDMAAvailable(ctx))               return PATH_RDMA;
    return PATH_TCP;
}
```

2. **Dual-Path并发传输**：大块KVCache传输时，同时利用灵衢HCCP和RDMA两条路径，带宽叠加：
```
数据块分片策略：
[Block 0] [Block 1] [Block 2] [Block 3] [Block 4] ...
    │         │         │         │         │
    ├─Path A──┤         ├─Path A──┤         ├─Path A──
    │         ├─Path B──┤         ├─Path B──┤
    │         │         │         │         │
    └─────────┴─────────┴─────────┴─────────┘
                  带宽聚合: Path A + Path B
```

3. **自适应指标采集**：基于EMA（指数移动平均）实时采集各路径延迟、带宽、错误率，动态调整路径权重。

#### 创新点3：GVA统一编址与跨节点透明访问

**设计目标**：利用灵衢GVA（Global Virtual Address）256TB统一地址空间，实现跨节点内存的透明访问，消除多次地址转换开销。

```
┌──────────────────────────────────────────────────────────────────┐
│                    GVA统一编址空间 (256TB)                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  0x100000000000 ──┐                                              │
│                   ├─ HBM空间 (8TB)   ← L0+L1层映射              │
│  0x180000000000 ──┘                                              │
│                                                                  │
│  0x280000000000 ──┐                                              │
│                   ├─ GVM空间 (128TB) ← L2+L3层映射              │
│  0xA80000000000 ──┘                                              │
│                                                                  │
│  0x30000000000 ───┐                                              │
│                   ├─ Host空间 (16TB) ← 控制通道                 │
│  0x130000000000 ──┘                                              │
│                                                                  │
│  核心能力：                                                      │
│  • 远程内存导入: ImportMemory(remote_rank, remote_lva, size)     │
│  • GVA→LVA转换: 本地/远程自动路由                               │
│  • 设备直接访问: NPU通过GVA直接读取远程HBM/DRAM                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**关键优化**：

- **消除中转跳步**：GVA编址下，远程HBM可直接映射到本地地址空间，无需Host DRAM中转
- **减少地址转换**：一次GVA查找替代多次段描述符查找+Rank映射
- **大页对齐**：2MB大页减少TLB Miss，提升连续内存访问效率

#### 创新点4：场景感知的缓存策略

**设计目标**：针对多轮对话和Coding Agent的差异化访问模式，设计专用缓存策略。

```
┌──────────────────────────────────────────────────────────────────┐
│                    场景感知缓存策略                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  多轮对话模式：                                                  │
│  ┌──────────────────────────────────────────────────────┐        │
│  │  Round 1: [Prompt] → Prefill → KVCache写入L0+L1     │        │
│  │  Round 2: 加载Round1 KVCache → 增量Prefill → 写入    │        │
│  │  Round 3: 加载Round1+2 KVCache → 增量Prefill → 写入  │        │
│  │  ...                                                  │        │
│  │  策略：                                               │        │
│  │  • 历史KVCache Soft Pin保存在L1(灵衢HBM池)           │        │
│  │  • 异步预取下一轮KVCache到L0                         │        │
│  │  • 对话结束时降级到L2/L3等待复用                     │        │
│  └──────────────────────────────────────────────────────┘        │
│                                                                  │
│  Coding Agent模式：                                              │
│  ┌──────────────────────────────────────────────────────┐        │
│  │  Task: 代码补全/审查                                  │        │
│  │  Context: 仓库代码 + API文档 + 历史操作              │        │
│  │  策略：                                               │        │
│  │  • System Prompt KVCache永久Pin在L1                  │        │
│  │  • 代码上下文按文件粒度管理，热文件KVCache在L0/L1    │        │
│  │  • 跨任务KVCache复用：相同仓库不同Agent共享           │        │
│  │  • 生命周期管理：基于Task结束信号触发降级/淘汰        │        │
│  └──────────────────────────────────────────────────────┘        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.3 AS IS vs TO BE 关键指标对比

```
┌────────────────────┬──────────────────────┬──────────────────────┬───────────┐
│ 指标维度           │ AS IS (当前)          │ TO BE (目标)          │ 改进幅度  │
├────────────────────┼──────────────────────┼──────────────────────┼───────────┤
│ 超节点内延迟       │ ~5-10μs (RDMA)       │ <1μs (灵衢HCCP)     │ 5-10倍    │
│ 超节点内带宽       │ ~40GB/s (RDMA)       │ >100GB/s (灵衢HCCP) │ 2.5倍     │
│ 跨超节点延迟       │ ~20-50μs             │ ~10-20μs (路径优化)  │ 2倍       │
│ KVCache缓存层数    │ 2层 (DRAM+SSD)       │ 5层 (L0-L4)         │ 新增3层   │
│ 超节点内缓存容量   │ ~100GB (DRAM)        │ ~1TB (HBM池化)      │ 10倍      │
│ 缓存命中率         │ ~60-70%              │ >90% (五级层次)      │ +20-30%   │
│ 路径选择           │ 静态协议             │ 运行时自适应         │ 质变      │
│ Dual-Path并发      │ 不支持               │ HCCP+RDMA聚合       │ 新增      │
│ GVA统一编址        │ 不支持               │ 256TB统一空间        │ 新增      │
│ TTFT (多轮对话)    │ 基准                 │ -30%                 │ 30%降低   │
│ KVCache复用率      │ ~40-50%              │ >80%                 │ 50%+提升  │
└────────────────────┴──────────────────────┴──────────────────────┴───────────┘
```

---

## 四、详细技术方案

### 4.1 系统架构设计

#### 4.1.1 整体分层架构

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: 推理框架集成层                                         │
│   vLLM KV Connector │ SGLang HiCache │ 自研推理引擎适配器      │
├─────────────────────────────────────────────────────────────────┤
│ Layer 2: Mooncake Store API层                                   │
│   标准API (Put/Get/Remove) + GVA扩展API                        │
│   + 场景感知策略接口 (SessionPin/ContextGroup)                  │
├─────────────────────────────────────────────────────────────────┤
│ Layer 3: Tiered-Cache 数据平面                                  │
│   CacheTier管理器 + DataCopier + 层间迁移引擎                  │
│   L0(HBM) ←→ L1(灵衢) ←→ L2(DRAM) ←→ L3(分布式) ←→ L4(SSD) │
├─────────────────────────────────────────────────────────────────┤
│ Layer 4: Dual-Path 传输引擎                                     │
│   智能路径选择器 + 灵衢Transport + RDMA Transport + TCP         │
│   Dual-Path并发管理 + 自适应指标采集                            │
├─────────────────────────────────────────────────────────────────┤
│ Layer 5: GVA编址与内存管理                                      │
│   GVA管理器 (MemFabric Hybrid BM) + 内存对齐 + 大页管理        │
├─────────────────────────────────────────────────────────────────┤
│ Layer 6: 硬件抽象层                                             │
│   A2: Ascend Direct Transport (HCCS/RDMA)                      │
│   A3/A5: 灵衢Transport (HCCP/HCOM) + Ascend Direct Transport  │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.1.2 数据流优化对比

**AS IS 数据流（当前）**：
```
KVCache传输 (Prefill → Decode):
Prefill NPU HBM
    → (PCIe DMA) Host DRAM
    → (RDMA Write) Decode Host DRAM
    → (PCIe DMA) Decode NPU HBM
总跳步: 3次拷贝, 延迟: ~15-30μs
```

**TO BE 数据流（优化后）**：
```
超节点内KVCache传输 (灵衢HCCP):
Prefill NPU HBM
    → (灵衢HCCP, GVA直连) Decode NPU HBM
总跳步: 1次, 延迟: <1μs

超节点内KVCache传输 (灵衢Fabric Memory):
Prefill NPU HBM
    → (RDMA + GVA映射) Decode NPU HBM
总跳步: 1次(零拷贝), 延迟: ~2-5μs

跨超节点KVCache传输:
Prefill NPU HBM
    → (Dual-Path: HCCP+RDMA) Decode NPU HBM
总跳步: 1-2次, 延迟: ~10-15μs
```

### 4.2 核心模块设计

#### 4.2.1 Tiered-Cache核心组件

基于上游PR #1212的Tiered Backend设计，扩展实现灵衢CacheTier：

| 组件 | 职责 | 实现要点 |
|------|------|---------|
| **CacheTier** | 缓存层抽象接口 | 统一Init/Allocate/Free/WriteAt/AsDataSource接口 |
| **TieredBackend** | 数据平面管理 | 层间分配、写入、提交、迁移 |
| **DataCopier** | 数据拷贝引擎 | 任意层间数据拷贝，支持灵衢/RDMA/DMA多种路径 |
| **LingQuCacheTier** | L1灵衢缓存层 | GVA管理、RH2D传输、2MB对齐内存管理 |
| **PrefetchStrategy** | 预取策略 | 基于访问模式预测的异步预取 |
| **EvictionPolicy** | 淘汰策略 | 场景感知的近LRU淘汰，支持Soft Pin |

#### 4.2.2 Dual-Path传输核心组件

| 组件 | 职责 | 实现要点 |
|------|------|---------|
| **IntelligentPathSelector** | 路径选择决策 | 拓扑感知、数据大小感知、延迟带宽自适应 |
| **LingQuTransport** | 灵衢传输适配 | HCCP/HCOM协议封装，GVA地址操作 |
| **DualPathExecutor** | 并发传输执行 | 大块数据分片、多路径并行、结果聚合 |
| **PathMetricsCollector** | 指标采集 | EMA平滑的延迟/带宽/错误率实时采集 |

#### 4.2.3 A2/A3/A5平台适配

| 平台 | 硬件特性 | 传输策略 |
|------|---------|---------|
| **A2** | PCIe互联 + RDMA网卡 | Ascend Direct Transport (HCCS) + RDMA |
| **A3** | 灵衢UB 1.0 + HCCP/HCOM | 灵衢HCCP优先 + HCOM/RDMA备选 + Dual-Path |
| **A5** | 灵衢UB 2.0(预期增强) | 灵衢HCCP + Fabric Memory + 增强Dual-Path |

### 4.3 关键接口设计

#### 4.3.1 TieredBackend扩展接口

```cpp
// 新增灵衢感知的分配接口
AllocationHandle Allocate(size_t size,
                          std::optional<uint64_t> preferred_tier,
                          std::optional<GVAHint> gva_hint);

// 场景感知缓存策略接口
struct SessionPinConfig {
    std::string session_id;          // 会话标识
    uint64_t ttl_seconds;            // 生存时间
    bool pin_to_lingqu;              // 是否Pin到灵衢层
    std::vector<std::string> keys;   // 关联的KVCache键
};
bool PinSession(const SessionPinConfig& config);
bool UnpinSession(const std::string& session_id);

struct ContextGroupConfig {
    std::string group_id;            // 上下文组标识 (如仓库路径)
    std::string system_prompt_key;   // System Prompt KVCache键
    bool shared_across_agents;       // 是否跨Agent共享
};
bool CreateContextGroup(const ContextGroupConfig& config);
```

#### 4.3.2 Dual-Path传输接口

```cpp
// Dual-Path传输请求
struct DualPathTransferRequest {
    void* local_addr;
    uint64_t remote_gva;
    size_t size;
    enum Direction { READ, WRITE } direction;
    bool enable_dual_path;           // 是否启用双路径
    size_t dual_path_threshold;      // 双路径触发阈值
};

// 执行Dual-Path传输
bool DualPathTransfer(const DualPathTransferRequest& request);

// 获取路径性能指标
struct PathMetricsReport {
    double lingqu_hccp_latency_us;
    double lingqu_hccp_bandwidth_gbps;
    double lingqu_hcom_latency_us;
    double rdma_latency_us;
    double rdma_bandwidth_gbps;
    uint64_t dual_path_activation_count;
};
PathMetricsReport GetPathMetricsReport();
```

---

## 五、先进性论证

### 5.1 技术先进性

#### 5.1.1 业界首次实现灵衢互联与Mooncake架构的深度融合

| 对比维度 | 本方案 | 业界现状 |
|---------|--------|---------|
| NPU互联传输 | 灵衢HCCP设备直连，亚微秒延迟 | GPU RDMA传输，微秒级延迟 |
| 统一编址 | GVA 256TB全局地址空间 | 各节点独立编址，需多次转换 |
| 缓存层次 | 五级层次（HBM/灵衢HBM池/DRAM/分布式/SSD） | 二至三级（HBM/DRAM/SSD） |
| 路径选择 | 运行时自适应双路径 | 静态单路径 |
| 设备直连 | NPU-to-NPU零拷贝（GVA映射） | 需Host中转 |

#### 5.1.2 对比同类技术方案

| 方案 | 缓存层次 | 跨节点延迟 | NPU原生支持 | 场景感知 |
|------|---------|-----------|------------|---------|
| **本方案** | L0-L4五级 | <1μs (超节点内) | 灵衢HCCP原生 | 多轮对话+Agent专用策略 |
| Mooncake原版 | DRAM+SSD二级 | ~5-10μs | Ascend Direct (HCCS) | 通用LRU |
| vLLM KV Connector | GPU HBM+Host DRAM | ~10-20μs | 不支持 | 通用策略 |
| SGLang HiCache | GPU/Host/Remote三级 | ~5-15μs | 不支持 | 通用策略 |
| LMCache | Host DRAM+Remote | ~10-20μs | 不支持 | 通用策略 |

#### 5.1.3 灵衢技术优势分析

| 灵衢特性 | 技术优势 | 对Mooncake的增益 |
|---------|---------|-----------------|
| 亚微秒延迟 | 比RDMA延迟低5-10倍 | KVCache加载延迟降低30%+ |
| >100GB/s带宽 | 比单RDMA链路带宽高2-3倍 | 大块KVCache传输时间减半 |
| GVA统一编址 | 256TB全局透明访问 | 消除地址转换开销，简化数据路径 |
| HBM池化 | 超节点内HBM聚合为共享池 | 缓存容量提升10倍，命中率提升20%+ |
| Fabric Memory | 设备直接访问远程Host内存 | 零拷贝跨节点传输 |

### 5.2 业务先进性

#### 5.2.1 多轮对话场景收益

```
场景假设：企业客服对话，32K上下文，8轮对话

AS IS（当前）：
每轮需加载历史KVCache (~2GB)：
  L0 Miss → L2 DRAM Hit → PCIe传输 → NPU HBM
  加载延迟: ~15μs × 2000 (分块) ≈ 30ms
  TTFT构成: Prefill(80ms) + KVCache加载(30ms) = 110ms
  KVCache加载占比: ~27%

TO BE（优化后）：
每轮加载历史KVCache：
  L0 Miss → L1 灵衢HBM池 Hit → HCCP传输 → NPU HBM
  加载延迟: <1μs × 500 (灵衢聚合) ≈ 0.5ms
  TTFT构成: Prefill(80ms) + KVCache加载(0.5ms) = 80.5ms
  KVCache加载占比: ~0.6%

收益：TTFT降低 ~27%，KVCache加载延迟降低 ~98%
```

#### 5.2.2 Coding Agent场景收益

```
场景假设：代码补全Agent，128K上下文，跨10个文件

AS IS（当前）：
Agent任务切换时KVCache迁移：
  跨节点RDMA传输 (~4GB KVCache)
  迁移延迟: ~80ms
  任务间KVCache复用率: ~40%

TO BE（优化后）：
Agent任务切换时KVCache迁移：
  L1灵衢池缓存命中 → 无需迁移
  或灵衢HCCP传输: ~40ms (Dual-Path)
  System Prompt永久Pin在L1: 零加载开销
  任务间KVCache复用率: >80%

收益：任务切换延迟降低 50%+，KVCache复用率提升 100%+
```

### 5.3 社区影响力论证

1. **上游社区贡献**：将Tiered-Cache架构优化、灵衢Transport适配、Dual-Path路径选择等核心代码贡献至Mooncake上游社区
2. **CODEOWNER获取**：通过持续高质量贡献，争取成为mooncake-store模块的CODEOWNER
3. **技术品牌建设**：发表技术博客、社区分享，建立团队在LLM推理加速领域的技术影响力
4. **生态拓展**：为昇腾NPU生态提供高性能KVCache解决方案，促进国产算力平台在LLM推理领域的应用

---

## 六、实施路线图

### Phase 1: 基础传输层适配（月1-月3）

**目标**：完成灵衢传输层基础集成和A2/A3平台适配

**关键里程碑**：
- 灵衢Transport（HCCP/HCOM）适配Mooncake Transfer Engine
- GVA管理器基础实现
- A2平台Ascend Direct Transport验证通过
- 单元测试覆盖率 >80%

**交付物**：
- `mooncake-transfer-engine/src/transport/lingqu_transport.h/cpp`
- `mooncake-store/include/gva_manager.h`
- A2平台性能基准测试报告

### Phase 2: Tiered-Cache架构实现（月3-月6）

**目标**：实现五级Tiered-Cache架构，完成LingQuCacheTier

**关键里程碑**：
- CacheTier/TieredBackend/DataCopier核心接口实现
- LingQuCacheTier (L1层) 完整实现
- 智能路径选择器基础版
- A3平台灵衢HCCP传输验证通过

**交付物**：
- `mooncake-store/src/tiered_cache/` 完整实现
- A3平台Tiered-Cache集成测试报告
- Mooncake Store PR提交至上游

### Phase 3: Dual-Path与性能优化（月6-月9）

**目标**：实现Dual-Path并发传输和场景感知缓存策略

**关键里程碑**：
- Dual-Path并发传输实现与调优
- 多轮对话场景Soft Pin + 异步预取策略
- Coding Agent场景上下文组管理
- 性能基准测试达标

**交付物**：
- Dual-Path传输模块
- 场景感知缓存策略模块
- 端到端性能测试报告
- vLLM/SGLang集成验证

### Phase 4: 生产就绪与社区贡献（月9-月12）

**目标**：生产环境部署验证，完成上游社区贡献

**关键里程碑**：
- 高可用部署方案验证
- 监控告警系统集成
- 上游社区PR合并
- 技术博客和分享

**交付物**：
- 生产部署指南
- 监控运维手册
- 上游社区贡献记录
- 技术影响力总结

---

## 七、资源与风险

### 7.1 资源需求

**人力**：
- C++开发工程师: 3-4人
- 灵衢技术专家: 1-2人（需华为技术支持）
- 测试工程师: 1-2人
- 架构师: 1人

**硬件**：
- 昇腾A2集群: 至少2节点（基础验证）
- 昇腾A3集群: 至少4节点 + 灵衢交换机（核心开发与测试）
- 测试环境: 独立测试集群

### 7.2 风险评估

| 风险项 | 可能性 | 影响 | 缓解措施 |
|--------|--------|------|---------|
| 灵衢驱动/CANN版本兼容性 | 中 | 高 | 早期环境搭建验证，与华为建立技术支持通道 |
| MemFabric Hybrid接口稳定性 | 中 | 高 | 充分的单元测试，准备降级方案 |
| Tiered-Cache架构与现有Store冲突 | 低 | 中 | 模块化设计，保持向后兼容 |
| 上游社区PR合并周期 | 中 | 中 | 提前与社区维护者沟通，渐进式提交 |
| 性能目标未达预期 | 低 | 高 | 分阶段性能验证，持续优化关键路径 |
| A5平台延期交付 | 中 | 低 | A3优先实现，A5作为增强目标 |

### 7.3 关键依赖

| 依赖项 | 版本要求 | 状态 |
|--------|---------|------|
| 昇腾CANN | >=8.0 | 需获取 |
| 灵衢驱动 | 最新版 | 需华为提供 |
| MemFabric Hybrid | >=1.0 | 开源可用 |
| Mooncake上游 | main分支 | 持续跟进 |
| etcd | >=3.5 | 已支持 |

---

## 八、总结

本项目基于昇腾NPU A2/A3/A5平台及灵衢高速互联总线技术，对Mooncake KVCache传输及多级管理架构进行深度优化，核心创新包括：

1. **五级Tiered-Cache架构**：首次将灵衢GVA统一编址与Mooncake Store多级缓存深度融合，实现从NPU HBM到SSD的完整缓存层次
2. **Dual-Path智能传输**：基于灵衢HCCP/HCOM双协议栈实现运行时自适应路径选择和并发传输
3. **场景感知缓存策略**：针对多轮对话和Coding Agent的差异化访问模式设计专用策略

预期收益：
- 超节点内KVCache传输延迟降低5-10倍（~10μs → <1μs）
- 缓存命中率提升20-30%（60-70% → >90%）
- 多轮对话TTFT降低30%+，Coding Agent KVCache复用率提升50%+
- 成为Mooncake上游社区NPU平台的核心贡献者

---

*附录：参考文档*
- A3灵衢架构技术白皮书 (`cc-analyzer/A3_LingQu_架构技术白皮书.md`)
- Mooncake TieredCache灵衢架构演进方案 (`cc-analyzer/Mooncake_TieredCache_LingQu_Architecture_Evolution.md`)
- Mooncake Store设计文档 (`docs/source/design/mooncake-store.md`)
- Heterogeneous Ascend Transport设计 (`docs/source/design/transfer-engine/heterogeneous_ascend.md`)
- Ascend Direct Transport设计 (`docs/source/design/transfer-engine/ascend_direct_transport.md`)
- Mooncake上游PR #1212: Tiered Backend实现

---
title: 分布式 KVCache 技术趋势与 openFuyao 战略规划
subtitle: 面向领导与决策层的汇报材料
date: 2026-06-11
type: 决策层汇报
status: final
audience: 管理层 / 决策层
scope: 技术趋势洞察 + openFuyao KVCache 生态影响力路径
---

# 分布式 KVCache 技术趋势与 openFuyao 战略规划

> **汇报目的**：为领导层提供分布式 KVCache 领域的技术趋势全景判断，明确 openFuyao 的差异化定位和生态影响力构建路径。

---

## 执行摘要（核心结论，一页纸，3 分钟）

### 一句话定位

```
openFuyao/InferNex = 超节点硬件使能层 + 异构编排调度层 + 云原生治理层 + KVCache 存储优化贡献者
```

**核心论点**：openFuyao 不应成为"另一个 Mooncake"，而应通过 **超节点 + 灵衢 UB 总线硬件底座** + **Mooncake 上游席位获取**，成为"异构推理的云原生编排层"。

### 三个关键数据支撑

| 数据 | 来源 | 对 openFuyao 的意义 |
|------|------|---------------------|
| Mooncake + vLLM：吞吐 **3.8x**，TTFT 降低 **46x** | [vLLM Blog, 2026-05-06] | 验证 KVCache 方案有效性，openFuyao 集成 Mooncake 已有收益 |
| CloudMatrix384 UB：KVCache 90% 重用，TTFT 降低 **59%** | [arXiv 论文] | 华为超节点 + UB 硬件优势，openFuyao 核心差异化底座 |
| InferNex KVCache 路由：E2EL 降低 **22%** | [openFuyao v26.03] | openFuyao 已有能力验证，可继续增强 |

### 三个核心判断

1. **底层存储引擎趋于收敛 → Mooncake Store 成为主流**（PyTorch 生态，HiCache/LMCache 均以其为后端）
2. **上层管理层竞争本质是推理引擎竞争 → HiCache vs LMCache 分别绑定 SGLang/vLLM**
3. **异构硬件是中国市场独特变量 → openFuyao 需填补 Ascend 深度优化 + GPU 互通空白**

### 下一步行动建议（摘要）

| 时间 | 行动 | 目标 |
|------|------|------|
| **本周** | 发起 Layout Handler RFC | 进入 Mooncake 社区讨论 |
| **Q3** | 热点缓存 PR 合并 + 灵衢 GVA PoC | Store Top 5 + UB 验证 |
| **Q4** | 申请 Reviewer 席位 | CODEOWNERS 申请资格 |
| **待协调** | 与 Yuanrong/MemCache 明确分工 | 避免内部竞争 |

---

## 一、行业痛点与 KVCache 价值（精选 3 个代表场景）

### 1.1 为什么 KVCache 是 LLM 推理的核心瓶颈？

LLM 推理中，KVCache（键值缓存）占用 GPU/NPU 高带宽内存的 **60-80%**（来源：[Mooncake 论文, arXiv 2407.00079](https://arxiv.org/abs/2407.00079)）。随着上下文长度增长，KVCache 成为核心瓶颈，直接影响吞吐、延迟和部署成本。

```mermaid
graph LR
    subgraph 痛点["三大核心痛点"]
        P1["内存瓶颈<br/>KVCache 占 HBM 60-80%<br/>限制批处理大小"]
        P2["长上下文<br/>128K+ KVCache 数十 GB<br/>单卡无法承载"]
        P3["重复计算<br/>相同前缀重复计算<br/>浪费算力"]
    end

    subgraph KVCache方案["分布式 KVCache 解决方案"]
        S1["多级存储卸载<br/>HBM to DRAM to SSD"]
        S2["高速零拷贝传输<br/>RDMA / UB GVA"]
        S3["智能缓存管理<br/>命中感知调度 + 前缀复用"]
    end

    P1 --> S1
    P2 --> S1
    P3 --> S3
```

### 1.2 三个代表行业场景

#### 场景一：金融行业 — 智能客服 Agent 多轮对话

| 维度 | 具体情况 |
|------|---------|
| **业务场景** | 银行智能客服 Agent 处理多轮对话，历史上下文累积超过 64K tokens |
| **核心痛点** | 上下文累积超 64K 时，TTFT >10s，严重影响用户体验 |
| **KVCache 价值** | 缓存历史对话 KVCache，新请求仅计算增量部分 |
| **量化收益** | 热点缓存优化：**TTFT 降低 55-93%**，跨节点延迟从 881ms 降至 287ms（来源：[SIG 性能报告 v25.12]） |

#### 场景二：运营商 — 大规模异构集群推理

| 维度 | 具体情况 |
|------|---------|
| **业务场景** | 运营商 10,000+ 节点集群，昇腾 910B/910C 混合部署普遍 |
| **核心痛点** | 异构算力利用率 <50%，调度器无法感知 KVCache 分布 |
| **KVCache 价值** | KVCache 命中感知路由，将请求路由到缓存命中节点 |
| **量化收益** | Hermes-router KVCache 感知路由：**TPS 提升 16-30%**，PD 感知路由 **E2EL 降低 22.08%**（来源：[openFuyao v26.03]） |

#### 场景三：互联网 — RAG 知识库与企业 Agent

| 维度 | 具体情况 |
|------|---------|
| **业务场景** | 企业知识库 RAG 检索增强生成，大量用户查询共享相同文档前缀 |
| **核心痛点** | 相同文档前缀被不同用户请求重复计算 KVCache，浪费大量算力 |
| **KVCache 价值** | CacheBlend 跨请求智能混合，共享前缀 KVCache 直接复用 |
| **量化收益** | LMCache CacheBlend：**RAG 场景接近 100% 命中率**（EuroSys 2025 Best Paper）；蚂蚁集团 DeepSeek-R1：**TTFT 降低 84%**（来源：[SGLang HiCache Blog]） |

### 1.3 主流组件全景图

```mermaid
graph TB
    subgraph 上层["上层编排与治理"]
        OF["openFuyao / InferNex<br/>云原生推理基础设施"]
    end

    subgraph 中层["中间管理层"]
        HC["HiCache + SGLang<br/>分层 KV 缓存"]
        LMC["LMCache<br/>知识交付网络"]
    end

    subgraph 底层["底层传输与存储引擎"]
        MK["Mooncake Store + TE<br/>跨硬件统一存储引擎<br/>PyTorch 生态"]
        YR["Yuanrong<br/>Ascend Serverless 缓存"]
    end

    subgraph 硬件层["异构硬件"]
        NVIDIA["NVIDIA GPU"]
        ASCEND["昇腾 NPU"]
        KUNPENG["鲲鹏 CPU"]
        UB_BUS["灵衢 UB 总线"]
    end

    OF --> MK
    HC --> MK
    LMC --> MK
    MK --> NVIDIA
    MK --> ASCEND
    YR --> ASCEND
    YR --> UB_BUS
    OF --> UB_BUS
    OF --> KUNPENG

    classDef bottom fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef top fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef hw fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    class MK,YR bottom
    class OF top
    class NVIDIA,ASCEND,KUNPENG,UB_BUS hw
```

---

## 二、技术演进三大趋势

### 趋势总览与交叉关系

```mermaid
graph TB
    T1["趋势 1：存储与集成协同<br/>多级存储 + 插件式后端"]
    T2["趋势 2：注意力机制多样化<br/>MHA → GQA/MLA/DSA"]
    T3["趋势 3：异构硬件生态<br/>NVIDIA → Ascend/UB"]

    T1 -->|"驱动"| T2
    T2 -->|"适配"| T3
    T3 -->|"催生"| T1
```

### 趋势 1：存储与集成协同 — 多级存储 + 插件式后端

**演进路径**：

```mermaid
graph LR
    V1["V1: 单层 HBM<br/>直传 RDMA"] --> V2["V2: 两层<br/>HBM + DRAM"]
    V2 --> V3["V3: 三层<br/>HBM + DRAM + SSD<br/>Mooncake Store"]
    V3 --> V4["V4: 四层<br/>+ NVMe GDS<br/>LMCache"]
```

**关键创新点**：

| 系统 | 创新点 | 数据 |
|------|--------|------|
| HiCache | GPU 辅助 I/O 内核 | **3x cudaMemcpy 吞吐**（来源：[SGLang HiCache Blog]） |
| LMCache | NVMe GDS 直通 + NUMA 感知 | 四层存储最深 |
| HiCache | 3 函数后端接口（get/exist/set） | Mooncake/3FS/NIXL 已接入（来源：[HiCache Design]） |

### 趋势 2：注意力机制多样化

```mermaid
graph LR
    MHA["MHA<br/>传统"] --> GQA["GQA<br/>GLM-4/Qwen"]
    GQA --> MLA["MLA<br/>DeepSeek V3<br/>4-8x 压缩"]
    MLA --> DSA["DSA 稀疏<br/>仅保留活跃 KV<br/>5x 吞吐提升"]
```

| 机制 | 存储影响 | 代表系统 |
|------|---------|---------|
| MLA | 压缩潜在向量，**存储缩减 4-8x** | Mooncake Store MLA Handler（来源：[mooncake-store/include/mla_layout_handler.h]） |
| DSA 稀疏 | 仅存储活跃 KV 子集 | **HiSparse 5x 吞吐**（来源：[SGLang HiSparse Blog]） |

**趋势判断**：Mooncake Store 已有 MHA/GQA/MLA/Hybrid 四种布局处理器——领先优势；稀疏注意力是下一个竞争焦点。

### 趋势 3：异构硬件生态 — NVIDIA vs 华为 UB

```mermaid
graph TB
    subgraph NVIDIA生态["NVIDIA 生态"]
        NV["NVLink 节点内限 576 GPU"]
        RDMA["RDMA 跨节点<br/>延迟 5-50 us"]
    end

    subgraph 华为生态["华为生态"]
        UB["灵衢 UB 总线<br/>节点间延迟 lt 2 us"]
        CM["CloudMatrix384<br/>384 NPU 全互联"]
    end

    subgraph 对比["关键差异"]
        D1["NVLink: 节点内有限<br/>跨节点依赖 RDMA"]
        D2["UB: 节点间扁平化<br/>带宽损失 lt 3%<br/>延迟仅增 lt 1 us"]
    end

    NV --> D1
    UB --> D2
```

**实测数据（CloudMatrix384，来源：[arXiv 论文]）**：

| 指标 | NVIDIA RDMA | 华为 UB | 优势 |
|------|-------------|---------|------|
| NPU-NPU 读取带宽 | ~50 GB/s | **164 GB/s** | 3.3x |
| NPU-NPU 读取延迟 | ~5-10 μs | **1.9 μs** | 3-5x |
| 节点内/间带宽比 | 显著下降 | **98%（损失 <3%）** | 数量级优势 |
| KVCache 90% 重用 TTFT | — | **降低 59%** | 实测验证 |

---

## 三、生态格局与竞争态势

### 3.1 六大系统竞合关系

```mermaid
graph TB
    MK["Mooncake<br/>PyTorch 生态"]
    HC["HiCache<br/>SGLang 绑定"]
    LMC["LMCache<br/>vLLM 绑定"]
    YR["Yuanrong<br/>Ascend UB 原生"]
    OF["openFuyao<br/>云原生编排"]

    MK <-.->|战略合作<br/>LMCache 桥接| LMC
    MK <-.->|后端合作<br/>Mooncake 是 HiCache 后端| HC

    HC <-->|竞争<br/>分别绑定 SGLang/vLLM| LMC
    MK <-->|竞争<br/>跨硬件 vs Ascend 专用| YR

    OF ==>|上游贡献 + 下游集成| MK

    classDef coop fill:#c8e6c9,stroke:#2e7d32
    classDef comp fill:#ffcdd2,stroke:#c62828
    classDef supply fill:#bbdefb,stroke:#1565c0
    class MK,LMC,HC coop
    class YR comp
    class OF supply
```

### 3.2 三大关键判断（高亮）

> **判断 1：底层存储引擎趋于收敛 → Mooncake Store 成为主流**
>
> Mooncake 2026.02 进入 PyTorch 生态，HiCache/LMCache 均以其为远程后端。vLLM 官方集成（2026.05）标志着主流推理引擎认可。（来源：[Mooncake GitHub]）

> **判断 2：上层管理层竞争本质是推理引擎竞争 → HiCache vs LMCache 分别绑定 SGLang/vLLM**
>
> HiCache 深度绑定 SGLang RadixAttention，LMCache 深度绑定 vLLM KV Connector。竞争格局跟随推理引擎市场份额演变。（来源：本文分析）

> **判断 3：异构硬件是中国市场独特变量 → openFuyao 需填补 Ascend 深度优化 + GPU 互通空白**
>
> Yuanrong 仅 Ascend、缺乏跨硬件互通；Mooncake 覆盖广但 Ascend 深度不足。**openFuyao 应做"深度 Ascend 优化 + Ascend↔NVIDIA 互通"**。（来源：本文分析）

### 3.3 收敛趋势图示

```mermaid
graph LR
    subgraph 底层["底层收敛"]
        MK2["Mooncake Store<br/>事实标准"]
    end

    subgraph 上层["上层竞争"]
        HC2["HiCache<br/>SGLang"]
        LMC2["LMCache<br/>vLLM"]
    end

    subgraph 异构["异构变量"]
        YR2["Yuanrong<br/>Ascend 深度"]
        OF2["openFuyao<br/>编排 + UB 使能"]
    end

    HC2 --> MK2
    LMC2 --> MK2
    OF2 --> MK2
    YR2 -. Ascend 专有 .-> MK2
```

---

## 四、openFuyao 差异化定位：超节点 + UB 总线是核心底座

### 4.1 定位论证：为何不做"另一个 Mooncake"？

| 维度 | Mooncake | openFuyao 应该做 | 论据 |
|------|----------|-----------------|------|
| 技术栈层级 | 底层存储引擎 | **上层编排 + 硬件使能** | Yuanrong/MemCache 已做底层，openFuyao 做编排避免竞争 |
| 硬件策略 | 广度覆盖 4+ 平台 | **深度优化昇腾 + 灵衢 UB** | CloudMatrix384 实测优势，NVIDIA 无法复制 |
| 核心能力 | 传输 + 存储 | **智能路由 + 弹性伸缩 + 可观测** | InferNex 已验证 E2EL ↓22% |
| 上游策略 | — | **贡献 Mooncake → 获取 CODEOWNERS** | 席位 = 生态影响力 |

### 4.2 核心差异化底座：超节点 + UB 总线

> **华为超节点 + 灵衢 UB 总线是 openFuyao 区别于 Mooncake/HiCache/LMCache 的根本性差异**
>
> NVIDIA NVLink 节点内限 576 GPU，跨节点依赖 RDMA（4 跳，9-14 μs）；
> 华为 UB 总线通过 GVA 统一编址，将超节点扁平化为单一逻辑节点，节点间延迟 <2 μs，实现 1 跳零拷贝 KVCache 直访。

```mermaid
graph TB
    subgraph RDMA路径["NVIDIA RDMA 路径 4 跳"]
        N1["NPU HBM"] --> N2["Host DRAM"]
        N2 --> N3["RDMA"]
        N3 --> N4["对端 Host"]
        N4 --> N5["对端 NPU"]
        LAB1["延迟: 9-14 us<br/>带宽: 40-50 GB/s"]
    end

    subgraph UB路径["华为 UB 路径 1 跳"]
        U1["NPU HBM"] --> U2["UB GVA"]
        U2 --> U3["对端 NPU"]
        LAB2["延迟: lt 1 us<br/>带宽: gt 100 GB/s"]
    end
```

**两种超节点场景的 KVCache 发力空间**：

| 场景 | 硬件组合 | KVCache 发力点 | 关键指标 |
|------|---------|---------------|---------|
| **智算超节点** | Ascend 910C/950 + UB 全互联 | L0-L1 层 GVA 零拷贝直访 | 超节点内延迟 <1μs，带宽 >100 GB/s |
| **通算超节点** | Kunpeng 950 + Ascend NPU | CPU DRAM 冷存储 + NPU HBM 热缓存 | CPU-NPU 带宽 >100 GB/s，冷热切换 <2μs |

---

## 五、四大突破方向与 Mooncake 上游深耕路径

### 5.1 四大突破方向总览

```mermaid
graph TB
    subgraph P0方向["P0 优先级（0-6个月）"]
        D1["方向 1<br/>NPU 原生 KVCache 优化<br/>差异化护城河"]
        D2["方向 2<br/>异构集群 KVCache 互通<br/>生态桥梁"]
    end

    subgraph P1方向["P1 优先级（持续进行）"]
        D3["方向 3<br/>云原生 KVCache 治理<br/>管理层突破"]
        D4["方向 4<br/>上游贡献战略<br/>生态共建"]
    end

    D1 --> D2
    D1 --> D4
    D2 --> D4
    D3 --> D4
```

**逻辑依赖**：方向 1 和方向 2 是 P0 并行推进（都需要深入理解 Ascend KVCache 布局）；方向 3 在方向 1/2 基础能力后推进；方向 4 持续进行，将 1/2/3 的产出转化为上游贡献。

---

### 5.2 方向 1：NPU 原生 KVCache 优化（差异化护城河）—— P0

**核心价值**：

> 成为 Ascend NPU 生态的 KVCache 标准实现，在 NPU 上实现与 GPU 上 Mooncake Store 对标的传输性能。这是 openFuyao 区别于 Mooncake/HiCache/LMCache 的 **硬件级差异化护城河**。

**两个子场景的突破路径**：

| 场景 | 硬件架构 | 技术路径 | 性能目标 |
|------|---------|---------|---------|
| **A. 智算超节点** | Ascend 910C/950 + UB 全互联 | L0-L1 GVA 零拷贝，构建 LingQuCacheTier | 超节点内延迟 <1μs，带宽 >100 GB/s |
| **B. 通算超节点** | Kunpeng 950 + Ascend NPU | CPU DRAM 冷存储 + NPU HBM 热缓存 | Kunpeng-NPU 110-151 GB/s |

**关键对比：UB vs RDMA**

| 传输路径 | 跳数 | 延迟 | 带宽 |
|---------|------|------|------|
| RDMA（NPU→Host→RDMA→Host→NPU） | 4 跳 | 9-14 μs | 40-50 GB/s |
| **UB GVA 零拷贝**（NPU→UB→NPU） | **1 跳** | **<1 μs** | **>100 GB/s** |

**Mooncake 上游贡献点**：
- NPU 专用布局处理器 → Mooncake Store `KVCacheLayoutHandler` 框架
- Ascend ADXL Direct Transport 性能优化 → Mooncake TE
- 建立 Ascend NPU KVCache 性能基线 → 持续对标 GPU 版本

---

### 5.3 方向 2：异构集群 KVCache 互通（生态桥梁）—— P0

**核心价值**：

> 成为"Ascend Prefill + NVIDIA Decode"异构部署场景的 **唯一完整 KVCache 编排方案**。Yuanrong 仅支持 Ascend，无法实现跨硬件互通——这是 openFuyao 的 **差异化护城河**。

**技术突破路径**：

1. **格式分析**：深入分析 Ascend vs NVIDIA KVCache 内存布局差异（数据类型、内存对齐、GQA 组划分方式）
2. **转换层设计**：设计高效双向格式转换层，支持零拷贝策略
3. **异构传输优化**：在 Mooncake TE（HCCL + RDMA）基础上补充格式适配层
4. **路由策略扩展**：Hermes-router 增加"硬件类型感知"维度，选择转换开销最小路径

**预期成果**：异构集群 PD 分离性能损失控制在同构集群的 **10% 以内**

---

### 5.4 方向 3：云原生 KVCache 治理（管理层突破）—— P1

**核心价值**：

> 从"组件提供者"升级为"治理平台"，通过 K8s Operator 实现 KVCache 全生命周期自动化管理——这是上层编排定位的具体落地。

**技术突破路径**：

| 功能 | 技术实现 | 预期成果 |
|------|---------|---------|
| **生命周期管理** | K8s Operator + CRD（预热/淘汰/迁移/压缩） | 运维策略声明化 |
| **主动缓存调度** | Hermes-router + 流量预测模型 | 命中率提升 20%+ |
| **超节点拓扑感知** | 优先超节点内匹配（延迟 <2μs vs 跨超节点 5-50μs） | 超节点内命中率 >80% |
| **UB 监控扩展** | Eagle-eye UB 带宽/延迟/GVA 空间监控 | 形成运维闭环 |

---

### 5.5 方向 4：Mooncake 上游深耕路径（生态共建）—— P1

**核心定位**：成为 Mooncake 核心 Maintainer 之一，通过持续高质量贡献建立技术影响力，使 openFuyao 成为 Mooncake 生态中 **异构 NPU 方向的权威贡献者**。

#### 5.5.1 Mooncake 深耕领域

基于方向 1/2/3 的技术产出，确定 Mooncake 上游深耕的四个高价值领域：

| 深耕领域 | 具体贡献点 | PR 类型 | 对应突破方向 |
|---------|-----------|---------|-------------|
| **布局处理器框架** | NPU 专用 Handler（Ascend 内存布局适配） | Store PR | 方向 1 |
| **传输引擎优化** | ADXL Direct Transport 性能优化、UB GVA 后端 | TE PR | 方向 1 |
| **异构格式转换** | Ascend↔NVIDIA KVCache 格式转换模块 | TE PR | 方向 2 |
| **新注意力机制** | DSA 稀疏注意力 Handler（紧跟 DeepSeek/Qwen/GLM 发布） | Store PR | 方向 4 |

#### 5.5.2 三阶段节奏路标

```mermaid
graph LR
    subgraph 阶段一["阶段一 Q2-Q3：核心贡献者确立"]
        A1["RFC: Layout Handler"]
        A2["PR: NPU Handler 合并"]
        A3["PR: 热缓存优化 5+ 合并"]
        A4["Store Top 5 贡献者"]
    end

    subgraph 阶段二["阶段二 Q3-Q4：模块主导权申请"]
        B1["主导热点缓存<br/>架构演进讨论"]
        B2["PR: ADXL 性能优化"]
        B3["PR: UB GVA 后端"]
        B4["Reviewer 席位申请"]
    end

    subgraph 阶段三["阶段三 Q4-Q1: CODEOWNERS 申请"]
        C1["累计 20+ commits"]
        C2["3+ 高价值 PR<br/>代表作"]
        C3["Review 10+ 次"]
        C4["CODEOWNERS 权限"]
    end

    阶段一 --> 阶段二 --> 阶段三
```

#### 5.5.3 关键里程碑与验收标准

| 里程碑 | 时间 | 验收标准 | Mooncake 深耕内容 | 硬件结合点 |
|--------|------|---------|------------------|-----------|
| **M1** | 2026 Q3 | Store Top 5 + Layout Handler PR 合并 | NPU 布局处理器 PR | 昇腾 NPU 适配 |
| **M1.5** | 2026 Q3 | 灵衢 GVA 直访 PoC 验证 <1μs | UB GVA 后端 RFC 提交 | **灵衢 UB 验证** |
| **M2** | 2026 Q4 | Reviewer 席位 + InferNex 增强版 | 热缓存架构演进主导 + ADXL PR | 昇腾性能对标 GPU |
| **M2.5** | 2026 Q4 | Kunpeng+NPU 混合验证 | 通算超节点格式转换 PR | **鲲鹏 950 + 昇腾** |
| **M3** | 2027 Q1 | 异构互通 PoC | Ascend↔NVIDIA 转换模块 PR | 跨硬件格式转换 |
| **M3.5** | 2027 Q1 | 超节点 KVCache 能力验证 | DSA 稀疏注意力 Handler PR | 智算超节点验证 |
| **M4** | 2027 Q2 | 云原生治理平台发布 + CODEOWNERS | 代码审核参与 + 发布决策 | 全栈集成 |

#### 5.5.4 CODEOWNERS 申请触发条件

必须同时满足：

- [x] 累计 **20+ Store commits**
- [x] 获得 **1 位现有 CODEOWNER 公开认可**（@ykwd 或 @stmatengss）
- [x] 有 **3+ 高价值 PR** 作为代表作：
  1. NPU Layout Handler（方向 1）
  2. 热缓存架构优化（方向 1）
  3. 异构格式转换模块（方向 2）或 DSA Handler（方向 4）
- [x] 持续 Review 他人 PR **10+ 次**

---

### 5.6 核心风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Yuanrong Ascend 性能大幅领先 | Mooncake 社区降低 NPU 优先级 | 持续高质量贡献保持影响力，推动 NPU 成为核心路线图 |
| Kunpeng 950 上市延迟（目标 Q4） | 通算超节点验证受阻 | 先用 Kunpeng 920 验证 UB 传输可行性 |
| MemCache 与 openFuyao 定位冲突 | 内部重复投入 | 明确分工：MemCache 底层引擎，openFuyao 上层编排 + 上游贡献 |
| vLLM-Ascend DSA 接口延迟 | 稀疏注意力验证受阻 | 联合对齐排期，同步准备算法侧验证 |

---

### 5.7 下一步行动建议

**立即启动（本周）**：

| 行动 | 负责人 | Mooncake 深耕点 |
|------|--------|----------------|
| 发起 Layout Handler RFC Issue | 技术负责人 | Store 布局处理器框架讨论 |
| 完善 GQA/MLA/Hybrid Handler 代码 | 开发团队 | RFC 定稿后提交 PR |

**近期推进（Q3 2026）**：

| 行动 | 目标 | Mooncake 深耕点 |
|------|------|----------------|
| 热点缓存优化 PR 合并 | Store Top 5 贡献者 | 热缓存模块持续贡献 |
| 灵衢 GVA 直访 PoC 启动 | UB 零拷贝验证 | TE UB GVA 后端 RFC |
| ADXL Direct 性能优化 | Ascend 传输优化 | TE Ascend 优化 PR |

**中期推进（Q4 2026 - Q1 2027）**：

| 行动 | 目标 | Mooncake 深耕点 |
|------|------|----------------|
| 申请 Store Reviewer 席位 | CODEOWNERS 申请资格 | 参与代码审核 |
| Kunpeng+NPU 混合验证 | 通算超节点能力 | 格式转换模块 PR |
| 异构互通 PoC | Ascend↔NVIDIA | 异构转换层 PR |
| DSA Handler 实现 | 稀疏注意力适配 | 新 Handler PR |

**待协调事项**：

| 事项 | 协调对象 | 目的 |
|------|---------|------|
| 与 Yuanrong/MemCache 明确分工 | 产品线 | 避免内部竞争，明确上游贡献边界 |
| 申请灵衢联合测试环境 | 灵衢团队 | UB GVA PoC 硬件验证资源 |
| vLLM-Ascend DSA 接口对齐 | vLLM-Ascend 团队 | DSA Handler 验证前提 |
| Mooncake Maintainer 沟通 | @ykwd/@stmatengss | 建立技术信任，获取认可 |

---

> **数据来源**：详见完整版技术洞察报告 `docs/superpowers/insights/2026-06-10-distributed-kvcache-technology-insight.md` Section 5.3-5.4 及附录 B
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

## 五、生态影响力构建路径与决策建议

### 5.1 路径总览：三阶段获取 Mooncake CODEOWNERS

```mermaid
graph LR
    subgraph Q2Q3["阶段一 Q2-Q3<br/>核心贡献者"]
        A1["Layout Handler PR"]
        A2["Store Top 5"]
    end

    subgraph Q3Q4["阶段二 Q3-Q4<br/>模块主导权"]
        B1["主导热点缓存<br/>架构演进"]
        B2["Reviewer 席位"]
    end

    subgraph Q4Q1["阶段三 Q4-Q1<br/>CODEOWNERS"]
        C1["20+ commits"]
        C2["CODEOWNERS 权限"]
    end

    Q2Q3 --> Q3Q4 --> Q4Q1
```

### 5.2 四个差异化技术切入点

| 优先级 | 切入点 | 已有基础 | 竞争优势 |
|--------|--------|---------|---------|
| **P0** | KVCache Layout Handler | GQA/MLA/Hybrid 代码已完成 | 仅 @ykwd 深度理解（来源：本文分析） |
| **P0** | Ascend NPU 适配 + 灵衢直访 | 热缓存 PR 5+（来源：[SIG 运作报告]） | 灵衢联合验证独有 |
| **P1** | 热点缓存架构演进 | TTFT ↓55-93%（来源：[SIG v25.12]） | 可主导架构讨论 |
| **P1** | 稀疏注意力布局处理器 | 设计完成 | **社区空白**（来源：本文分析） |

### 5.3 关键里程碑

| 里程碑 | 时间 | 验收标准 | 硬件结合点 |
|--------|------|---------|-----------|
| **M1** | 2026 Q3 | Store Top 5 贡献者 | 昇腾 NPU 适配层 |
| **M1.5** | 2026 Q3 | 灵衢 GVA 直访 PoC <1μs | **灵衢 UB 验证** |
| **M2** | 2026 Q4 | Reviewer 席位 + InferNex 增强 | 昇腾性能对标 GPU |
| **M2.5** | 2026 Q4 | Kunpeng+NPU 混合验证 | **鲲鹏 950 + 昇腾** |
| **M3** | 2027 Q1 | 异构互通 PoC | Ascend↔NVIDIA 格式转换 |
| **M4** | 2027 Q2 | 云原生治理平台 | 全栈集成 |

### 5.4 核心风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| Yuanrong Ascend 性能领先 | 持续贡献保持影响力，推动 NPU 成为核心路线图 |
| Kunpeng 950 延迟 | 先用 Kunpeng 920 验证 UB 传输可行性 |
| MemCache 定位冲突 | 明确分工：MemCache 底层引擎，openFuyao 上层编排 |

### 5.5 成功指标

| 指标 | 当前 | 2026 Q4 | 2027 Q2 |
|------|------|---------|---------|
| Store commits | ~10 | 35+ | 50+ |
| CODEOWNERS 状态 | 无 | **Reviewer** | **CODEOWNERS** |
| 超节点 KVCache 延迟 | 未验证 | GVA PoC | 生产级 |
| InferNex E2EL | 22% | 30%+ | 40%+ |

### 5.6 下一步行动建议（决策建议）

**立即启动（本周）**：

| 行动 | 负责人 | 目标 |
|------|--------|------|
| 发起 Layout Handler RFC Issue | 技术负责人 | 进入 Mooncake 社区讨论，展示设计深度 |
| 完善已有代码实现（GQA/MLA/Hybrid） | 开发团队 | RFC 定稿后提交 PR |

**近期推进（Q3 2026）**：

| 行动 | 目标 | 协调需求 |
|------|------|---------|
| 热点缓存优化 PR 合并 | Store Top 5 贡献者 | 与 Mooncake Maintainer (@ykwd) 沟通 |
| 灵衢 GVA 直访 PoC 启动 | UB 零拷贝验证 | 申请灵衢联合测试环境 |

**中期推进（Q4 2026）**：

| 行动 | 目标 | 前置条件 |
|------|------|---------|
| 申请 Store Reviewer 席位 | CODEOWNERS 申请资格 | M1 达成 + 1 位 CODEOWNER 认可 |
| Kunpeng 950 + NPU 混合验证 | 通算超节点能力 | Kunpeng 950 上市 |

**待协调事项**：

| 事项 | 协调对象 | 目的 |
|------|---------|------|
| 与 Yuanrong/MemCache 明确分工 | 产品线 | 避免内部竞争，明确定位边界 |
| 申请灵衢联合测试环境 | 灵衢团队 | 硬件验证资源 |
| vLLM-Ascend DSA 接口对齐 | vLLM-Ascend 团队 | 稀疏注意力验证前提 |

---

> **数据来源**：详见完整版技术洞察报告 `docs/superpowers/insights/2026-06-10-distributed-kvcache-technology-insight.md` 附录 B
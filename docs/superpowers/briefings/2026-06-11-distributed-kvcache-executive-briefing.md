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

> **汇报目的**：为领导层提供分布式 KVCache 领域的技术趋势全景判断，明确 openFuyao 在此领域的差异化定位和生态影响力构建路径。

---

## 一、分布式 KVCache 技术背景与组件全景图

### 1.1 核心解决的场景与痛点

```mermaid
graph LR
    subgraph 痛点["LLM 推理核心瓶颈"]
        P1["KVCache 占 GPU/NPU HBM<br/>60-80%"]
        P2["长上下文场景<br/>KVCache 可达数十 GB"]
        P3["PD 分离场景<br/>跨节点传输延迟高"]
    end

    subgraph 解决方案["分布式 KVCache 方案"]
        S1["多级存储<br/>HBM - DRAM - SSD"]
        S2["高速传输<br/>RDMA/UB 零拷贝"]
        S3["智能缓存管理<br/>命中感知调度"]
    end

    P1 --> S1
    P2 --> S1
    P3 --> S2
    S1 --> S3
    S2 --> S3
```

| 痛点 | 具体表现 | 分布式 KVCache 如何解决 |
|------|----------|------------------------|
| **内存瓶颈** | KVCache 占 HBM 60-80%，限制批处理大小和吞吐 | 多级存储卸载冷 KV 到 DRAM/SSD，释放 HBM |
| **长上下文** | 128K+ 上下文 KVCache 达数十 GB，单卡无法承载 | 分布式存储 + 跨节点传输，突破单卡容量限制 |
| **PD 分离传输** | 预填充节点到解码节点的 KVCache 传输成为延迟瓶颈 | RDMA/UB 零拷贝传输，跨节点延迟降至亚微秒级 |
| **重复计算** | 相同前缀重复计算 KVCache，浪费算力 | 缓存命中复用，避免重复计算 |

**量化收益（行业实测数据）**：

| 指标 | 无 KVCache | 有分布式 KVCache | 来源 |
|------|-----------|-----------------|------|
| 吞吐量 | 基线 | **3.8x 提升** | vLLM + Mooncake (2026.05) |
| TTFT（首 token 延迟） | 基线 | **降低 46x** | vLLM + Mooncake (2026.05) |
| RAG 场景 KVCache 命中率 | ~0% | **接近 100%** | LMCache CacheBlend (EuroSys 2025) |
| 长上下文吞吐量 | 基线 | **5x 提升** | HiSparse (SGLang 2026.04) |

### 1.2 主流组件全景图

```mermaid
graph TB
    subgraph 上层["上层编排与治理"]
        OF["openFuyao / InferNex<br/>云原生推理基础设施<br/>智能路由 + 弹性伸缩 + 可观测"]
    end

    subgraph 中层["中间管理层"]
        HC["HiCache + SGLang<br/>分层 KV 缓存<br/>RadixAttention 深度集成"]
        LMC["LMCache<br/>知识交付网络 KDN<br/>CacheBlend ~100% 命中率"]
    end

    subgraph 底层["底层传输与存储引擎"]
        MK["Mooncake Store + TE<br/>跨硬件统一存储引擎<br/>MIT / PyTorch 生态<br/>6+ 传输协议 / 4+ 硬件平台"]
        YR["Yuanrong Data System<br/>内存中心 Serverless 缓存<br/>Ascend UB 原生 / 分布式元数据"]
    end

    subgraph 硬件层["异构硬件"]
        NVIDIA["NVIDIA GPU"]
        ASCEND["昇腾 NPU"]
        KUNPENG["鲲鹏 CPU"]
        UB_BUS["灵衢 UB 总线"]
    end

    OF --> HC
    OF --> LMC
    OF --> MK

    HC --> MK
    LMC --> MK

    MK --> NVIDIA
    MK --> ASCEND
    YR --> ASCEND
    YR --> UB_BUS

    OF --> ASCEND
    OF --> KUNPENG
    OF --> UB_BUS

    classDef bottom fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef mid fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef top fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef hw fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    class MK,YR bottom
    class HC,LMC mid
    class OF top
    class NVIDIA,ASCEND,KUNPENG,UB_BUS hw
```

**六大系统定位一句话**：

| 系统 | 一句话定位 | 硬件覆盖 | 开源状态 |
|------|-----------|---------|---------|
| **Mooncake** | 跨硬件分布式 KVCache 存储引擎 + 传输引擎 | NVIDIA/AMD/Ascend/MT | MIT, PyTorch 生态 |
| **HiCache** | SGLang 内嵌的分层 KV 缓存（GPU 辅助 I/O 3x 加速） | NVIDIA 为主 | Apache 2.0 |
| **LMCache** | vLLM 与存储之间的知识交付网络（CacheBlend RAG ~100% 命中） | NVIDIA | Apache 2.0 |
| **Yuanrong** | 华为 openEuler 的 Serverless 分布式缓存（UB 总线原生） | 仅 Ascend | Apache 2.0 |
| **MemCache** | 华为 Ascend 专用分布式 KVCache 引擎 | 仅 Ascend | 华为内部 |
| **openFuyao** | 云原生 AI 推理基础设施（智能路由 + KVCache 编排） | x86/ARM/GPU/NPU | Apache 2.0 |

---

## 二、技术演进趋势

### 四大关键趋势总览

```mermaid
graph LR
    T1["趋势 1<br/>存储层级深化<br/>单层 → 三/四层"]
    T2["趋势 2<br/>注意力机制多样化<br/>MHA → GQA/MLA/DSA"]
    T3["趋势 3<br/>异构硬件生态<br/>NVIDIA → 多平台"]
    T4["趋势 4<br/>生态集成深化<br/>独立 → 全栈协作"]

    T1 --> |"驱动"| T2
    T2 --> |"需要"| T3
    T3 --> |"催生"| T4
    T4 --> |"反馈"| T1
```

### 趋势 1：存储层级深化 — 从单层到三/四层

```mermaid
graph LR
    subgraph 存储演进["存储层级演进"]
        direction TB
        V1["V1: 单层<br/>GPU HBM 直传 RDMA"]
        V2["V2: 两层<br/>HBM + DRAM"]
        V3["V3: 三层<br/>HBM + DRAM + SSD<br/>Mooncake Store"]
        V4["V4: 四层<br/>HBM + DRAM + NVMe + 远程<br/>LMCache"]
    end

    V1 --> V2 --> V3 --> V4

    subgraph 关键创新["关键创新"]
        K1["HiCache: GPU 辅助 I/O 内核<br/>3x cudaMemcpy 吞吐"]
        K2["LMCache: NVMe GDS 直通<br/>NUMA 感知分配"]
        K3["Mooncake: 多 NIC 聚合<br/>拓扑感知路径选择"]
    end
```

**趋势判断**：未来 CXL 内存、持久内存等新介质将加入层级体系。华为灵衢 UB 总线通过 GVA 统一编址，可将超节点内所有 NPU HBM 和 CPU DRAM 扁平化为单一地址空间，实现零拷贝跨层访问——这是 NVIDIA 生态无法复制的硬件级优势。

### 趋势 2：注意力机制多样化

```mermaid
graph LR
    MHA["MHA<br/>传统多头注意力"] --> GQA["GQA<br/>分组查询<br/>GLM-4 / Qwen"]
    GQA --> MLA["MLA<br/>潜在注意力<br/>DeepSeek V3<br/>4-8x 存储压缩"]
    MLA --> HYBRID["Hybrid<br/>滑动窗口 + 全局<br/>Qwen 3.5"]
    HYBRID --> DSA["DSA 稀疏注意力<br/>DeepSeek V3.2 / GLM-5.1<br/>仅保留活跃 KV 子集"]
```

| 注意力机制 | 代表模型 | 对 KVCache 存储的影响 |
|-----------|---------|---------------------|
| MHA | 早期模型 | 每 head 独立 K/V，内存占用最大 |
| GQA | GLM-4, Qwen | KV 组共享，减少内存 |
| MLA | DeepSeek V3 | 压缩潜在向量，**存储缩减 4-8x** |
| Hybrid | Qwen 3.5 | 滑动窗口部分可淘汰，减少传输量 |
| DSA 稀疏 | DeepSeek V3.2 | 仅存储活跃 KV 子集，**5x 吞吐提升** |

**趋势判断**：注意力机制将持续多样化。Mooncake Store 已有 MHA/GQA/MLA/Hybrid 四种布局处理器——这是核心差异化优势。稀疏注意力适配是下一个竞争焦点。

### 趋势 3：异构硬件生态 — 尤其是中国市场

```mermaid
graph TB
    subgraph NVIDIA生态["NVIDIA 生态"]
        NV_GPU["NVIDIA GPU"]
        NVLINK["NVLink 节点内<br/>最多 576 GPU"]
        RDMA_NV["RDMA 跨节点<br/>延迟 5-50 us"]
    end

    subgraph 华为生态["华为生态"]
        ASC_NPU["昇腾 NPU"]
        UB_BUS["灵衢 UB 总线<br/>节点间延迟 lt 2 us"]
        KUN_CPU["鲲鹏 CPU"]
        SUPER["超节点 CloudMatrix384<br/>384 NPU 全互联"]
    end

    subgraph 关键差异["关键差异"]
        DIFF1["NVLink: 节点内限 576 GPU<br/>跨节点依赖 RDMA 5-50 us"]
        DIFF2["UB: 节点间带宽损失 lt 3%<br/>节点间延迟仅增 lt 1 us<br/>超节点扁平化为单一逻辑节点"]
    end

    NVLINK --> DIFF1
    UB_BUS --> DIFF2
```

**实测数据（CloudMatrix384）**：

| 指标 | NVIDIA RDMA 跨节点 | 华为 UB 跨节点 | 优势 |
|------|-------------------|---------------|------|
| NPU-NPU 读取带宽 | ~50 GB/s | **164 GB/s** | 3.3x |
| NPU-NPU 读取延迟 | ~5-10 μs | **1.9 μs** | 3-5x |
| 节点内/间带宽比 | 显著下降 | **98%**（损失 <3%） | 数量级优势 |
| KVCache 90% 重用 TTFT | — | **降低 59%** | CloudMatrix 实测 |

### 趋势 4：生态集成深化

**从独立组件到全栈协作**：

```mermaid
graph LR
    subgraph 阶段1["阶段 1: 独立存储"]
        A["Mooncake V1<br/>独立 KVCache 传输"]
    end

    subgraph 阶段2["阶段 2: 引擎集成"]
        B["vLLM KV Connector<br/>SGLang RadixAttention"]
    end

    subgraph 阶段3["阶段 3: 全栈协作"]
        C["LMCache 桥接 vLLM-Mooncake<br/>HiCache 3 函数后端<br/>openFuyao K8s 编排"]
    end

    A --> B --> C
```

**趋势判断**：集成深度正从"put/get 接口"向"注意力感知决策"演进。HiCache 的 3 函数后端接口（get/exist/set）降低了集成门槛——Mooncake Store、DeepSeek 3FS、NVIDIA NIXL 等已作为后端接入。

---

## 三、生态格局与竞争合作关系

### 3.1 六大系统竞合关系图

```mermaid
graph TB
    MK["Mooncake<br/>跨硬件存储引擎<br/>PyTorch 生态"]
    HC["HiCache + SGLang<br/>推理引擎内层缓存"]
    LMC["LMCache<br/>知识交付网络"]
    YR["Yuanrong<br/>Ascend Serverless 缓存"]
    MC["MemCache<br/>Ascend 专用引擎"]
    OF["openFuyao / InferNex<br/>云原生编排层"]

    MK <-.->|战略合作<br/>LMCache 桥接 vLLM-Mooncake| LMC
    MK <-.->|后端合作<br/>Mooncake Store 是 HiCache 后端| HC

    HC <-->|分层缓存竞争<br/>分别绑定 SGLang / vLLM| LMC
    MK <-->|底层存储竞争<br/>跨硬件 vs Ascend 专用| YR
    YR <-->|Ascend 后端竞争<br/>Serverless vs 专用| MC

    OF ==>|上游贡献 + 下游集成<br/>热缓存优化已合并上游| MK

    classDef coop fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    classDef comp fill:#ffcdd2,stroke:#c62828,stroke-width:2px
    classDef supply fill:#bbdefb,stroke:#1565c0,stroke-width:2px
    class MK,LMC,HC coop
    class YR,MC comp
    class OF supply
```

### 3.2 关键竞合判断

| 关系 | 性质 | 判断 |
|------|------|------|
| **Mooncake ↔ LMCache** | 战略合作 | LMCache 作为 vLLM-Mooncake 桥接层，2025.05 合作。实测 TTFT 降低 69.1% |
| **Mooncake ↔ HiCache** | 互补 | Mooncake Store 是 HiCache 远程后端。蚂蚁集团 TTFT 降低 84% |
| **HiCache ↔ LMCache** | 竞争 | 都做分层缓存，分别绑定 SGLang/vLLM。竞争格局跟随推理引擎市场份额 |
| **Mooncake ↔ Yuanrong** | **核心竞争** | 同类底层存储引擎，vLLM-Ascend KVPool 后端直接竞争 |
| **openFuyao → Mooncake** | 上下游 | 上游贡献（热缓存已合并）+ 下游集成（InferNex 使用 Mooncake Store） |

### 3.3 底层存储引擎趋于收敛

```mermaid
graph LR
    subgraph 底层收敛["底层存储引擎 → Mooncake Store 成为主流"]
        MK_CORE["Mooncake Store<br/>事实标准"]
    end

    subgraph 上层竞争["上层管理层 → HiCache vs LMCache 继续竞争"]
        HC2["HiCache<br/>SGLang 绑定"]
        LMC2["LMCache<br/>vLLM 绑定"]
    end

    subgraph 异构变量["中国市场异构变量"]
        YR2["Yuanrong<br/>Ascend 原生优化"]
        OF2["openFuyao<br/>超节点编排 + UB 使能"]
    end

    HC2 --> MK_CORE
    LMC2 --> MK_CORE
    OF2 --> MK_CORE
    YR2 -. Ascend 专有 .-> MK_CORE
```

---

## 四、关键判断总结与 openFuyao 启示

### 判断 1：openFuyao 不应成为"另一个 Mooncake"

**论据**：

| 维度 | Mooncake | openFuyao 应该做 |
|------|----------|-----------------|
| 技术栈层级 | 底层存储引擎 | **上层编排 + 硬件使能** |
| 硬件策略 | 广度覆盖 4+ 平台 | **深度优化昇腾 + 灵衢 UB** |
| 核心能力 | 传输 + 存储 | **智能路由 + 弹性伸缩 + 可观测** |
| 竞争对手 | Yuanrong/MemCache | **无直接对手（新象限）** |

**定位公式**：

```
openFuyao/InferNex = 超节点硬件使能层 + 异构编排调度层 + 云原生治理层 + KVCache 存储优化贡献者
```

### 判断 2：超节点 + UB 总线是核心差异化底座

```mermaid
graph TB
    subgraph NVIDIA路径["NVIDIA 生态路径（4 跳）"]
        N1["NPU HBM"] --> N2["Host DRAM"]
        N2 --> N3["RDMA 网卡"]
        N3 --> N4["对端 Host DRAM"]
        N4 --> N5["对端 NPU HBM"]
    end

    subgraph UB路径["华为 UB 路径（1 跳，零拷贝）"]
        U1["NPU HBM"] --> U2["UB 总线"]
        U2 --> U3["对端 NPU HBM"]
    end

    subgraph 性能对比["性能对比"]
        P1["RDMA: 9-14 us 延迟<br/>40-50 GB/s 带宽"]
        P2["UB GVA: lt 1 us 延迟<br/>gt 100 GB/s 带宽"]
    end
```

**两种超节点场景的 KVCache 发力空间**：

| 场景 | 硬件组合 | KVCache 发力点 | 关键指标 |
|------|---------|---------------|---------|
| **智算超节点** | Ascend 910C/950 + UB 全互联 | L0-L1 层 GVA 零拷贝直访 | 超节点内延迟 <1μs，带宽 >100 GB/s |
| **通算超节点** | Kunpeng 950 + Ascend NPU | CPU DRAM 冷存储 + NPU HBM 热缓存 | CPU-NPU 带宽 >100 GB/s，冷热切换 <2μs |

**实测验证（CloudMatrix384，来源：arXiv 论文）**：

- 节点间 NPU-NPU 读取带宽：164 GB/s（vs 节点内 167 GB/s，损失仅 2%）
- 节点间 NPU-NPU 读取延迟：1.9 μs（vs 节点内 1.2 μs，仅增 0.7 μs）
- KVCache 90% 重用率：预填充吞吐提升 2.28x，TTFT 降低 59%

### 判断 3：Yuanrong 竞争是双刃剑

| 维度 | 正面影响 | 风险因素 |
|------|---------|---------|
| Mooncake 贡献 | 竞争压力推动 Mooncake 重视 Ascend 生态 | Mooncake 可能降低 NPU 优先级 |
| 定位分工 | Yuanrong 做底层存储，openFuyao 做上层编排 | 需明确分工避免内部竞争 |
| 应对策略 | 支持多后端（Mooncake/Yuanrong/MemCache） | 避免绑定单一后端 |

### 判断 4：稀疏注意力是下一个竞争焦点

- Mooncake Store 已有 MHA/GQA/MLA/Hybrid 四种布局处理器——领先优势
- HiSparse 在稀疏注意力场景实现 **5x 吞吐提升**——但尚未集成到存储层
- **openFuyao 机会**：为 Mooncake Store 贡献 DSA 稀疏注意力布局处理器，填补社区空白

---

## 五、openFuyao KVCache 生态影响力构建路径

### 5.1 路径总览：通过 Mooncake 上游席位获取生态影响力

```mermaid
graph LR
    subgraph 阶段一["阶段一 Q2-Q3<br/>核心贡献者"]
        A1["Layout Handler PR"]
        A2["热点缓存优化 PR"]
        A3["Store Top 5 贡献者"]
    end

    subgraph 阶段二["阶段二 Q3-Q4<br/>模块主导权"]
        B1["主导热点缓存<br/>架构演进"]
        B2["NPU 适配层 PR"]
        B3["Reviewer 席位"]
    end

    subgraph 阶段三["阶段三 Q4-Q1<br/>CODEOWNERS"]
        C1["20+ Store commits"]
        C2["3+ 重大 PR"]
        C3["CODEOWNERS 权限"]
    end

    阶段一 --> 阶段二 --> 阶段三
```

### 5.2 四个差异化技术切入点

| 优先级 | 切入点 | 对应趋势 | 已有基础 | 竞争优势 |
|--------|--------|---------|---------|---------|
| **P0** | **KVCache Layout Handler** | 趋势 2（注意力多样化） | GQA/MLA/Hybrid 代码已完成 | 仅 @ykwd 深度理解，缺乏第二专家 |
| **P0** | **Ascend NPU 适配层 + 灵衢直访** | 趋势 3（异构硬件） | 热缓存 PR 5+ 已提交，灵衢合作已建 | 灵衢联合验证场景独有 |
| **P1** | **热点缓存架构演进** | 趋势 1（存储层级） | 已有性能数据（TTFT ↓55-93%） | 可主导该方向架构讨论 |
| **P1** | **稀疏注意力布局处理器** | 趋势 2 | 设计完成 | **社区无人在此方向发力** |

### 5.3 关键里程碑与验收标准

```mermaid
gantt
    title openFuyao KVCache 生态影响力构建路线图
    dateFormat YYYY-MM-DD
    axisFormat %Y-%m

    section 上游贡献线
    Layout Handler RFC + PR           :a1, 2026-07-01, 60d
    热点缓存优化持续贡献              :a2, 2026-07-01, 120d
    NPU 适配层 PR                     :a3, 2026-09-01, 60d
    稀疏注意力布局处理器              :a4, 2026-10-01, 60d
    灵衢 GVA 直访传输后端             :a5, 2026-10-01, 60d

    section 自研体系线
    智算超节点 KVCache 零拷贝验证     :b1, 2026-07-01, 120d
    InferNex KVCache 感知调度增强      :b2, 2026-07-01, 120d
    云原生 KVCache Operator           :b3, 2026-07-01, 120d
    异构集群 KVCache 互通              :b4, 2027-01-01, 90d
    通算超节点混合 KVCache 验证        :b5, 2027-01-01, 90d

    section 里程碑
    M1: Store Top 5 贡献者            :m1, 2026-09-30, 0d
    M1.5: 灵衢 GVA 直访 PoC           :m1_5, 2026-09-30, 0d
    M2: Reviewer 席位 + InferNex 增强  :m2, 2026-12-31, 0d
    M3: 异构互通 PoC                   :m3, 2027-03-31, 0d
    M3.5: 超节点验证                   :m3_5, 2027-03-31, 0d
    M4: 云原生治理平台                 :m4, 2027-06-30, 0d
```

| 里程碑 | 时间 | 核心验收标准 | 对昇腾/鲲鹏/灵衢的结合 |
|--------|------|-------------|----------------------|
| **M1** | 2026 Q3 | Store Top 5 贡献者，Layout Handler PR 合并 | 昇腾 NPU 适配层作为贡献核心 |
| **M1.5** | 2026 Q3 | 灵衢 GVA 直访 KVCache PoC 验证 | **灵衢总线**零拷贝验证 |
| **M2** | 2026 Q4 | Store Reviewer 席位，InferNex 增强版发布 | 昇腾性能对标 GPU 版（差距 <10%） |
| **M2.5** | 2026 Q4 | 通算超节点 Kunpeng+NPU 混合验证 | **鲲鹏 950 + 昇腾** UB 分层验证 |
| **M3** | 2027 Q1 | 异构集群 Ascend↔NVIDIA 互通 PoC | 跨硬件 KVCache 格式转换 |
| **M3.5** | 2027 Q1 | 超节点 KVCache 能力验证 | **智算超节点** GVA + **通算超节点**混合 |
| **M4** | 2027 Q2 | 云原生 KVCache 治理平台发布 | 全栈昇腾/鲲鹏/灵衢集成 |

### 5.4 核心风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Yuanrong 在 Ascend 性能大幅领先 | Mooncake 社区降低 NPU 优先级 | 持续高质量贡献保持影响力，推动 NPU 成为核心路线图 |
| Kunpeng 950 上市延迟（目标 Q4） | 通算超节点验证受阻 | 先用 Kunpeng 920 验证 UB 传输可行性 |
| MemCache 与 openFuyao 定位冲突 | 内部重复投入 | 明确分工：MemCache 做底层引擎，openFuyao 做上层编排 |
| vLLM-Ascend DSA 接口延迟 | 稀疏注意力验证受阻 | 联合对齐排期，同步准备算法侧验证 |

### 5.5 成功指标

| 指标 | 当前 | 2026 Q3 | 2026 Q4 | 2027 Q2 |
|------|------|---------|---------|---------|
| Store commits | ~10 | 20+ | 35+ | 50+ |
| Merged PRs | ~5 | 10+ | 18+ | 25+ |
| CODEOWNERS 状态 | 无 | 1 位认可 | **Reviewer 席位** | **CODEOWNERS** |
| 超节点 KVCache 延迟 | 未验证 | GVA PoC <1μs | — | 生产级验证 |
| InferNex E2EL 改善 | 22% | — | 30%+ | 40%+ |

---

> **数据来源**：详见完整版报告 `docs/superpowers/insights/2026-06-10-distributed-kvcache-technology-insight.md` 附录 B

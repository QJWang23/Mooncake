# Qwen3.5 混合注意力 KVCache ��化设计方案

**日期**: 2026-03-16
**作者**: Claude (AI Assistant)
**状态**: 待审核
**版本**: 1.0

---

## 执行摘要

本设计文档描述了如何优化 Mooncake 的 KV Offloading 机制以支持 Qwen3.5 的混合注意力机制（Hybrid Attention）。Qwen3.5 采用了 Gated DeltaNet 线性注意力与 GQA 完整注意力的混合架构，其 KVCache 存储需求与传统架构有显著差异。

**核心优化目标：**
- 支持 128K-256K 超长上下文
- 传输量减少约 75%
- 存储效率提升 4 倍
- 最小化传输延迟

---

## 目录

1. [技术背景与问题分析](#1-技术背景与问题分析)
2. [系统架构设计](#2-系统架构设计)
3. [核心组件设计](#3-核心组件设计)
4. [数据流设计](#4-数据流设计)
5. [API 设计](#5-api-设计)
6. [测试策略](#6-测试策略)
7. [实现路线图](#7-实现路线图)
8. [预期收益与风险评估](#8-预期收益与风险评估)

---

## 1. 技术背景与问题分析

### 1.1 Qwen3.5 混合注意力机制概述

Qwen3.5 的架构设计融合了**三大核心技术组件**：

| 组件 | 说明 | 占比 |
|------|------|------|
| **Gated Delta Networks (GDN)** | 线性注意力机制，复杂度O(L·d²) | ~75%的层 |
| **Gated Attention (GQA)** | 完整注意力机制，传统O(L²·d) | ~25%的层 |
| **Sparse MoE (稀疏混合专家)** | 高稀疏度专家路由机制 | FFN层 |

### 1.2 Gated DeltaNet 线性注意力技术原理

**与传统注意力的核心差异：**

```
传统自注意力计算：
Attention(Q, K, V) = softmax(QK^T / √d_k) · V   // 复杂度 O(L²·d)

Gated DeltaNet 线性注意力：
LinearAttention(Q, K, V) = φ(Q)(φ(K)^T · V)      // 复杂度 O(L·d²)
```

**核心创新机制：**

- **门控参数 β**：控制每个时间步的记忆保留比例，实现快速"遗忘"过期信息
- **增量参数 Δ**：支持对特定记忆位置的精确更新，避免全量覆盖

**状态更新公式：**
```
S_t = β_t ⊙ S_{t-1} + Δ_t ⊗ (K_t ⊗ V_t)
```

### 1.2.1 GDN 与滑动窗口注意力（Sliding Window Attention）的本质区别

**关键区分：GDN ≠ 滑动窗口注意力**

| 维度 | Gated DeltaNet (GDN) | Sliding Window Attention (SWA) |
|------|----------------------|-------------------------------|
| **核心机制** | Delta规��� + 门控状态更新 | 固定窗口大小，丢弃旧token |
| **记忆保留** | 全局历史压缩到固定状态 | 仅保留窗口内token，窗口外直接丢弃 |
| **状态大小** | O(1) 固定大小，不随序列增长 | O(W) 窗口大小，固定但需存储窗口内所有KV |
| **全局上下文** | 通过状态矩阵保留全局信息 | 丢失窗口外所有信息 |
| **更新方式** | 增量式精确更新（类似"橡皮擦"） | 滑动式FIFO淘汰 |
| **信息检索** | 通过状态解码检索任意历史 | 无法检索窗口外信息 |

**为什么Qwen3.5选择GDN而非纯SWA：**

```
滑动窗口注意力的局限：
┌─────────────────────────────────────────────────────────────────┐
│  Token序列: [1][2][3]...[W-2][W-1][W]...[N-W]...[N-2][N-1][N]          │
│                          ↑______窗口内______↑   ↑_丢弃_↑             │
│                                                                  │
│  问题：Token 1...N-W 的信息完全丢失，无法进行全局信息检索          │
└─────────────────────────────────────────────────────────────────┘

Gated DeltaNet的解决方案：
┌─────────────────────────────────────────────────────────────────┐
│  Token序列: [1][2][3]...[N-2][N-1][N]                              │
│                    ↓                                           │
│              ┌─────────────────────┐                             │
│              │  压缩状态矩阵 S_t    │  ← 固定大小，包含全局信息    │
│              │  大小: [H, D, StateDim] │                             │
│              └─────────────────────┘                             │
│                                                                  │
│  优势：所有历史信息压缩保存在状态中，可进行全局检索          │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2.2 SSM状态张量结构详解

**GDN层的SSM状态包含以下张量组件：**

```cpp
// 单层SSM状态结构
struct SSMStateLayer {
    // === 权重张量（从模型加载，固定不变）===
    const float* A_log;           // 状态转移矩阵 [num_key_heads, head_dim]
                                 // 存储为对数形式: A = -exp(A_log) 保证数值稳定性
    const float* dt_proj_weight;  // 门控投影权重 [num_key_heads, hidden_dim]
    const float* dt_proj_bias;     // 门控投影偏置 [num_key_heads]
    const float* conv1d_weight;    // 1D卷积权重 [num_key_heads, 1, kernel_dim]
                                 // kernel_dim通常为4，用于局部依赖捕获
    const float* D_proj;           // 跳跃连接权重 [num_key_heads]

    // === 运行时状态（随序列动态更新）===
    float* ssm_state;              // SSM核心状态 [num_key_heads, head_dim, state_dim]
                                 // state_dim = 256 (典型值)
                                 // 这是"压缩的全局历史"
    float* conv_state;             // 卷积状态 [num_key_heads, kernel_dim]
                                 // 用于缓存卷积计算的中间结果
};
```

**各张量的作用说明：**

| 张量 | 形状 | 作用 | 存储需求 |
|------|------|------|----------|
| `A_log` | [H, D] | 状态转移矩阵，控制信息保留/遗忘 | 固定，随模型加载 |
| `dt_proj_weight` | [H, hidden_dim] | 生成时间步相关的门控参数 | 固定 |
| `dt_proj_bias` | [H] | 门控偏置 | 固定 |
| `conv1d_weight` | [H, 1, 4] | 局部依赖捕获（类似n-gram） | 固定 |
| `D_proj` | [H] | 跳跃连接，增强梯度流 | 固定 |
| `ssm_state` | [H, D, 256] | **核心状态**，压缩的全局历史 | **需存储** |
| `conv_state` | [H, 4] | 卷积缓存 | **需存储** |

**单层SSM状态大小计算：**
```
单层大小 = ssm_state + conv_state
         = H × D × state_dim × 4 + H × kernel_dim × 4
         = 16 × 128 × 256 × 4 + 16 × 4 × 4
         = 2,097,152 + 256
         ≈ 2MB per layer

18层GDN总状态大小 ≈ 18 × 2MB = 36MB (固定，不随序列长度变化)
```

### 1.2.3 Delta规则的核心原理

**传统线性注意力的问题：**
```
传统更新: S_t = S_{t-1} + K_t ⊗ V_t  // 简单累加

问题：
- 信息不断叠加，无法"擦除"过时内容
- 状态膨胀，信噪比下降
- 类似于"只写入不删除"的内存泄漏
```

**Delta规则的解决方案：**
```
Delta更新: S_t = S_{t-1} - Δ_pred ⊗ K_pred + Δ_new ⊗ K_new

核心思想：
1. 先"预测"要更新的位置 (K_pred)
2. 计算需要"擦除"的内容 (Δ_pred)
3. 添加新内容 (Δ_new ⊗ K_new)

类比：
┌─────────────────────────────────────────────────────────────────┐
│  传统线性注意力: 像在黑板上不断写新内容，从不擦除          │
│                     ↓                                           │
│  [内容1][内容2][内容3]...[内容N]  ← 信息混杂，难以检索        │
│                                                                  │
│  Delta规则: 像使用橡皮擦精确擦除旧内容后再写新内容          │
│                     ↓                                           │
│  [精确更新的内容]  ← 信息清晰，易于检索                       │
└─────────────────────────────────────────────────────────────────┘
```

**门控机制 (Gating) 的作用：**
```cpp
// 门控参数 β 控制记忆保留比例
β_t = σ(dt_proj(x_t))  // σ为sigmoid，β ∈ (0, 1)

// 状态更新
S_t = β_t ⊙ S_{t-1} + (1 - β_t) ⊙ Δ_t

// 作用：
// - β_t → 1: 保留历史，新信息影响小（处理已知模式）
// - β_t → 0: 快速遗忘，接受新信息（处理新模式）
```

### 1.3 层级配置策略（3:1混合比例）

```
Qwen3.5 层级结构示例（24层）：
┌─────────────────────────────────────────────────┐
│ Block 0: [GDN] [GDN] [GDN] [GQA]              │
│ Block 1: [GDN] [GDN] [GDN] [GQA]              │
│ Block 2: [GDN] [GDN] [GDN] [GQA]              │
│ ...                                            │
│ Block 5: [GDN] [GDN] [GDN] [GQA]              │
└─────────────────────────────────────────────────┘
→ 18层线性注意力 + 6层完整注意力 = 75% + 25%
```

**关键发现：Gated DeltaNet层不需要存储传统KVCache！**
- GDN使用递归状态更新，只需维护固定大小的状态矩阵
- 这意味着~75%的层**不需要**存储随序列长度增长的KVCache
- 只有GQA完整注意力层（~25%）需要传统KVCache存储

### 1.4 计算效率与内存对比

**计算效率对比（24层、32K序列）：**

| 注意力类型 | 完整注意力层数 | 线性注意力层数 | 相对计算量 |
|------------|----------------|----------------|------------|
| 纯完整注意力 | 24 | 0 | 100% |
| 纯线性注意力 | 0 | 24 | ~25% |
| **混合注意力（3:1）** | 6 | 18 | **~44%** |

**内存占用对比：**

| 序列长度 | 完整注意力内存 | 线性注意力内存 | 混合注意力内存 |
|----------|----------------|----------------|----------------|
| 8K | 512 MB | 64 MB | ~173 MB |
| 32K | 8 GB | 256 MB | ~2.7 GB |
| 128K | 128 GB | 1 GB | ~34 GB |
| **262K** | 512 GB | 2 GB | **~130 GB** |

### 1.5 对现有 Mooncake KV Offloading 的核心影响

**影响1：存储格式不匹配**
- 现有 `HybridLayoutHandler` 仅支持单一切片窗口模式
- Qwen3.5需要**按层级区分存储策略**

**影响2：传输效率问题**
- 全注意力层：128K tokens × 多层 = 大数据量传输瓶颈
- 窗口注意力层：仅4K窗口，但如果当作完整序列传输则浪费带宽

**影响3：增量更新需求**
- 窗口层在生成过程中需要滑动窗口更新
- 当前API不支持部分KVCache更新

---

## 2. KVCache 数据结构详解

### 2.1 传统 KVCache 数据结构

**MHA（Multi-Head Attention）结构：**

```
内存布局（单个请求）：
┌─────────────────────────────────────────────────────────────────┐
│                        K Cache                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Layer 0: [Head 0][Head 1]...[Head H-1]                  │   │
│  │          每个Head: [SeqLen × HeadDim]                   │   │
│  ├───────────────────────��─────────────────────────────────┤   │
│  │ Layer 1: [Head 0][Head 1]...[Head H-1]                  │   │
│  │ ...                                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                        V Cache                                   │
│  （结构同K Cache）                                               │
└─────────────────────────────────────────────────────────────────┘

大小计算：Size = 2 × L × H × S × D × sizeof(float)
其中：L=层数, H=注意力头数, S=序列长度, D=头维度
```

**GQA（Grouped Query Attention）结构：**

```
大小计算：Size = 2 × L × G × S × D × sizeof(float)
其中：G=KV组数 (G << H，通常G=H/8到H/4)
```

**MLA（Multi-Head Latent Attention）结构：**

```
大小计算：Size = L × S × D_latent × sizeof(float)
其中：D_latent << H × D（通常 D_latent ≈ D）
```

### 2.2 Mooncake 当前的存储结构

**统一序列化格式：**

```
┌─────────────────────────────────────────────────────────────────┐
│                    SerializedHeader (12 bytes)                   │
│  ┌────────────┬────────────┬────────────────────────────────┐  │
│  │ magic (4B) │ version(4B)│ metadata_json_length (4B)      │  │
│  └────────────┴────────────┴────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Metadata JSON (变长)                          │
├─────────────────────────────────────────────────────────────────┤
│                    KV Data (原始张量数据)                         │
└─────────────────────────────────────────────────────────────────┘
```

**各架构的Magic Number：**

| 布局类型 | Magic Number | ASCII |
|----------|--------------|-------|
| MHA | `0x4D484143` | "MHAC" |
| GQA | `0x4741434B` | "GACK" |
| MLA | `0x4D4C4143` | "MLAC" |
| HYBRID | `0x48594244` | "HYBD" |

### 2.3 Qwen3.5 混合注意力下的 KVCache 数据结构

```
Qwen3.5 混合注意力 KVCache 结构：

┌─────────────────────────────────────────────────────────────────┐
│                     完整请求的KVCache                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │          Part A: SSM状态（GDN线性注意力层）                │ │
│  │                                                            │ │
│  │  Layer 0:  [A_log][dt_proj.weight][conv1d.weight]         │ │
│  │            [D_proj][state_matrix]                          │ │
│  │            大小固定，不随序列长度变化                       │ │
│  │  Layer 1:  ...（同上）                                     │ │
│  │  ...（跳过Layer 3,7,11...全注意力层）                      │ │
│  │                                                            │ │
│  │  特点：固定大小，O(1)空间复杂度                            │ │
│  │  大小：≈ 18 layers × SSM_state_size ≈ 36MB               │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │          Part B: KV Cache（GQA完整注意力层）               │ │
│  │                                                            │ │
│  │  Layer 3:  [K Cache][V Cache]                              │ │
│  │  Layer 7:  ...（同上）                                     │ │
│  │  ...（共6层，25%）                                         │ │
│  │                                                            │ │
│  │  特点：随序列长度线性增长，O(L)空间复杂度                   │ │
│  │  大小：≈ 6 layers × 2 × KV_groups × SeqLen × HeadDim      │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**内存大小对比（128K上下文示例）：**

| 存储方式 | SSM状态 | KV Cache | 总计 | 相比传统GQA |
|----------|---------|----------|------|-------------|
| 传统GQA（24层） | 0 | ~12GB | 12GB | 100% |
| **Qwen3.5混合** | ~36MB | ~3GB | ~3.04GB | **25%** |

### 2.4 层索引映射与张量结构

#### 2.4.1 层索引映射关系

**物理层索引到逻辑组件的映射：**

```
24层模型示例（full_attention_interval = 4）：

物理层索引    层类型          逻���组件索引
───────────────────────────────────────────
Layer 0      LINEAR_ATTENTION   → SSM State[0]
Layer 1      LINEAR_ATTENTION   → SSM State[1]
Layer 2      LINEAR_ATTENTION   → SSM State[2]
Layer 3      FULL_ATTENTION     → KV Cache[0] (K[0], V[0])
Layer 4      LINEAR_ATTENTION   → SSM State[3]
Layer 5      LINEAR_ATTENTION   → SSM State[4]
Layer 6      LINEAR_ATTENTION   → SSM State[5]
Layer 7      FULL_ATTENTION     → KV Cache[1] (K[1], V[1])
Layer 8      LINEAR_ATTENTION   → SSM State[6]
Layer 9      LINEAR_ATTENTION   → SSM State[7]
Layer 10     LINEAR_ATTENTION   → SSM State[8]
Layer 11     FULL_ATTENTION     → KV Cache[2] (K[2], V[2])
Layer 12     LINEAR_ATTENTION   → SSM State[9]
Layer 13     LINEAR_ATTENTION   → SSM State[10]
Layer 14     LINEAR_ATTENTION   → SSM State[11]
Layer 15     FULL_ATTENTION     → KV Cache[3] (K[3], V[3])
Layer 16     LINEAR_ATTENTION   → SSM State[12]
Layer 17     LINEAR_ATTENTION   → SSM State[13]
Layer 18     LINEAR_ATTENTION   → SSM State[14]
Layer 19     FULL_ATTENTION     → KV Cache[4] (K[4], V[4])
Layer 20     LINEAR_ATTENTION   → SSM State[15]
Layer 21     LINEAR_ATTENTION   → SSM State[16]
Layer 22     LINEAR_ATTENTION   → SSM State[17]
Layer 23     FULL_ATTENTION     → KV Cache[5] (K[5], V[5])
```

**索引计算公式：**

```cpp
// 从物理层索引获取层类型
LayerType getLayerType(uint32_t physical_layer_idx, uint32_t interval) {
    return ((physical_layer_idx + 1) % interval == 0)
        ? LayerType::FULL_ATTENTION
        : LayerType::LINEAR_ATTENTION;
}

// 从物理层索引获取SSM逻辑索引
uint32_t toSSMIndex(uint32_t physical_layer_idx, uint32_t interval) {
    uint32_t full_count = (physical_layer_idx + 1) / interval;
    return physical_layer_idx - full_count;
}

// 从物理层索引获取KV逻辑索引
uint32_t toKVIndex(uint32_t physical_layer_idx, uint32_t interval) {
    return (physical_layer_idx + 1) / interval - 1;
}

// 从KV逻辑索引获取物理层索引
uint32_t toPhysicalLayerIndex(uint32_t kv_idx, uint32_t interval) {
    return kv_idx * interval + (interval - 1);
}
```

#### 2.4.2 KV张量详细结构

**GQA完整注意力层的KV张量布局：**

```
单个GQA层的KV Cache张量结构：

┌─────────────────────────────────────────────────────────────────────────────┐
│                            K Cache张量                                        │
│  Shape: [num_kv_groups, seq_len, head_dim]                                  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ KV_Group 0                                                               ││
│  │   [Token 0]: [d_0, d_1, d_2, ..., d_{head_dim-1}]                       ││
│  │   [Token 1]: [d_0, d_1, d_2, ..., d_{head_dim-1}]                       ││
│  │   ...                                                                    ││
│  │   [Token seq_len-1]: [d_0, d_1, ..., d_{head_dim-1}]                    ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │ KV_Group 1                                                               ││
│  │   ... (同上)                                                             ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │ ...                                                                      ││
│  │ KV_Group {num_kv_groups-1}                                               ││
│  │   ... (同上)                                                             ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  内存布局（C顺序）:                                                          │
│  [G0_T0][G0_T1]...[G0_T{S-1}][G1_T0]...[G{G-1}_T{S-1}]                   │
│                                                                              │
│  大小 = num_kv_groups × seq_len × head_dim × sizeof(float)                  │
│       = 4 × 128K × 128 × 4B = 256MB per layer (K or V)                      │
│       = 512MB per layer (K + V)                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**GDN线性注意力层的SSM状态张量布局：**

```
单个GDN层的SSM状态张量结构：

┌─────────────────────────────────────────────────────────────────────────────┐
│                         SSM State张量                                        │
│  Shape: [num_key_heads, head_dim, state_dim]                                │
│                                                                              │
│  ���─────────────────────────────────────────────────────────────────────────┐│
│  │ Key_Head 0                                                               ││
│  │   [d_0]: [s_0, s_1, ..., s_{state_dim-1}]    ← 压缩的历史状态           ││
│  │   [d_1]: [s_0, s_1, ..., s_{state_dim-1}]                               ││
│  │   ...                                                                    ││
│  │   [d_{head_dim-1}]: [s_0, ..., s_{state_dim-1}]                         ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │ Key_Head 1                                                               ││
│  │   ... (同上)                                                             ││
│  ├─────────────────────────────────────────────────────────────────────────┤│
│  │ ...                                                                      ││
│  │ Key_Head {num_key_heads-1}                                               ││
│  │   ... (同上)                                                             ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  内存布局（C顺序）:                                                          │
│  [H0_D0_S0...S255][H0_D1_S0...S255]...[H{15}_D{127}_S0...S255]            │
│                                                                              │
│  大小 = num_key_heads × head_dim × state_dim × sizeof(float)                │
│       = 16 × 128 × 256 × 4B = 2MB per layer (固定，不随序列长度变化)        │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 2.4.3 存储序列化格式

**混合布局的完整序列化格式：**

```
序列化字节流布局：

┌─────────────────────────────────────────────────────────────────────────────┐
│ Offset    │ Size      │ Content                                            │
├───────────┼───────────┼────────────────────────────────────────────────────┤
│ 0x0000    │ 16 B      │ SerializedHeader                                   │
│           │           │   - magic: 0x51333548 ("Q35H")                     │
│           │           │   - version: 1                                     │
│           │           │   - metadata_json_length                           │
│           │           │   - reserved (future use)                          │
├───────────┼───────────┼────────────────────────────────────────────────────┤
│ 0x0010    │ ~2KB      │ Metadata JSON                                      │
│           │           │   {                                                │
│           │           │     "layout_type": 5,                              │
│           │           │     "num_layers": 24,                              │
│           │           │     "seq_len": 131072,                             │
│           │           │     "layer_type_bitmap": [0x77, 0xDD, 0xEE],        │
│           │           │     "num_kv_groups": 4,                             │
│           │           │     "head_dim": 128,                                │
│           │           │     "ssm_states_offset": 0x1000,                    │
│           │           │     "ssm_states_size": 37748736,                    │
│           │           │     "kv_cache_offset": 0x266D000,                   │
│           │           │     "kv_cache_size": 3221225472,                    │
│           │           │     ...                                              │
│           │           │   }                                                 │
├───────────┼───────────┼────────────────────────────────────────────────────┤
│ ssm_offset │ ~36MB     │ Part A: SSM States (按物理层顺序)                   │
│           │           │   ┌─────────────────────────────────────────────┐  │
│           │           │   │ SSM State for Layer 0 (~2MB)                │  │
│           │           │   │   - ssm_state: [16, 128, 256]               │  │
│           │           │   │   - conv_state: [16, 4]                      │  │
│           │           │   ├─────────────────────────────────────────────┤  │
│           │           │   │ SSM State for Layer 1 (~2MB)                │  │
│           │           │   │ ...                                          │  │
│           │           │   │ (共18个GDN层的SSM状态)                        │  │
│           │           │   └─────────────────────────────────────────────┘  │
├───────────┼───────────┼────────────────────────────────────────────────────┤
│ kv_offset  │ ~3GB      │ Part B: KV Cache (按物理层顺序)                    │
│           │           │   ┌─────────────────────────────────────────────┐  │
│           │           │   │ KV Cache for Layer 3 (~512MB)               │  │
│           │           │   │   - K: [4, 131072, 128] (~256MB)            │  │
│           │           │   │   - V: [4, 131072, 128] (~256MB)            │  │
│           │           │   ├─────────────────────────────────────────────┤  │
│           │           │   │ KV Cache for Layer 7 (~512MB)               │  │
│           │           │   │ ...                                          │  │
│           │           │   │ (共6个GQA层的KV Cache)                        │  │
│           │           │   └─────────────────────────────────────────────┘  │
└───────────┴───────────┴────────────────────────────────────────────────────┘
```

**关键偏移量计算：**

```cpp
// 计算SSM状态偏移量
uint64_t calculateSSMStatesOffset(size_t header_size, size_t json_size) {
    // 对齐到4KB边界，优化RDMA传输
    return ((header_size + json_size + 4095) / 4096) * 4096;
}

// 计算KV Cache偏移量
uint64_t calculateKVCacheOffset(uint64_t ssm_offset, size_t ssm_size) {
    // 对齐到4KB边界
    return ((ssm_offset + ssm_size + 4095) / 4096) * 4096;
}

// 计算单层SSM状态在序列化流中的偏移
uint64_t getLayerSSMOffset(uint32_t physical_layer_idx,
                           uint64_t ssm_base_offset,
                           size_t single_ssm_size) {
    uint32_t ssm_idx = toSSMIndex(physical_layer_idx, interval);
    return ssm_base_offset + ssm_idx * single_ssm_size;
}

// 计算单层KV Cache在序列化流中的偏移
uint64_t getLayerKVOffset(uint32_t physical_layer_idx,
                          uint64_t kv_base_offset,
                          size_t single_kv_size) {
    uint32_t kv_idx = toKVIndex(physical_layer_idx, interval);
    return kv_base_offset + kv_idx * single_kv_size;
}
```

---

## 3. 系统架构设计

### 3.1 整体架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Mooncake Store 层级架构                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    Integration Layer (vLLM/SGLang)                 │ │
│  │  ┌────────────────────────────────────────���─────────────────────┐  │ │
│  │  │  Qwen35KVConnector / Qwen35HybridAttentionAdapter            │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│                                    ▼                                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    Layout Handler Layer                            │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │  Qwen35HybridLayoutHandler (新增)                            │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  │  现有Handlers:  GQALayoutHandler | MLALayoutHandler | ...         │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│                                    ▼                                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    Storage & Transfer Layer                        │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │ │
│  │  │   DRAM Buffer   │  │ Transfer Engine │  │   Metadata      │   │ │
│  │  │   Pool          │  │   (RDMA/TCP)    │  │   (etcd)        │   │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 组件职责划分

| 组件 | 职责 | 输入 | 输出 |
|------|------|------|------|
| **Qwen35KVConnector** | 与推理框架对接，识别模型架构 | 模型配置、KV Cache张量 | 分离的SSM状态和KV数据 |
| **Qwen35HybridLayoutHandler** | 序列化/反序列化混合注意力KVCache | 分离数据 + 元数据 | 统一字节流 |
| **SSMStateManager** | 管理GDN线性注意力层的SSM状态 | 层索引、状态张量 | 压缩的状态数据 |
| **GQAKVCacheManager** | 管理GQA完整注意力层的KV Cache | 层索引、KV张量 | KV字节数据 |
| **LayerTypeResolver** | 解析层类型配置 | model_config.json | 层类型位图 |

### 3.3 核心识��差异：Mooncake需要识别的关键信息

**3.3.1 模型架构识别**

Mooncake需要从推理引擎获取以下关键信息来正确处理混合注意力KVCache：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    需要识别的核心信息                                        │
├─────────────────────────────────────────────────────────────────────────────┤
��                                                                              │
│  1. 模型架构类型识别                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 输入: model_config.json                                             │    │
│  │                                                                      │    │
│  │ 关键字段:                                                            │    │
│  │   - "model_type": "qwen3_5_text"                                    │    │
│  │   - "architectures": ["Qwen3_5ForCausalLM"]                         │    │
│  │   - "layer_types": ["linear_attention", "linear_attention", ...]   │    │
│  │                                                                      │    │
│  │ 识别逻辑:                                                            │    │
│  │   if "layer_types" field exists:                                    │    │
│  │       → HYBRID_ATTENTION model (需要混合处理)                        │    │
│  │   else if all layers are "full_attention":                           │    │
│  │       → STANDARD_GQA model (传统处理)                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  2. 层类型配置解析                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 从 model_config.json 提取:                                         │    │
│  │                                                                      │    │
│  │ {                                                                    │    │
│  │   "num_hidden_layers": 24,                                          │    │
│  │   "layer_types": [                                                  │    │
│  │     "linear_attention", "linear_attention", "linear_attention",     │    │
│  │     "full_attention",                                               │    │
│  │     "linear_attention", "linear_attention", "linear_attention",     │    │
│  │     "full_attention",                                               │    │
│  │     ... (重复pattern)                                                │    │
│  │   ],                                                                 │    │
│  │   "full_attention_interval": 4,  // 每4层一个完整注意力层            │    │
│  │                                                                      │    │
│  │   // GDN参数                                                         │    │
│  │   "linear_num_key_heads": 16,                                       │    │
│  │   "linear_key_head_dim": 128,                                       │    │
│  │   "linear_value_head_dim": 128,                                     │    │
│  │   "linear_conv_kernel_dim": 4,                                      │    │
│  │   "ssm_state_dim": 256,          // SSM状态维度                     │    │
│  │                                                                      │    │
│  │   // GQA参数                                                         │    │
│  │   "num_attention_heads": 32,        // Query头数                    │    │
│  │   "num_key_value_heads": 4,         // KV头组数                     │    │
│  │   "head_dim": 128                                                    │    │
│  │ }                                                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  3. 运行时状态差异识别                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ GDN层状态结构 (与标准KV Cache完全不同):                             │    │
│  │                                                                      │    │
│  │ GDN层 past_key_values[layer_idx] = {                                │    │
│  │     "ssm_state": Tensor[num_heads, head_dim, state_dim],            │    │
│  │     "conv_state": Tensor[num_heads, kernel_dim]                     │    │
│  │ }                                                                    │    │
│  │                                                                      │    │
│  │ GQA层 past_key_values[layer_idx] = {                                │    │
│  │     "key": Tensor[num_kv_groups, seq_len, head_dim],                │    │
│  │     "value": Tensor[num_kv_groups, seq_len, head_dim]               │    │
│  │ }                                                                    │    │
│  │                                                                      │    │
│  │ 识别方法:                                                            │    │
│  │   if past_key_values[layer_idx] has "ssm_state" key:                │    │
│  │       → GDN层，提取SSM状态                                           │    │
│  │   elif past_key_values[layer_idx] has "key" and "value" keys:       │    │
│  │       → GQA层，提取KV Cache                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**3.3.2 推理引擎状态结构差异**

| 推理引擎 | GDN层状态结构 | GQA层状态结构 |
|----------|---------------|---------------|
| **vLLM** | `MambaCache` 对象，包含 `ssm_states` 和 `conv_states` | `torch.Tensor` [2, batch, num_kv_groups, seq_len, head_dim] |
| **SGLang** | `MambaState` 对象，字段 `states` 和 `conv_states` | `torch.Tensor` 结构类似vLLM |
| **TensorRT-LLM** | `MambaConvState` + `MambaSsmState` 分离存储 | `KVCache` tensor |

**关键差异点：**

```python
# vLLM中的状态结构示例
class MambaCache:
    def __init__(self, config, max_batch_size):
        # GDN层使用这个结构
        self.ssm_states = torch.zeros(
            max_batch_size,
            config.num_hidden_layers,
            config.num_key_heads,
            config.head_dim,
            config.ssm_state_dim  # 256
        )
        self.conv_states = torch.zeros(
            max_batch_size,
            config.num_hidden_layers,
            config.num_key_heads,
            config.conv_kernel_dim  # 4
        )

# GQA层使用传统结构
class KVCache:
    # shape: [2, batch, num_layers, num_kv_groups, seq_len, head_dim]
    self.key = torch.zeros(...)
    self.value = torch.zeros(...)
```

### 3.4 与推理引擎的协同机制

**3.4.1 Prefill阶段协同流程**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Prefill阶段：推理引擎 → Mooncake                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Step 1: 推理引擎完成Prefill                                            │  │
│  │                                                                        │  │
│  │ vLLM/SGLang内部状态:                                                   │  │
│  │   - 已完成prompt的所有token计算                                        │  │
│  │   - 每层的状态已更新:                                                  │  │
│  │     * GDN层: ssm_state, conv_state 已递归更新                         │  │
│  │     * GQA层: key_cache, value_cache 已填充完整序列                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                  │                                           │
���                                  ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Step 2: Mooncake请求状态提取                                           │  │
│  │                                                                        │  │
│  │ Qwen35KVConnector.extractFromEngine(engine, request_id):              │  │
│  │                                                                        │  │
│  │ // 1. 获取模型配置                                                     │  │
│  │ config = engine.get_model_config()                                    │  │
│  │ layer_types = config["layer_types"]                                   │  │
│  │                                                                        │  │
│  │ // 2. 遍历每层，按类型提取                                             │  │
│  │ for layer_idx in range(num_layers):                                   │  │
│  │     if layer_types[layer_idx] == "linear_attention":                  │  │
│  │         # 提取GDN层SSM状态                                             │  │
│  │         ssm_state = engine.get_ssm_state(layer_idx, request_id)       │  │
│  │         conv_state = engine.get_conv_state(layer_idx, request_id)     │  │
│  │         ssm_states.append(ssm_state)                                   │  │
│  │     else:  # "full_attention"                                          │  │
│  │         # 提取GQA层KV Cache                                            │  │
│  │         key = engine.get_key_cache(layer_idx, request_id)             │  │
│  │         value = engine.get_value_cache(layer_idx, request_id)         │  │
│  │         kv_cache.append((key, value))                                  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                  │                                           │
│                                  ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Step 3: 构建元数据并序列化                                             │  │
│  │                                                                        │  │
│  │ metadata = Qwen35HybridKVCacheMetadata(                               │  │
│  │     num_layers=24,                                                     │  │
│  │     seq_len=seq_len,                                                   │  │
│  │     layer_type_bitmap=generate_bitmap(layer_types),                   │  │
│  │     ...                                                                 │  │
│  │ )                                                                       │  │
│  │                                                                        │  │
│  │ store_client.PutQwen35HybridKVCache(                                  │  │
│  │     request_id, ssm_states, kv_cache, metadata                        │  │
│  │ )                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**3.4.2 Decode阶段协同流程**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Decode阶段：Mooncake → 推理引擎                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Step 1: 从Mooncake获取状态                                             │  │
│  │                                                                        │  │
│  │ ssm_states, kv_cache, metadata = store_client.GetQwen35HybridKVCache( │  │
│  │     request_id                                                         │  │
│  │ )                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                  │                                           │
│                                  ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Step 2: 按层类型恢复到引擎格式                                          │  │
│  │                                                                        │  │
│  │ Qwen35KVConnector.restoreToEngine(engine, request_id, ...):            │  │
│  │                                                                        │  │
│  │ ssm_idx = 0                                                            │  │
│  │ kv_idx = 0                                                              │  │
│  │                                                                        │  │
│  │ for layer_idx in range(num_layers):                                   │  │
│  │     if metadata.getLayerType(layer_idx) == LINEAR_ATTENTION:          │  │
│  │         # 恢复GDN层SSM状态                                             │  │
│  │         engine.set_ssm_state(layer_idx, ssm_states[ssm_idx])           │  │
│  │         engine.set_conv_state(layer_idx, ssm_states[ssm_idx].conv)    │  │
│  │         ssm_idx += 1                                                    │  │
│  │     else:  # FULL_ATTENTION                                            │  │
│  │         # 恢复GQA层KV Cache                                            │  │
│  │         engine.set_key_cache(layer_idx, kv_cache[kv_idx].key)          │  │
│  │         engine.set_value_cache(layer_idx, kv_cache[kv_idx].value)      │  │
│  │         kv_idx += 1                                                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                  │                                           │
│                                  ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Step 3: 推理引擎继续Decode                                             │  │
│  │                                                                        │  │
│  │ engine继续生成下一个token:                                             │  │
│  │   - GDN层: 使用恢复的ssm_state继续递归推理                             │  │
│  │   - GQA层: 使用恢复的KV Cache进行注意力计算                            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**3.4.3 与vLLM的具体协同接口**

```python
# mooncake-integration/vllm/qwen35_kv_connector.py

class Qwen35KVConnector:
    """
    vLLM专用连接器：处理Qwen3.5混合注意力模型的状态转换
    """

    def __init__(self, store_addr: str, model_config_path: str):
        self.store_client = StoreClient(store_addr)
        self.config = self._load_model_config(model_config_path)
        self.layer_types = self.config.get("layer_types", [])

    def extract_from_vllm(
        self,
        engine: "LlamaEngine",
        request_id: str,
        seq_len: int
    ) -> Tuple[List[Tensor], List[Tuple[Tensor, Tensor]], dict]:
        """
        从vLLM引擎提取混合注意力状态

        Args:
            engine: vLLM推理引擎实例
            request_id: 请求唯一标识
            seq_len: 当前序列长度

        Returns:
            ssm_states: GDN层的SSM状态列表
            kv_cache: GQA层的KV Cache列表
            metadata: 元数据字典
        """
        ssm_states = []
        kv_cache = []

        # 获取vLLM的内部cache管理器
        cache_engine = engine.cache_engine

        for layer_idx, layer_type in enumerate(self.layer_types):
            if layer_type == "linear_attention":
                # 提取GDN层状态
                # vLLM使用MambaCache存储GDN状态
                ssm_state = cache_engine.get_ssm_state(layer_idx)
                conv_state = cache_engine.get_conv_state(layer_idx)
                ssm_states.append({
                    "ssm_state": ssm_state,  # [num_heads, head_dim, state_dim]
                    "conv_state": conv_state  # [num_heads, kernel_dim]
                })
            else:  # "full_attention"
                # 提取GQA层KV Cache
                # vLLM使用标准KVCache结构
                key = cache_engine.get_key_cache(layer_idx)
                value = cache_engine.get_value_cache(layer_idx)
                kv_cache.append((key, value))

        # 构建元数据
        metadata = self._build_metadata(seq_len)

        return ssm_states, kv_cache, metadata

    def restore_to_vllm(
        self,
        engine: "LlamaEngine",
        request_id: str,
        ssm_states: List[dict],
        kv_cache: List[Tuple[Tensor, Tensor]],
        metadata: dict
    ) -> None:
        """
        将状态恢复到vLLM引擎

        Args:
            engine: vLLM推理引擎实例
            request_id: 请求唯一标识
            ssm_states: GDN层的SSM状态列表
            kv_cache: GQA层的KV Cache列表
            metadata: 元数据字典
        """
        cache_engine = engine.cache_engine

        ssm_idx = 0
        kv_idx = 0

        for layer_idx, layer_type in enumerate(self.layer_types):
            if layer_type == "linear_attention":
                # 恢复GDN层状态
                cache_engine.set_ssm_state(
                    layer_idx,
                    ssm_states[ssm_idx]["ssm_state"]
                )
                cache_engine.set_conv_state(
                    layer_idx,
                    ssm_states[ssm_idx]["conv_state"]
                )
                ssm_idx += 1
            else:  # "full_attention"
                # 恢复GQA层KV Cache
                key, value = kv_cache[kv_idx]
                cache_engine.set_key_cache(layer_idx, key)
                cache_engine.set_value_cache(layer_idx, value)
                kv_idx += 1

    def _build_metadata(self, seq_len: int) -> dict:
        """构建Qwen35专用元数据"""
        return {
            "layout_type": KVCacheLayoutType.QWEN35_HYBRID,
            "num_layers": len(self.layer_types),
            "seq_len": seq_len,
            "layer_type_bitmap": self._generate_layer_bitmap(),
            "full_attention_interval": self.config.get("full_attention_interval", 4),
            "num_full_attention_layers": sum(1 for t in self.layer_types if t == "full_attention"),
            "num_linear_attention_layers": sum(1 for t in self.layer_types if t == "linear_attention"),
            "num_query_heads": self.config.get("num_attention_heads"),
            "num_kv_groups": self.config.get("num_key_value_heads"),
            "head_dim": self.config.get("head_dim"),
            "linear_num_key_heads": self.config.get("linear_num_key_heads", 16),
            "linear_key_head_dim": self.config.get("linear_key_head_dim", 128),
            "ssm_state_dim": self.config.get("ssm_state_dim", 256),
        }
```

**3.4.4 与SGLang的具体协同接口**

```python
# mooncake-integration/sglang/qwen35_kv_connector.py

class Qwen35SGLangConnector:
    """
    SGLang专用连接器：处理Qwen3.5混合注意力模型的状态转换

    SGLang使用不同的状态管理结构：
    - GDN状态存储在 model_runner.mamba_states
    - KV Cache存储在 model_runner.kv_cache
    """

    def extract_from_sglang(
        self,
        model_runner: "ModelRunner",
        request_id: str,
        seq_len: int
    ) -> Tuple[List[dict], List[Tuple[Tensor, Tensor]], dict]:
        """
        从SGLang引擎提取混合注意力状态

        SGLang特定逻辑:
        - SGLang的MambaState结构与vLLM略有不同
        - 需要处理SGLang的radix attention cache结构
        """
        ssm_states = []
        kv_cache = []

        # SGLang的GDN状态
        if hasattr(model_runner, 'mamba_states'):
            for layer_idx, layer_type in enumerate(self.layer_types):
                if layer_type == "linear_attention":
                    mamba_state = model_runner.mamba_states[layer_idx]
                    ssm_states.append({
                        "ssm_state": mamba_state.states,
                        "conv_state": mamba_state.conv_states,
                    })

        # SGLang的KV Cache (可能使用radix attention)
        if hasattr(model_runner, 'kv_cache'):
            kv_idx = 0
            for layer_idx, layer_type in enumerate(self.layer_types):
                if layer_type == "full_attention":
                    kv = model_runner.kv_cache[layer_idx]
                    kv_cache.append((kv.key, kv.value))
                    kv_idx += 1

        metadata = self._build_metadata(seq_len)
        return ssm_states, kv_cache, metadata
```

### 3.5 协同中的关键挑战与解决方案

| 挑战 | 描述 | 解决方案 |
|------|------|----------|
| **状态结构不一致** | 不同推理引擎的GDN/KV状态结构略有差异 | 使用Connector抽象层适配不同引擎 |
| **内存布局差异** | vLLM使用连续tensor，SGLang可能分块存储 | 在extract/restore时进行格式转换 |
| **序列长度变化** | Decode阶段序列长度动态增长 | metadata中记录seq_len，支持增量更新 |
| **并发请求管理** | 多请求同时运行，状态隔离 | 使用request_id作为唯一标识 |
| **状态生命周期** | 需要与引擎的cache生命周期同步 | 实现状态清理回调机制 |

---

## 4. 核心组件设计

### 4.1 Qwen35HybridLayoutHandler

```cpp
// mooncake-store/include/qwen35_hybrid_layout_handler.h

#pragma once

#include "kvcache_layout_handler.h"
#include <cstdint>

namespace mooncake {

// 层类型枚举
enum class LayerType : uint8_t {
    LINEAR_ATTENTION = 0,  // Gated DeltaNet
    FULL_ATTENTION = 1,    // GQA
};

// SSM状态参数（用于计算大小)
// 注意： state_dim 是GDN内部状态维度，需要从模型配置获取
// 典型值: 256 (对应Qwen3.5的state_dim配置)
static constexpr uint32_t kDefaultSSMStateDim = 256;

// Qwen3.5混合注意力专用元数据
struct Qwen35HybridKVCacheMetadata {
    // === 基础信息 ===
    KVCacheLayoutType layout_type = KVCacheLayoutType::QWEN35_HYBRID;
    uint32_t num_layers = 0;
    uint32_t seq_len = 0;
    uint32_t max_position_embeddings = 262144;

    // === 层类型配置 ===
    uint32_t full_attention_interval = 4;
    uint32_t num_full_attention_layers = 0;
    uint32_t num_linear_attention_layers = 0;

    // === 层类型位图 ===
    std::vector<uint8_t> layer_type_bitmap;

    // === GQA参数（完整注意力层）===
    uint32_t num_query_heads = 0;
    uint32_t num_kv_groups = 0;
    uint32_t head_dim = 0;

    // === GDN参数（线性注意力层）===
    uint32_t linear_key_head_dim = 128;
    uint32_t linear_value_head_dim = 128;
    uint32_t linear_num_key_heads = 16;
    uint32_t linear_num_value_heads = 16;
    uint32_t linear_conv_kernel_dim = 4;

    // === 存储偏移量 ===
    uint64_t ssm_states_offset = 0;
    uint64_t ssm_states_size = 0;
    uint64_t kv_cache_offset = 0;
    uint64_t kv_cache_size = 0;

    // === 辅助方法 ===
    LayerType getLayerType(uint32_t layer_idx) const;
    std::vector<uint32_t> getFullAttentionLayerIndices() const;
    std::vector<uint32_t> getLinearAttentionLayerIndices() const;
};
YLT_REFL(Qwen35HybridKVCacheMetadata,
         layout_type, num_layers, seq_len, max_position_embeddings,
         full_attention_interval, num_full_attention_layers, num_linear_attention_layers,
         layer_type_bitmap,
         num_query_heads, num_kv_groups, head_dim,
         linear_key_head_dim, linear_value_head_dim,
         linear_num_key_heads, linear_num_value_heads, linear_conv_kernel_dim,
         ssm_states_offset, ssm_states_size, kv_cache_offset, kv_cache_size);

    size_t calculateSerializedSize(
        const std::vector<Slice>& kv_data,
        const KVCacheMetadataBase& metadata) const override;

    ErrorCode serialize(
        const std::vector<Slice>& kv_data,
        const KVCacheMetadataBase& metadata,
        void* buffer,
        size_t buffer_size) const override;

    ErrorCode deserialize(
        const void* buffer,
        size_t buffer_size,
        std::vector<Slice>& kv_data,
        KVCacheMetadataBase& metadata) const override;

    bool validate(
        const std::vector<Slice>& kv_data,
        const KVCacheMetadataBase& metadata) const override;

    size_t calculateKVSize(const KVCacheMetadataBase& metadata) const override;

    // === Qwen3.5专用接口 ===
    size_t calculateSSMStatesSize(const Qwen35HybridKVCacheMetadata& meta) const;
    size_t calculateKVCacheSize(const Qwen35HybridKVCacheMetadata& meta) const;

    ErrorCode separateData(
        const std::vector<Slice>& combined_data,
        const Qwen35HybridKVCacheMetadata& metadata,
        std::vector<Slice>& ssm_states,
        std::vector<Slice>& kv_cache) const;

private:
    // Magic Number: Qwen3.5 Hybrid (GDN + GQA)
    // 使用更具体的命名: "Q35H" (Qwen3.5 Hybrid)
    static constexpr uint32_t kVersion = 1;
    static constexpr uint32_t kHeaderSize = 16;  // 扩展为16字节，与现有12字节对齐，便于添加元数据offset信息
};

}  // namespace mooncake
```

### 4.2 LayerTypeResolver

```cpp
// mooncake-store/include/layer_type_resolver.h

#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace mooncake {

class LayerTypeResolver {
public:
    static std::vector<uint8_t> parseFromConfig(const std::string& config_path);
    static std::vector<uint8_t> parseFromJson(const std::string& json_str);
    static std::vector<uint8_t> generateBitmap(
        uint32_t num_layers,
        uint32_t interval);
    static LayerType getLayerType(
        const std::vector<uint8_t>& bitmap,
        uint32_t layer_idx);
    static void countLayerTypes(
        const std::vector<uint8_t>& bitmap,
        uint32_t& num_linear,
        uint32_t& num_full);
};

}  // namespace mooncake
```

### 4.3 SSMStateManager

```cpp
// mooncake-store/include/ssm_state_manager.h

#pragma once

#include "types.h"
#include <vector>

namespace mooncake {

struct SSMStateLayer {
    const float* A_log;           // [num_key_heads, head_dim]
    const float* dt_proj_weight;  // [num_key_heads, hidden_dim]
    const float* dt_proj_bias;    // [num_key_heads]
    const float* conv1d_weight;   // [num_key_heads, 1, kernel_dim]
    const float* D_proj;          // [num_key_heads]
    float* state;                 // [num_key_heads, head_dim, state_dim]
    float* conv_state;            // [num_key_heads, kernel_dim]
};

class SSMStateManager {
public:
    static size_t calculateSerializedSize(
        uint32_t num_key_heads,
        uint32_t head_dim,
        uint32_t kernel_dim,
        uint32_t state_dim);

    static ErrorCode serializeLayer(
        const SSMStateLayer& layer,
        void* buffer,
        size_t buffer_size,
        uint32_t num_key_heads,
        uint32_t head_dim,
        uint32_t kernel_dim,
        uint32_t state_dim);

    static ErrorCode deserializeLayer(
        const void* buffer,
        size_t buffer_size,
        SSMStateLayer& layer,
        uint32_t num_key_heads,
        uint32_t head_dim,
        uint32_t kernel_dim,
        uint32_t state_dim);
};

}  // namespace mooncake
```

### 4.4 KVCacheLayoutType 枚举扩展

```cpp
enum class KVCacheLayoutType : int32_t {
    UNKNOWN = 0,
    MHA = 1,           // Traditional Multi-Head Attention
    GQA = 2,           // Grouped Query Attention (GLM-4, Qwen2)
    MLA = 3,           // Multi-Head Latent Attention (DeepSeek)
    HYBRID = 4,        // Sliding Window Hybrid (通用滑动窗口，用于Qwen2��模型)
    QWEN35_HYBRID = 5, // Qwen3.5 GDN+GQA Hybrid Attention [新增]
};
```

**设计说明：**
- `HYBRID = 4`：保留用于通用滑动窗口注意力模式（如Qwen2的sliding window），其元数据为 `HybridKVCacheMetadata`
- `QWEN35_HYBRID = 5`：新增用于Qwen3.5的Gated DeltaNet + GQA混合模式，其元数据为 `Qwen35HybridKVCacheMetadata`
- 两者虽然都是"混合"概念，但内部数据结构完全不同，因此需要独立的枚举值和Handler

### 4.5 建议的新序列化格式

```
┌─────────────────────────────────────────────────────────────────┐
│                    SerializedHeader (16 bytes)                   │
│  ┌────────────┬────────────┬────────────────────────────────┐  │
│  │ magic (4B) │ version(4B)│ metadata_json_length (4B)      │  │
│  │            │            │ reserved (4B)                   │  │
│  └────────────┴────────────┴────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Metadata JSON                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Part A: SSM States                            │
│  大小：固定，约 18 layers × ~2MB/layer ≈ 36MB                   │
├─────────────────────────────────────────────────────────────────┤
│                    Part B: GQA KV Cache                          │
│  大小：6 layers × 2 × KV_groups × SeqLen × HeadDim              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 数据流设计

### 5.1 Prefill 阶段数据流

```
┌────────────────────────────────────────────────────────────────┐
│  Step 1: 模型推理完成，获取原始KVCache                         │
│    输入: prompt_tokens [1, 2, 3, ..., 131072]                  │
│    输出: past_key_values (GDN层SSM状态 + GQA层KV Cache)        │
│                          ↓                                     │
│  Step 2: Qwen35KVConnector 分离数据                            │
│    - 读取 model_config.json 识别层类型                         │
│    - 分离SSM状态和KV Cache                                     │
│    - 构建Qwen35HybridKVCacheMetadata                           │
│                          ↓                                     │
│  Step 3: 调用 Mooncake Store API 存储                          │
│                          ↓                                     │
│  Step 4: Qwen35HybridLayoutHandler 序列化                      │
│    [Header][JSON][SSM States][KV Cache]                        │
│                          ↓                                     │
│  Step 5: Transfer Engine 传输到 Decode Cluster                 │
│    优先传输SSM状态（36MB）→ 再传输KV Cache（3GB）              │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 Decode 阶段数据流

```
┌────────────────────────────────────────────────────────────────┐
│  Step 1: 从 Mooncake Store 获取 KVCache                        │
│    - 可选择仅获取SSM状态优先启动                               │
│                          ↓                                     │
│  Step 2: Qwen35HybridLayoutHandler 反序列化                    │
│    - 读取Header验证magic和version                              │
│    - 解析Metadata JSON                                         │
│    - 根据offset分离SSM和KV数据                                 │
│                          ↓                                     │
│  Step 3: 恢复到推理引擎格式                                    │
│    - 根据层类型位图恢复各层数据                                │
│    - 加载到推理引擎                                            │
│                          ↓                                     │
│  Step 4: 继续Decode推理                                        │
└────────────────────────────────────────────────────────────────┘
```

---

## 6. API 设计

### 6.1 扩展 Store Client API

```cpp
class StoreClient {
public:
    // === 现有API (保持兼容) ===
    virtual int Put(const ObjectKey& key, const Slice& data) = 0;
    virtual int Get(const ObjectKey& key, std::vector<Slice>& data) = 0;

    // === 新增: Qwen3.5混合注意力专用API ===

    virtual int PutQwen35HybridKVCache(
        const ObjectKey& key,
        const std::vector<Slice>& ssm_states,
        const std::vector<Slice>& kv_cache,
        const Qwen35HybridKVCacheMetadata& metadata);

    virtual int GetQwen35HybridKVCache(
        const ObjectKey& key,
        std::vector<Slice>& ssm_states,
        std::vector<Slice>& kv_cache,
        Qwen35HybridKVCacheMetadata& metadata);

    virtual int GetPartial(
        const ObjectKey& key,
        uint64_t offset,
        uint64_t size,
        Slice& data);

    virtual int BatchGetKVCache(
        const std::vector<ObjectKey>& keys,
        std::vector<std::vector<Slice>>& data_list);
};
```

### 6.2 Error Handling Specification

```cpp
// Qwen3.5 专用错误码扩展
enum class Qwen35ErrorCode : int32_t {
    QWEN35_OK = 0,
    QWEN35_INVALID_LAYER_CONFIGURATION = 2,
    QWEN35_SSM_STATE_SIZE_MISMATCH = 3,
    QWEN35_KV_CACHE_SIZE_MISMATCH = 4,
    QWEN35_INVALID_LAYER_INDEX = 5,
    QWEN35_METADATA_VALIDATION_FAILED = 6,
    QWEN35_SERIALIZATION_FAILED = 7,
    QWEN35_DESERIALIZATION_FAILED = 8,
    QWEN35_UNSUPPORTED_LAYER_TYPE = 9,
    QWEN35_BUFFER_OVERFLOW = 10,
    QWEN35_PARTIAL_TRANSFER_FAILED = 11,
};
```

### 6.3 Integration API (vLLM/SGLang对接)

```cpp
class Qwen35KVConnector {
public:
    Qwen35KVConnector(
        const std::string& store_addr,
        const std::string& model_config_path);

    ErrorCode OffloadKVCache(
        const std::string& request_id,
        const std::vector<std::tuple<torch::Tensor, torch::Tensor>>& past_key_values,
        uint32_t seq_len);

    ErrorCode LoadKVCache(
        const std::string& request_id,
        std::vector<std::tuple<torch::Tensor, torch::Tensor>>& past_key_values);

    bool IsKVCacheReady(const std::string& request_id);
};
```

### 6.3 Python 绑定 API

```python
class Qwen35HybridKVStore:
    def __init__(self, store_addr: str, model_config_path: str): ...

    def put(self, request_id: str, past_key_values: List, seq_len: int) -> int: ...
    def get(self, request_id: str) -> Tuple[List, Qwen35HybridMetadata]: ...
    def get_partial(self, request_id: str, offset: int, size: int) -> bytes: ...
    def batch_get(self, request_ids: List[str]) -> List: ...
```

### 6.4 传输优化策略

| 策略 | 描述 | 适用场景 | 预估延迟 (128K) |
|------|------|----------|-----------------|
| SEQUENTIAL | 先传SSM→再传KV | 单NIC环境 | ~560ms |
| PARALLEL | 多NIC同时传输不同层 | 多NIC高带宽 | ~180ms |
| PRIORITIZED | SSM高优先，KV后台传 | 需快速启动 | SSM: ~10ms |
| ADAPTIVE | 动态选择最优策略 | 变化网络环境 | 动态 |

---

## 7. 测试策略

### 7.1 单元测试

| 测试场景 | 测试要点 |
|----------|----------|
| calculateSSMStatesSize | SSM状态大小计算正确性 |
| calculateKVCacheSize | KV Cache大小计算正确性 |
| SerializeDeserialize | 序列化/反序列化数据一致性 |
| LayerTypeBitmapParsing | 层类型位图解析正确性 |
| InvalidMetadata | 无效元数据验证 |
| LargeSequenceLength | 大序列长度场景 |
| DataSeparation | 数据分离正确性 |

### 7.2 集成测试

| 测试场景 | 测试要点 |
|----------|----------|
| EndToEndPutGet | 端到端存取流程 |
| PrefillToDecodeTransfer | Prefill到Decode完整传输 |
| MultipleConcurrentRequests | 多并发请求处理 |

### 7.3 性能基准测试

| 测试���景 | 序列长度 | 测试要点 | 基线比较 |
|----------|----------|----------|-----------------|
| SerializationThroughput | 4K/32K/128K/262K | 序列化吞吐量 | 与现有GQA Handler对比 |
| TransferLatency | 128K | 不同传输策略延迟 | 与传统GQA传输对比 |
| MemoryUsage | 256K | 内存使用效率 | 与传统GQA内存对比 |

---

## 8. 实现路线图

### Phase 1: 核心Layout Handler实现 (1-2周)

| 任务 | 交付物 |
|------|--------|
| Qwen35HybridKVCacheMetadata 结构定义 | qwen35_hybrid_layout_handler.h |
| Qwen35HybridLayoutHandler 基本框架 | qwen35_hybrid_layout_handler.cpp |
| LayerTypeResolver 实现 | layer_type_resolver.h/cpp |
| SSMStateManager 实现 | ssm_state_manager.h/cpp |
| 单元测试 | qwen35_hybrid_layout_handler_test.cpp |

### Phase 2: Store API扩展 (1周)

| 任务 | 交付物 |
|------|--------|
| PutQwen35HybridKVCache 实现 | client_api.cpp (扩展) |
| GetQwen35HybridKVCache 实现 | client_api.cpp (扩展) |
| GetPartial 实现 | client_api.cpp (扩展) |
| KVCacheLayoutHandlerFactory 扩展 | layout_handler_factory.cpp |
| 集成测试 | qwen35_hybrid_integration_test.cpp |

### Phase 3: 传输优化实现 (1周)

| 任务 | 交付物 |
|------|--------|
| Qwen35TransferOptimizer 实现 | transfer_optimizer.h/cpp |
| 优先传输策略实现 | transfer_optimizer.cpp |
| 多路径并行传输支持 | transfer_optimizer.cpp |
| 性能基准测试 | qwen35_hybrid_benchmark.cpp |

### Phase 4: Python绑定与vLLM集成 (1-2周)

| 任务 | 交付物 |
|------|--------|
| Python绑定实现 | qwen35_hybrid.py |
| Qwen35KVConnector 实现 | qwen35_kv_connector.h/cpp |
| vLLM集成测试 | vllm_integration_test.cpp |
| 端到端性能测试 | e2e_performance_test.cpp |

### Phase 5: 文档与性能验证 (1周)

| 任务 | 交付物 |
|------|--------|
| API文档 | docs/api/qwen35_hybrid.md |
| 使用指南 | docs/guides/qwen35_integration.md |
| 性能报告 | docs/benchmarks/qwen35_performance.md |
| 最佳实践 | docs/best_practices/qwen35_offloading.md |

---

## 9. 预期收益与风险评估

### 9.1 预期性能收益

| 指标 | 优化前(传统GQA) | 优化后(Qwen3.5) | 提升 |
|------|----------------|-----------------|------|
| KVCache大小 (128K) | ~12GB | ~3GB | **75%减少** |
| 传输时间 (50Gbps) | ~2s | ~500ms | **4x加速** |
| 存储容量 | 100请求/节点 | 400请求/节点 | **4x提升** |
| 首token延迟 | ~2s | ~10ms(SSM) | **200x改善** |

### 9.2 文件变更清单

| 文件路径 | 变更类型 |
|----------|----------|
| mooncake-store/include/kvcache_layout.h | 修改 |
| mooncake-store/include/qwen35_hybrid_layout_handler.h | **新增** |
| mooncake-store/src/qwen35_hybrid_layout_handler.cpp | **新增** |
| mooncake-store/include/layer_type_resolver.h | **新增** |
| mooncake-store/src/layer_type_resolver.cpp | **新增** |
| mooncake-store/include/ssm_state_manager.h | **新增** |
| mooncake-store/src/ssm_state_manager.cpp | **新增** |
| mooncake-store/include/transfer_optimizer.h | **新增** |
| mooncake-store/src/transfer_optimizer.cpp | **新增** |
| mooncake-store/include/store_client.h | 修改 |
| mooncake-store/src/client_api.cpp | 修改 |
| mooncake-store/tests/qwen35_hybrid_*.cpp | **新增** |
| mooncake-wheel/mooncake/qwen35_hybrid.py | **新增** |
| mooncake-integration/vllm/qwen35_kv_connector.* | **新增** |
| mooncake-store/src/CMakeLists.txt | 修改 (添加新文件) |
| mooncake-store/tests/CMakeLists.txt | 修改 (添加测试文件) |

### 9.3 风险与缓解措施

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| GDN层SSM状态格式变化 | 需要更新Handler | 版本化元数据，支持多版本 |
| 大序列内存压力 | 可能OOM | 流式传输，分块处理 |
| 与现有集成冲突 | 兼容性问题 | 保留原有Handler，新增专用Handler |

---

## 10. 参考资料

- [Qwen3.5 混合注意力架构全解析 - 知乎专栏](https://zhuanlan.zhihu.com/p/2013625870263273172)
- [Qwen3.5 Official Blog](https://qwen.ai/blog?id=qwen3.5)
- [NVIDIA Developer - Qwen3-Next Architecture](https://developer.nvidia.com/blog/new-open-source-qwen3-next-models-preview-hybrid-moe-architecture-delivering-improved-accuracy-and-accelerated-parallel-processing-across-nvidia-platforms/)
- [Gated Delta Networks 论文](https://arxiv.org/abs/2405.04434)
- [Mooncake KVCache架构优化设计](../superpowers/specs/2026-03-10-kvcache-architecture-optimization-design.md)

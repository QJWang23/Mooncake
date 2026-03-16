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
    }

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

### 6.2 Integration API (vLLM/SGLang对接)

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

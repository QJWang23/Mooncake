# KVCache Architecture-Specific Optimization Design for Mooncake

**Date**: 2026-03-10
**Author**: Claude (AI Assistant)
**Status**: Approved

## Executive Summary

This document analyzes how different LLM model architectures impact KVCache management patterns and offload strategies, and proposes optimization recommendations for Mooncake to support GLM-4/5 (GQA), DeepSeek V3.2 (MLA/DSA), and Qwen2.5 (Hybrid Attention) architectures.

---

## 1. Model Architecture Analysis

### 1.1 GLM-4/5 Architecture (Zhipu AI)

**Core Principles:**
- **GQA (Grouped Query Attention)**: Replaces traditional Multi-Head Attention (MHA)
- Query heads are grouped, with each group sharing a set of Key-Value heads
- GLM-4 reallocates saved parameters to expand FFN to 10/3 of hidden size

**KVCache Characteristics:**
- KV cache size proportional to KV head group count
- Significantly reduced compared to MHA
- Supports 128K long context

**Advantages:**
- Improved inference throughput
- Optimized memory efficiency
- Reduced VRAM usage while maintaining model performance

---

### 1.2 DeepSeek V3.2 DSA/MLA Architecture

**Core Principles:**

**MLA (Multi-Head Latent Attention):**
- **Low-rank KV joint compression**: Compresses K/V tensors into low-dimensional latent space for storage
- **On-demand decompression**: Reconstructs full multi-head K/V during attention computation
- Implements "space for time" optimization strategy

**DSA (DeepSeek Sparse Attention) - New in V3.2:**
- **Lightning Indexer**: Dynamically selects relevant historical tokens
- **Two-stage indexer**: Top-k selection reduces quadratic complexity
- **Local Window Attention**: Each token only attends to neighbors within a fixed window

**KVCache Characteristics:**
- MLA achieves **4-8x memory reduction**
- DeepSeek-V2's KV cache reduced by **93.3%** compared to dense model
- Compressed latent vectors stored instead of full K/V

**Advantages:**
- Breaks inference "memory wall" limitation
- Efficient long context support
- Dual optimization for training and inference

---

### 1.3 Qwen2.5 Hybrid Attention Architecture

**Core Principles:**
- **GQA as foundation**: Similar to GLM, uses grouped query attention
- **Hybrid attention strategy**: Combines global attention with local/sparse attention
- **Sliding window mechanism**: Limits KV cache to fixed window size

**KVCache Characteristics:**
- Reduced KV heads count, correspondingly smaller cache
- Qwen2.5-72B KV Cache uses approximately **1.01 GB** (0.7% of total VRAM)
- Recent research shows conversion to pseudo-MLA for further optimization

**Advantages:**
- Improved inference throughput
- Good compatibility with existing frameworks (vLLM)
- Ultra-long context support (Qwen2.5-1M supports 1M tokens)

---

## 2. Impact Analysis on KVCache Management

### 2.1 GLM-4/5 GQA Architecture Impact

**Storage Structure Change:**
```
Traditional MHA:  [Layer × Head × SeqLen × HeadDim] × 2 (K+V)
GQA:              [Layer × KV_Groups × SeqLen × HeadDim] × 2
```

| Dimension | MHA | GQA (GLM-4) | Difference |
|-----------|-----|-------------|------------|
| KV Heads | 32-128 | 2-8 | 90%+ reduction |
| Memory Usage | Baseline | 10-25% | Significant reduction |
| Transfer Data | Baseline | 10-25% | Network bandwidth savings |

**Impact on Mooncake:**
- **Beneficial**: Smaller KVCache per request, more concurrent requests storable
- **Challenge**: GQA group structure needs storage-layer awareness for correct restoration
- **Offload Strategy**: Can directly offload compressed KV without additional processing

---

### 2.2 DeepSeek V3.2 MLA/DSA Architecture Impact

**Storage Structure Change:**
```
Traditional MHA:  [Layer × Head × SeqLen × HeadDim] × 2 (K+V)
MLA:              [Layer × SeqLen × LatentDim] × 1 (compressed latent vector)
```

| Dimension | MHA | MLA (DS-V3) | Difference |
|-----------|-----|-------------|------------|
| Storage Dimension | Multi-head separated | Single latent vector | Completely different structure |
| Memory Usage | Baseline | 6-12% | 88-94% reduction |
| Decompression Overhead | None | Required at compute time | Increased CPU/GPU compute |

**DSA Additional Impact:**
- **Sparse Index**: Needs to store Lightning Indexer index information
- **Local Window**: Only store KV within window, beyond window can be discarded
- **Dynamic Selection**: KVCache needs random access pattern support

**Impact on Mooncake:**
- **Major Challenge**: MLA latent vector format completely different from traditional KVCache
- **Storage Optimization**: Smaller individual data, but needs new serialization format
- **Transfer Optimization**: Less data transfer, but needs to preserve metadata for decompression
- **Offload Strategy**: Can directly store compressed latent vectors, but needs additional metadata management

---

### 2.3 Qwen2.5 Hybrid Attention Architecture Impact

**Storage Structure Change:**
```
Traditional MHA:     [Layer × Head × SeqLen × HeadDim] × 2 (K+V)
GQA + Sliding:       [Layer × KV_Groups × WindowSize × HeadDim] × 2
```

| Dimension | MHA | GQA + Sliding | Difference |
|-----------|-----|---------------|------------|
| Sequence Length | Full | Window limited | Sliding discards old data |
| KV Heads | Full | Group shared | Reduced |
| Memory Usage | O(L) | O(W) | Constant level |

**Impact on Mooncake:**
- **Beneficial**: Sliding window naturally supports FIFO offload strategy
- **Challenge**: Needs partial update support (new tokens in, old tokens out)
- **Offload Strategy**:
  - Within window: Keep in memory
  - Outside window: Can discard or offload to cold storage (for backtracking)

---

### 2.4 Comparison Summary

| Feature | GLM-4 GQA | DS-V3.2 MLA/DSA | Qwen2.5 Hybrid |
|---------|-----------|-----------------|----------------|
| KVCache Size | Small (25%) | Extremely small (6-12%) | Small + Window limit |
| Storage Format Change | Grouped structure | Latent vector format | Grouped + Window |
| Random Access Need | Low | Medium (index) | High (sparse) |
| Existing System Compatibility | High | Low (needs rewrite) | Medium |
| Offload Complexity | Low | Medium | Medium-High |

---

## 3. Mooncake Optimization Recommendations

### 3.1 Generic Architecture Abstraction Layer Design

**Goal**: Enable Mooncake to support multiple model architectures while maintaining high performance.

**Recommended Abstraction Layer Structure:**

```
┌─────────────────────────────────────────────────────┐
│              Model-Aware KVCache Layer              │
├─────────────┬─────────────┬─────────────────────────┤
│  GQA Layout │ MLA Layout  │  Hybrid/Sparse Layout   │
│   Handler   │  Handler    │       Handler           │
├─────────────┴─────────────┴─────────────────────────┤
│              Unified Storage Interface               │
│         (Slice + Metadata + Compression)            │
├─────────────────────────────────────────────────────┤
│              Mooncake Store / Transfer Engine       │
└─────────────────────────────────────────────────────┘
```

**Core Components:**

| Component | Responsibility | Implementation Points |
|-----------|---------------|----------------------|
| Layout Handler | Handle different architecture data layouts | Pluggable, loaded by model type |
| Metadata Manager | Store KV grouping/compression info | Lightweight key-value pairs, stored separately from data |
| Compression Adapter | Transparent compression/decompression | MLA latent vectors need no additional compression |

---

### 3.2 GQA (GLM-4/Qwen) Specific Optimizations

**A. Storage Efficiency Optimization**

```cpp
// Recommendation: GQA-aware storage layout
struct GQAKVCacheLayout {
    uint32_t num_query_heads;    // Q head count
    uint32_t num_kv_groups;      // KV group count
    uint32_t group_size;         // Q heads per group
    // Store only KV groups, not all heads
};
```

**Optimization Strategies:**
- **Group-aware storage**: Only store `num_kv_groups` KV copies, not `num_query_heads`
- **Layout metadata**: Store group mapping relationships, support different GQA configurations

**B. Transfer Optimization**
- Leverage small GQA data size for **batch transfer of multiple requests' KVCache**
- Merge by KV group during RDMA writes, reduce small data transfer count

**C. Offload Strategy**
- **Hot Tier (DRAM)**: Complete KVCache for active requests
- **Warm Tier (SSD)**: KVCache waiting for decode after prefill complete
- **Cold Tier**: Optional, for early token backup in long context

---

### 3.3 MLA (DeepSeek V3.2) Specific Optimizations

**A. Storage Efficiency Optimization**

```cpp
// Recommendation: MLA-aware storage layout
struct MLAKVCacheLayout {
    uint32_t latent_dim;         // Latent vector dimension (much smaller than head × head_dim)
    bool has_rope_embedding;     // Whether contains RoPE embedding
    // Latent vectors already compressed, no additional compression needed
};
```

**Optimization Strategies:**
- **Direct latent vector storage**: MLA already compressed, no secondary compression needed
- **Metadata separation**: Latent vectors and decompression parameters stored separately
- **DSA index storage**: Dedicated index storage for Lightning Indexer

**B. Transfer Optimization**
- **Ultra-low bandwidth requirement**: MLA data extremely small, fully utilize network bandwidth
- **Batch aggregated transfer**: Aggregate latent vectors from multiple layers for single transfer
- **Index-first transfer**: In DSA mode, transfer index first, pull KV on demand

**C. Offload Strategy**
```
┌────────────────────────────────────────────────────┐
│                    DRAM                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│  │ Latent  │  │ Indexer │  │ Active  │           │
│  │ Vectors │  │ Metadata│  │ Window  │           │
│  └────┬────┘  └────┬────┘  └─────────┘           │
└───────┼────────────┼──────────────────────────────┘
        │            │
        ▼            ▼
┌────────────────────────────────────────────────────┐
│                    SSD                              │
│  ┌─────────────────────────────────────────────┐  │
│  │  Historical Latent Vectors (optional)        │  │
│  │  - For long context backtracking             │  │
│  │  - DSA can re-index on demand                │  │
│  └─────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
```

---

### 3.4 Hybrid Attention (Qwen) Specific Optimizations

**A. Storage Efficiency Optimization**

```cpp
// Recommendation: Sliding window-aware storage
struct SlidingWindowLayout {
    uint32_t window_size;        // Window size
    uint32_t current_start;      // Current window start position
    bool supports_random_access; // Whether supports random access (sparse mode)
};
```

**Optimization Strategies:**
- **Ring buffer**: Sliding window naturally fits ring storage, reduces memory fragmentation
- **Partial update API**: Support updating only new tokens within window, not full replacement
- **Sparse index**: Global attention tokens in hybrid attention need additional indexing

**B. Transfer Optimization**
- **Incremental transfer**: Only transfer new KV entering window, not full
- **Differential update**: Support `PartialPut` operation, update partial KVCache

**C. Offload Strategy**
```
Timeline:  [---- Outside Window (offload/discard) ----][Inside Window (hot data)]
                                                       ↑
                                                  Sliding window boundary
```
- **Inside window**: Always keep in DRAM
- **Outside window**:
  - Short context: Directly discard
  - Long context: Offload to SSD, support backtracking

---

### 3.5 Unified API Design Recommendations

**Extending Existing Mooncake Store API:**

```cpp
// New: Architecture-aware KVCache operations
class ModelAwareKVStore {
public:
    // Basic operations (maintain compatibility)
    virtual int Put(const ObjectKey& key, const Slice& data) = 0;
    virtual int Get(const ObjectKey& key, Slice& data) = 0;

    // New: Architecture-aware operations
    virtual int PutKVCache(
        const ObjectKey& key,
        const Slice& kv_data,
        const KVCacheLayout& layout,      // Describe data layout
        const KVCacheMetadata& meta       // Grouping/compression metadata
    ) = 0;

    // New: Incremental update (sliding window)
    virtual int PartialUpdate(
        const ObjectKey& key,
        uint64_t offset,
        const Slice& new_data
    ) = 0;

    // New: Fetch by index (DSA sparse access)
    virtual int GetByIndices(
        const ObjectKey& key,
        const std::vector<uint64_t>& indices,
        std::vector<Slice>& results
    ) = 0;
};
```

---

## 4. Implementation Roadmap

### 4.1 Implementation Priority

**Phase 1: GQA Support First** (Short-term)
- Reason: Both GLM-4 and Qwen2.5 use GQA, broad coverage
- Change volume: Medium, mainly adding group-aware storage logic
- Expected benefit: Immediate support for two mainstream models

**Phase 2: MLA Support** (Medium-term)
- Reason: DeepSeek series growing rapidly, MLA technology leading
- Change volume: Large, needs new serialization format and metadata management
- Expected benefit: Support DeepSeek V3.x series, maximize memory efficiency

**Phase 3: Advanced Features** (Long-term)
- Sliding window incremental update
- DSA sparse index support
- Adaptive tiered offload

---

### 4.2 Key Code Change Points

| Module | Change Content | Priority |
|--------|----------------|----------|
| `mooncake-store/include` | Add `KVCacheLayout` abstract class | P0 |
| `mooncake-store/src/client` | Implement architecture-aware Put/Get | P0 |
| `mooncake-transfer-engine` | Support incremental transfer | P1 |
| `mooncake-integration/vllm` | GQA/MLA-aware Connector | P1 |
| `mooncake-integration/sglang` | Hybrid attention support | P2 |

---

## 5. Summary

This design provides Mooncake with optimization recommendations to support multiple LLM architectures:

| Dimension | GLM-4 GQA | DS-V3.2 MLA | Qwen2.5 Hybrid |
|-----------|-----------|-------------|----------------|
| **Storage Optimization** | Group-aware storage | Direct latent storage | Ring buffer |
| **Transfer Optimization** | Batch aggregation | Ultra-low bandwidth utilization | Incremental transfer |
| **Offload Strategy** | Three-tier grading | Index-first | Window-aware |
| **Implementation Difficulty** | Low | Medium | Medium |
| **Priority** | P0 | P1 | P1 |

---

## References

- [DeepSeek Official Docs - DSA Introduction](https://api-docs.deepseek.com/news/news250929)
- [DeepSeek-V2 arXiv Paper](https://arxiv.org/pdf/2405.04434)
- [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2512.02556)
- [MLA Explanation - HuggingFace](https://huggingface.co/blog/NormalUhr/mla-explanation)
- [Technical DeepSeek V3→V3.2 Analysis](https://magazine.sebastianraschka.com/p/technical-deepseek)
- [Qwen2.5-1M Technical Report](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/Qwen_5_1M_Technical_Report.pdf)
- [GLM4 Technical Points](https://saicat.github.io/a5206abd.html)
- [TransMLA: Improve Qwen2.5 - Kaitchup](https://kaitchup.substack.com/p/transmla-improve-qwen25-and-llama)
- [KV Cache Size Calculations in GQA - Medium](https://medium.com/@liu.peng.uppsala/key-value-kv-cache-size-calculations-in-grouped-query-attention-gqa-e090d3037ab3)
- [MLA Implementation - LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/05_mla/README.md)

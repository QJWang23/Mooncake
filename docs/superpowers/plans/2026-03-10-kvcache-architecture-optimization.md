# KVCache Architecture-Specific Optimization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable Mooncake Store to support multiple LLM model architectures (GQA, MLA, Hybrid Attention) with architecture-aware KVCache storage, transfer, and offload optimizations.

**Architecture:** Add a pluggable layout handler layer between the Client API and storage backend. Layout handlers serialize/deserialize architecture-specific KVCache formats while preserving metadata for efficient transfer and offload. The design maintains backward compatibility with existing APIs.

**Tech Stack:** C++17, ylt/struct_json for serialization, existing Mooncake Store infrastructure (TransferEngine, MasterClient, StorageBackend)

**Spec Document:** `docs/superpowers/specs/2026-03-10-kvcache-architecture-optimization-design.md`

---

## Phase 1: Core Abstraction Layer (P0)

### Task 1: Define KVCache Layout Types and Metadata Structures

**Files:**
- Create: `mooncake-store/include/kvcache_layout.h`
- Test: `mooncake-store/tests/kvcache_layout_test.cpp`

- [ ] **Step 1: Write the failing test for KVCacheLayoutType enum**

```cpp
// In kvcache_layout_test.cpp
#include "kvcache_layout.h"
#include <gtest/gtest.h>

TEST(KVCacheLayoutTest, LayoutTypeToString) {
    EXPECT_EQ(toString(KVCacheLayoutType::MHA), "MHA");
    EXPECT_EQ(toString(KVCacheLayoutType::GQA), "GQA");
    EXPECT_EQ(toString(KVCacheLayoutType::MLA), "MLA");
    EXPECT_EQ(toString(KVCacheLayoutType::HYBRID), "HYBRID");
}

TEST(KVCacheLayoutTest, LayoutTypeFromString) {
    EXPECT_EQ(fromString("GQA"), KVCacheLayoutType::GQA);
    EXPECT_EQ(fromString("MLA"), KVCacheLayoutType::MLA);
    EXPECT_EQ(fromString("unknown"), KVCacheLayoutType::UNKNOWN);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd build && cmake -DBUILD_UNIT_TESTS=ON .. && make -j && ctest -R kvcache_layout_test -V`
Expected: Test fails (file doesn't exist)

- [ ] **Step 3: Create kvcache_layout.h with KVCacheLayoutType enum**

```cpp
#pragma once

#include <string>
#include <cstdint>
#include "ylt/struct_json/json_reader.h"
#include "ylt/struct_json/json_writer.h"

namespace mooncake {

enum class KVCacheLayoutType : int32_t {
    UNKNOWN = 0,
    MHA = 1,      // Traditional Multi-Head Attention
    GQA = 2,      // Grouped Query Attention (GLM-4, Qwen)
    MLA = 3,      // Multi-Head Latent Attention (DeepSeek)
    HYBRID = 4,   // Hybrid/Sliding Window Attention
};

inline std::string toString(KVCacheLayoutType type) {
    switch (type) {
        case KVCacheLayoutType::MHA: return "MHA";
        case KVCacheLayoutType::GQA: return "GQA";
        case KVCacheLayoutType::MLA: return "MLA";
        case KVCacheLayoutType::HYBRID: return "HYBRID";
        default: return "UNKNOWN";
    }
}

inline KVCacheLayoutType fromString(const std::string& str) {
    if (str == "MHA") return KVCacheLayoutType::MHA;
    if (str == "GQA") return KVCacheLayoutType::GQA;
    if (str == "MLA") return KVCacheLayoutType::MLA;
    if (str == "HYBRID") return KVCacheLayoutType::HYBRID;
    return KVCacheLayoutType::UNKNOWN;
}

}  // namespace mooncake
```

- [ ] **Step 4: Add kvcache_layout.h to CMakeLists.txt**

Modify: `mooncake-store/CMakeLists.txt` (add to headers list)

- [ ] **Step 5: Run test to verify it passes**

Run: `cd build && make -j && ctest -R kvcache_layout_test -V`
Expected: Tests pass

- [ ] **Step 6: Commit**

Run: `git add -A && git commit -m "feat(store): add KVCacheLayoutType enum for architecture awareness"`

---

### Task 2: Define KVCacheMetadata Structure

**Files:**
- Modify: `mooncake-store/include/kvcache_layout.h`
- Modify: `mooncake-store/include/types.h` (if needed for include)
- Test: `mooncake-store/tests/kvcache_layout_test.cpp`

- [ ] **Step 1: Write the failing test for KVCacheMetadata serialization**

```cpp
// Add to kvcache_layout_test.cpp
TEST(KVCacheLayoutTest, GQAMetadataSerialization) {
    GQAKVCacheMetadata meta;
    meta.num_query_heads = 32;
    meta.num_kv_groups = 4;
    meta.group_size = 8;
    meta.head_dim = 128;

    std::string json;
    ylt::struct_json::to_json(meta, json);

    GQAKVCacheMetadata parsed;
    ylt::struct_json::from_json(parsed, json);

    EXPECT_EQ(parsed.num_query_heads, 32);
    EXPECT_EQ(parsed.num_kv_groups, 4);
    EXPECT_EQ(parsed.group_size, 8);
    EXPECT_EQ(parsed.head_dim, 128);
}

TEST(KVCacheLayoutTest, MLAMetadataSerialization) {
    MLAKVCacheMetadata meta;
    meta.latent_dim = 512;
    meta.has_rope_embedding = true;
    meta.num_heads = 32;
    meta.head_dim = 128;

    std::string json;
    ylt::struct_json::to_json(meta, json);

    MLAKVCacheMetadata parsed;
    ylt::struct_json::from_json(parsed, json);

    EXPECT_EQ(parsed.latent_dim, 512);
    EXPECT_TRUE(parsed.has_rope_embedding);
}

TEST(KVCacheLayoutTest, HybridMetadataSerialization) {
    HybridKVCacheMetadata meta;
    meta.window_size = 4096;
    meta.current_start = 100;
    meta.supports_random_access = true;
    meta.num_kv_groups = 8;

    std::string json;
    ylt::struct_json::to_json(meta, json);

    HybridKVCacheMetadata parsed;
    ylt::struct_json::from_json(parsed, json);

    EXPECT_EQ(parsed.window_size, 4096);
    EXPECT_TRUE(parsed.supports_random_access);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd build && make -j && ctest -R kvcache_layout_test -V`
Expected: Test fails (structs not defined)

- [ ] **Step 3: Add metadata structures to kvcache_layout.h**

```cpp
// Add to kvcache_layout.h after KVCacheLayoutType

// Base metadata structure
struct KVCacheMetadataBase {
    KVCacheLayoutType layout_type = KVCacheLayoutType::UNKNOWN;
    uint32_t num_layers = 0;
    uint32_t seq_len = 0;
};
YLT_REFL(KVCacheMetadataBase, layout_type, num_layers, seq_len);

// GQA-specific metadata (GLM-4, Qwen)
struct GQAKVCacheMetadata {
    KVCacheLayoutType layout_type = KVCacheLayoutType::GQA;
    uint32_t num_layers = 0;
    uint32_t seq_len = 0;
    uint32_t num_query_heads = 0;    // Total query heads
    uint32_t num_kv_groups = 0;      // Number of KV groups
    uint32_t group_size = 0;         // Query heads per group
    uint32_t head_dim = 0;           // Dimension per head
};
YLT_REFL(GQAKVCacheMetadata, layout_type, num_layers, seq_len,
         num_query_heads, num_kv_groups, group_size, head_dim);

// MLA-specific metadata (DeepSeek)
struct MLAKVCacheMetadata {
    KVCacheLayoutType layout_type = KVCacheLayoutType::MLA;
    uint32_t num_layers = 0;
    uint32_t seq_len = 0;
    uint32_t latent_dim = 0;         // Compressed latent dimension
    uint32_t num_heads = 0;          // Original number of heads
    uint32_t head_dim = 0;           // Original head dimension
    bool has_rope_embedding = false; // Whether RoPE is included
};
YLT_REFL(MLAKVCacheMetadata, layout_type, num_layers, seq_len,
         latent_dim, num_heads, head_dim, has_rope_embedding);

// Hybrid/Sliding Window metadata (Qwen with window)
struct HybridKVCacheMetadata {
    KVCacheLayoutType layout_type = KVCacheLayoutType::HYBRID;
    uint32_t num_layers = 0;
    uint32_t seq_len = 0;
    uint32_t window_size = 0;        // Sliding window size
    uint32_t current_start = 0;      // Current window start position
    uint32_t num_kv_groups = 0;      // For GQA component
    uint32_t head_dim = 0;
    bool supports_random_access = false; // For sparse access
};
YLT_REFL(HybridKVCacheMetadata, layout_type, num_layers, seq_len,
         window_size, current_start, num_kv_groups, head_dim, supports_random_access);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd build && make -j && ctest -R kvcache_layout_test -V`
Expected: Tests pass

- [ ] **Step 5: Commit**

Run: `git add -A && git commit -m "feat(store): add GQA/MLA/Hybrid KVCache metadata structures"`

---

### Task 3: Define KVCacheLayout Abstract Interface

**Files:**
- Create: `mooncake-store/include/kvcache_layout_handler.h`
- Test: `mooncake-store/tests/kvcache_layout_handler_test.cpp`

- [ ] **Step 1: Write the failing test for layout handler interface**

```cpp
// In kvcache_layout_handler_test.cpp
#include "kvcache_layout_handler.h"
#include <gtest/gtest.h>

class MockLayoutHandler : public KVCacheLayoutHandler {
public:
    KVCacheLayoutType getType() const override {
        return KVCacheLayoutType::GQA;
    }

    size_t calculateSerializedSize(const std::vector<Slice>& kv_data,
                                   const KVCacheMetadataBase& metadata) const override {
        return 1024;  // Mock implementation
    }

    ErrorCode serialize(const std::vector<Slice>& kv_data,
                       const KVCacheMetadataBase& metadata,
                       void* buffer,
                       size_t buffer_size) const override {
        return ErrorCode::OK;
    }

    ErrorCode deserialize(const void* buffer,
                         size_t buffer_size,
                         std::vector<Slice>& kv_data,
                         KVCacheMetadataBase& metadata) const override {
        return ErrorCode::OK;
    }
};

TEST(KVCacheLayoutHandlerTest, MockHandlerReturnsCorrectType) {
    MockLayoutHandler handler;
    EXPECT_EQ(handler.getType(), KVCacheLayoutType::GQA);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd build && make -j && ctest -R kvcache_layout_handler_test -V`
Expected: Test fails

- [ ] **Step 3: Create kvcache_layout_handler.h with abstract interface**

```cpp
#pragma once

#include "kvcache_layout.h"
#include "types.h"
#include <memory>
#include <vector>

namespace mooncake {

/**
 * @brief Abstract interface for handling different KVCache layouts
 *
 * Layout handlers are responsible for:
 * - Serializing architecture-specific KVCache data
 * - Deserializing stored data back to architecture-specific format
 * - Calculating storage requirements
 * - Validating data integrity
 */
class KVCacheLayoutHandler {
public:
    virtual ~KVCacheLayoutHandler() = default;

    /**
     * @brief Get the layout type this handler supports
     */
    virtual KVCacheLayoutType getType() const = 0;

    /**
     * @brief Calculate the serialized size for given KVCache data
     * @param kv_data Vector of slices containing KV cache data
     * @param metadata Architecture-specific metadata
     * @return Size in bytes needed for serialization
     */
    virtual size_t calculateSerializedSize(
        const std::vector<Slice>& kv_data,
        const KVCacheMetadataBase& metadata) const = 0;

    /**
     * @brief Serialize KVCache data to a buffer
     * @param kv_data Vector of slices containing KV cache data
     * @param metadata Architecture-specific metadata
     * @param buffer Destination buffer
     * @param buffer_size Size of destination buffer
     * @return ErrorCode::OK on success
     */
    virtual ErrorCode serialize(
        const std::vector<Slice>& kv_data,
        const KVCacheMetadataBase& metadata,
        void* buffer,
        size_t buffer_size) const = 0;

    /**
     * @brief Deserialize KVCache data from a buffer
     * @param buffer Source buffer
     * @param buffer_size Size of source buffer
     * @param kv_data Output vector of slices (caller manages memory)
     * @param metadata Output architecture-specific metadata
     * @return ErrorCode::OK on success
     */
    virtual ErrorCode deserialize(
        const void* buffer,
        size_t buffer_size,
        std::vector<Slice>& kv_data,
        KVCacheMetadataBase& metadata) const = 0;

    /**
     * @brief Validate that the data matches the expected layout
     * @param kv_data Vector of slices to validate
     * @param metadata Expected metadata
     * @return true if valid, false otherwise
     */
    virtual bool validate(
        const std::vector<Slice>& kv_data,
        const KVCacheMetadataBase& metadata) const {
        return true;  // Default implementation
    }
};

/**
 * @brief Factory for creating layout handlers
 */
class KVCacheLayoutHandlerFactory {
public:
    static std::unique_ptr<KVCacheLayoutHandler> create(KVCacheLayoutType type);
};

}  // namespace mooncake
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd build && make -j && ctest -R kvcache_layout_handler_test -V`
Expected: Tests pass

- [ ] **Step 5: Commit**

Run: `git add -A && git commit -m "feat(store): add KVCacheLayoutHandler abstract interface"`

---

### Task 4: Implement GQA Layout Handler

**Files:**
- Create: `mooncake-store/include/gqa_layout_handler.h`
- Create: `mooncake-store/src/gqa_layout_handler.cpp`
- Modify: `mooncake-store/CMakeLists.txt`
- Test: `mooncake-store/tests/gqa_layout_handler_test.cpp`

- [ ] **Step 1: Write the failing test for GQA layout handler**

```cpp
// In gqa_layout_handler_test.cpp
#include "gqa_layout_handler.h"
#include <gtest/gtest.h>
#include <cstring>

class GQALayoutHandlerTest : public ::testing::Test {
protected:
    void SetUp() override {
        handler_ = std::make_unique<GQALayoutHandler>();
    }

    std::unique_ptr<GQALayoutHandler> handler_;
};

TEST_F(GQALayoutHandlerTest, ReturnsCorrectType) {
    EXPECT_EQ(handler_->getType(), KVCacheLayoutType::GQA);
}

TEST_F(GQALayoutHandlerTest, CalculateSize) {
    GQAKVCacheMetadata meta;
    meta.num_layers = 32;
    meta.num_kv_groups = 4;
    meta.head_dim = 128;
    meta.seq_len = 1024;

    // K + V, each layer has num_kv_groups * seq_len * head_dim
    // Total = 2 * num_layers * num_kv_groups * seq_len * head_dim
    size_t expected = 2 * 32 * 4 * 1024 * 128;
    EXPECT_EQ(handler_->calculateKVSize(meta), expected);
}

TEST_F(GQALayoutHandlerTest, SerializeDeserialize) {
    GQAKVCacheMetadata meta;
    meta.num_layers = 2;
    meta.num_kv_groups = 2;
    meta.head_dim = 64;
    meta.seq_len = 16;
    meta.num_query_heads = 8;
    meta.group_size = 4;

    // Create mock KV data
    size_t kv_size = 2 * 2 * 2 * 16 * 64 * sizeof(float);  // 2 layers, 2 groups, K+V
    std::vector<float> kv_buffer(kv_size / sizeof(float), 1.5f);

    std::vector<Slice> kv_data = {
        {kv_buffer.data(), kv_size}
    };

    size_t serialized_size = handler_->calculateSerializedSize(kv_data, meta);
    std::vector<uint8_t> buffer(serialized_size);

    auto result = handler_->serialize(kv_data, meta, buffer.data(), buffer.size());
    EXPECT_EQ(result, ErrorCode::OK);

    std::vector<Slice> out_kv_data;
    KVCacheMetadataBase out_meta;
    result = handler_->deserialize(buffer.data(), buffer.size(), out_kv_data, out_meta);
    EXPECT_EQ(result, ErrorCode::OK);

    // Verify metadata
    EXPECT_EQ(out_meta.layout_type, KVCacheLayoutType::GQA);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd build && make -j && ctest -R gqa_layout_handler_test -V`
Expected: Test fails

- [ ] **Step 3: Create gqa_layout_handler.h**

```cpp
#pragma once

#include "kvcache_layout_handler.h"
#include <cstdint>

namespace mooncake {

/**
 * @brief Layout handler for GQA (Grouped Query Attention) KVCache
 *
 * GQA stores K/V for groups of query heads, reducing memory compared to MHA.
 * Storage format: [metadata][layer_0_k][layer_0_v][layer_1_k]...
 */
class GQALayoutHandler : public KVCacheLayoutHandler {
public:
    KVCacheLayoutType getType() const override {
        return KVCacheLayoutType::GQA;
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

    /**
     * @brief Calculate raw KV data size from metadata
     */
    size_t calculateKVSize(const GQAKVCacheMetadata& meta) const;

private:
    // Header format: [magic][version][metadata_json_length][metadata_json]
    static constexpr uint32_t kMagic = 0x4741434B;  // "GACK" (GQA Cache)
    static constexpr uint32_t kVersion = 1;
};

}  // namespace mooncake
```

- [ ] **Step 4: Create gqa_layout_handler.cpp**

```cpp
#include "gqa_layout_handler.h"
#include <cstring>
#include "ylt/struct_json/json_writer.h"
#include "ylt/struct_json/json_reader.h"

namespace mooncake {

struct SerializedHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t metadata_json_length;
    // Followed by: metadata_json (length = metadata_json_length)
    // Followed by: kv_data
};

size_t GQALayoutHandler::calculateKVSize(const GQAKVCacheMetadata& meta) const {
    // K + V, each layer has num_kv_groups * seq_len * head_dim elements
    return 2 * meta.num_layers * meta.num_kv_groups * meta.seq_len * meta.head_dim * sizeof(float);
}

size_t GQALayoutHandler::calculateSerializedSize(
    const std::vector<Slice>& kv_data,
    const KVCacheMetadataBase& metadata) const {

    std::string json;
    ylt::struct_json::to_json(metadata, json);

    size_t total_size = sizeof(SerializedHeader);
    total_size += json.size();

    for (const auto& slice : kv_data) {
        total_size += slice.size;
    }

    return total_size;
}

ErrorCode GQALayoutHandler::serialize(
    const std::vector<Slice>& kv_data,
    const KVCacheMetadataBase& metadata,
    void* buffer,
    size_t buffer_size) const {

    std::string json;
    ylt::struct_json::to_json(metadata, json);

    size_t required = sizeof(SerializedHeader) + json.size();
    for (const auto& slice : kv_data) {
        required += slice.size;
    }

    if (buffer_size < required) {
        return ErrorCode::BUFFER_OVERFLOW;
    }

    uint8_t* ptr = static_cast<uint8_t*>(buffer);

    // Write header
    SerializedHeader header;
    header.magic = kMagic;
    header.version = kVersion;
    header.metadata_json_length = static_cast<uint32_t>(json.size());
    std::memcpy(ptr, &header, sizeof(header));
    ptr += sizeof(header);

    // Write metadata JSON
    std::memcpy(ptr, json.data(), json.size());
    ptr += json.size();

    // Write KV data
    for (const auto& slice : kv_data) {
        std::memcpy(ptr, slice.ptr, slice.size);
        ptr += slice.size;
    }

    return ErrorCode::OK;
}

ErrorCode GQALayoutHandler::deserialize(
    const void* buffer,
    size_t buffer_size,
    std::vector<Slice>& kv_data,
    KVCacheMetadataBase& metadata) const {

    if (buffer_size < sizeof(SerializedHeader)) {
        return ErrorCode::INVALID_PARAMS;
    }

    const uint8_t* ptr = static_cast<const uint8_t*>(buffer);

    // Read header
    SerializedHeader header;
    std::memcpy(&header, ptr, sizeof(header));
    ptr += sizeof(header);

    if (header.magic != kMagic) {
        return ErrorCode::INVALID_PARAMS;
    }

    if (header.version != kVersion) {
        return ErrorCode::INVALID_VERSION;
    }

    if (buffer_size < sizeof(SerializedHeader) + header.metadata_json_length) {
        return ErrorCode::INVALID_PARAMS;
    }

    // Read metadata JSON
    std::string json(reinterpret_cast<const char*>(ptr), header.metadata_json_length);
    ptr += header.metadata_json_length;

    GQAKVCacheMetadata gqa_meta;
    ylt::struct_json::from_json(gqa_meta, json);
    metadata = gqa_meta;

    // Calculate expected KV size
    size_t expected_kv_size = calculateKVSize(gqa_meta);
    size_t remaining = buffer_size - sizeof(SerializedHeader) - header.metadata_json_length;

    if (remaining < expected_kv_size) {
        return ErrorCode::INVALID_PARAMS;
    }

    // Return slice pointing to KV data (caller must manage memory)
    Slice kv_slice;
    kv_slice.ptr = const_cast<void*>(static_cast<const void*>(ptr));
    kv_slice.size = expected_kv_size;
    kv_data.push_back(kv_slice);

    return ErrorCode::OK;
}

bool GQALayoutHandler::validate(
    const std::vector<Slice>& kv_data,
    const KVCacheMetadataBase& metadata) const {

    if (metadata.layout_type != KVCacheLayoutType::GQA) {
        return false;
    }

    const auto& gqa_meta = static_cast<const GQAKVCacheMetadata&>(metadata);

    if (gqa_meta.num_layers == 0 || gqa_meta.num_kv_groups == 0 ||
        gqa_meta.head_dim == 0 || gqa_meta.seq_len == 0) {
        return false;
    }

    size_t expected = calculateKVSize(gqa_meta);
    size_t actual = 0;
    for (const auto& slice : kv_data) {
        actual += slice.size;
    }

    return actual >= expected;
}

}  // namespace mooncake
```

- [ ] **Step 5: Update CMakeLists.txt**

Add `gqa_layout_handler.cpp` to source files and create test target.

- [ ] **Step 6: Run test to verify it passes**

Run: `cd build && make -j && ctest -R gqa_layout_handler_test -V`
Expected: Tests pass

- [ ] **Step 7: Commit**

Run: `git add -A && git commit -m "feat(store): implement GQA layout handler for KVCache"`

---

## Chunk 1 Checkpoint

At this point, we have:
1. ✅ `KVCacheLayoutType` enum
2. ✅ `GQAKVCacheMetadata`, `MLAKVCacheMetadata`, `HybridKVCacheMetadata` structures
3. ✅ `KVCacheLayoutHandler` abstract interface
4. ✅ `GQALayoutHandler` implementation

Ready to proceed with MLA handler and Client API integration?

---

## Phase 2: MLA Layout Handler (P1)

### Task 5: Implement MLA Layout Handler

**Files:**
- Create: `mooncake-store/include/mla_layout_handler.h`
- Create: `mooncake-store/src/mla_layout_handler.cpp`
- Modify: `mooncake-store/CMakeLists.txt`
- Test: `mooncake-store/tests/mla_layout_handler_test.cpp`

- [ ] **Step 1: Write the failing test for MLA layout handler**

```cpp
// In mla_layout_handler_test.cpp
#include "mla_layout_handler.h"
#include <gtest/gtest.h>

class MLALayoutHandlerTest : public ::testing::Test {
protected:
    void SetUp() override {
        handler_ = std::make_unique<MLALayoutHandler>();
    }
    std::unique_ptr<MLALayoutHandler> handler_;
};

TEST_F(MLALayoutHandlerTest, ReturnsCorrectType) {
    EXPECT_EQ(handler_->getType(), KVCacheLayoutType::MLA);
}

TEST_F(MLALayoutHandlerTest, CalculateSizeIsSmallerThanMHA) {
    MLAKVCacheMetadata mla_meta;
    mla_meta.num_layers = 32;
    mla_meta.latent_dim = 512;
    mla_meta.seq_len = 1024;

    // MLA size: num_layers * seq_len * latent_dim
    size_t mla_size = handler_->calculateLatentSize(mla_meta);

    // Equivalent MHA would be: num_layers * seq_len * num_heads * head_dim * 2
    // With num_heads=32, head_dim=128: much larger
    size_t mha_equivalent = 32 * 1024 * 32 * 128 * 2;

    EXPECT_LT(mla_size, mha_equivalent / 8);  // MLA should be < 12.5% of MHA
}
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Create mla_layout_handler.h**

```cpp
#pragma once

#include "kvcache_layout_handler.h"

namespace mooncake {

/**
 * @brief Layout handler for MLA (Multi-Head Latent Attention) KVCache
 *
 * MLA stores compressed latent vectors instead of full K/V heads.
 * Storage format: [metadata][layer_0_latent][layer_1_latent]...
 */
class MLALayoutHandler : public KVCacheLayoutHandler {
public:
    KVCacheLayoutType getType() const override {
        return KVCacheLayoutType::MLA;
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

    /**
     * @brief Calculate latent vector size from metadata
     */
    size_t calculateLatentSize(const MLAKVCacheMetadata& meta) const;

private:
    static constexpr uint32_t kMagic = 0x4D4C4143;  // "MLAC"
    static constexpr uint32_t kVersion = 1;
};

}  // namespace mooncake
```

- [ ] **Step 4: Create mla_layout_handler.cpp** (similar pattern to GQA)

- [ ] **Step 5: Run test to verify it passes**

- [ ] **Step 6: Commit**

Run: `git add -A && git commit -m "feat(store): implement MLA layout handler for DeepSeek KVCache"`

---

### Task 6: Implement Hybrid Layout Handler

**Files:**
- Create: `mooncake-store/include/hybrid_layout_handler.h`
- Create: `mooncake-store/src/hybrid_layout_handler.cpp`
- Test: `mooncake-store/tests/hybrid_layout_handler_test.cpp`

- [ ] **Step 1: Write failing test for hybrid handler with sliding window**

- [ ] **Step 2: Implement hybrid_layout_handler.h/cpp**

- [ ] **Step 3: Run test to verify it passes**

- [ ] **Step 4: Commit**

---

### Task 7: Implement Layout Handler Factory

**Files:**
- Create: `mooncake-store/src/kvcache_layout_handler.cpp`
- Modify: `mooncake-store/include/kvcache_layout_handler.h`

- [ ] **Step 1: Write failing test for factory**

```cpp
TEST(KVCacheLayoutHandlerFactoryTest, CreatesCorrectHandlers) {
    auto gqa = KVCacheLayoutHandlerFactory::create(KVCacheLayoutType::GQA);
    EXPECT_NE(gqa, nullptr);
    EXPECT_EQ(gqa->getType(), KVCacheLayoutType::GQA);

    auto mla = KVCacheLayoutHandlerFactory::create(KVCacheLayoutType::MLA);
    EXPECT_NE(mla, nullptr);
    EXPECT_EQ(mla->getType(), KVCacheLayoutType::MLA);

    auto unknown = KVCacheLayoutHandlerFactory::create(KVCacheLayoutType::UNKNOWN);
    EXPECT_EQ(unknown, nullptr);
}
```

- [ ] **Step 2: Implement factory**

```cpp
// In kvcache_layout_handler.cpp
#include "kvcache_layout_handler.h"
#include "gqa_layout_handler.h"
#include "mla_layout_handler.h"
#include "hybrid_layout_handler.h"

namespace mooncake {

std::unique_ptr<KVCacheLayoutHandler> KVCacheLayoutHandlerFactory::create(KVCacheLayoutType type) {
    switch (type) {
        case KVCacheLayoutType::GQA:
            return std::make_unique<GQALayoutHandler>();
        case KVCacheLayoutType::MLA:
            return std::make_unique<MLALayoutHandler>();
        case KVCacheLayoutType::HYBRID:
            return std::make_unique<HybridLayoutHandler>();
        default:
            return nullptr;
    }
}

}  // namespace mooncake
```

- [ ] **Step 3: Run test to verify it passes**

- [ ] **Step 4: Commit**

---

## Phase 3: Client API Integration (P0)

### Task 8: Add Architecture-Aware Put Method to Client

**Files:**
- Modify: `mooncake-store/include/client_service.h`
- Modify: `mooncake-store/src/client_service.cpp`
- Test: `mooncake-store/tests/client_kvcache_test.cpp`

- [ ] **Step 1: Write failing test for PutKVCache**

```cpp
TEST(ClientKVCacheTest, PutKVCacheWithGQAMetadata) {
    // Setup client
    auto client = Client::Create(/* params */);
    ASSERT_TRUE(client.has_value());

    // Create GQA metadata
    GQAKVCacheMetadata meta;
    meta.num_layers = 32;
    meta.num_kv_groups = 4;
    meta.head_dim = 128;
    meta.seq_len = 1024;
    meta.num_query_heads = 32;
    meta.group_size = 8;

    // Create mock KV data
    size_t kv_size = 2 * 32 * 4 * 1024 * 128 * sizeof(float);
    std::vector<float> kv_buffer(kv_size / sizeof(float), 1.0f);
    std::vector<Slice> slices = {{kv_buffer.data(), kv_size}};

    ReplicateConfig config;
    config.num_replicas = 1;

    auto result = (*client)->PutKVCache("test_gqa_key", slices, config, meta);
    EXPECT_EQ(result.has_value(), true);
}
```

- [ ] **Step 2: Add PutKVCache declaration to client_service.h**

```cpp
// Add after existing Put method
/**
 * @brief Stores KVCache data with architecture-aware metadata
 * @param key Object key
 * @param slices Vector of data slices to store
 * @param config Replication configuration
 * @param metadata Architecture-specific KVCache metadata
 * @return ErrorCode indicating success/failure
 */
tl::expected<void, ErrorCode> PutKVCache(
    const ObjectKey& key,
    std::vector<Slice>& slices,
    const ReplicateConfig& config,
    const KVCacheMetadataBase& metadata);
```

- [ ] **Step 3: Implement PutKVCache in client_service.cpp**

```cpp
tl::expected<void, ErrorCode> Client::PutKVCache(
    const ObjectKey& key,
    std::vector<Slice>& slices,
    const ReplicateConfig& config,
    const KVCacheMetadataBase& metadata) {

    // Create appropriate layout handler
    auto handler = KVCacheLayoutHandlerFactory::create(metadata.layout_type);
    if (!handler) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    // Calculate serialized size
    size_t serialized_size = handler->calculateSerializedSize(slices, metadata);

    // Allocate buffer for serialized data
    std::vector<uint8_t> buffer(serialized_size);

    // Serialize
    auto result = handler->serialize(slices, metadata, buffer.data(), buffer.size());
    if (result != ErrorCode::OK) {
        return tl::make_unexpected(result);
    }

    // Create slice for serialized data and call regular Put
    Slice serialized_slice{buffer.data(), buffer.size()};
    std::vector<Slice> serialized_slices = {serialized_slice};

    return Put(key, serialized_slices, config);
}
```

- [ ] **Step 4: Run test to verify it passes**

- [ ] **Step 5: Commit**

---

### Task 9: Add Architecture-Aware Get Method to Client

**Files:**
- Modify: `mooncake-store/include/client_service.h`
- Modify: `mooncake-store/src/client_service.cpp`
- Test: `mooncake-store/tests/client_kvcache_test.cpp`

- [ ] **Step 1: Write failing test for GetKVCache**

```cpp
TEST(ClientKVCacheTest, GetKVCachePreservesMetadata) {
    auto client = Client::Create(/* params */);
    ASSERT_TRUE(client.has_value());

    // First put with metadata
    GQAKVCacheMetadata put_meta;
    put_meta.num_layers = 32;
    put_meta.num_kv_groups = 4;
    // ... set other fields

    // ... put data

    // Now get with expected layout type
    std::vector<Slice> out_slices;
    KVCacheMetadataBase out_meta;
    auto result = (*client)->GetKVCache("test_gqa_key", out_slices, out_meta, KVCacheLayoutType::GQA);

    EXPECT_EQ(result.has_value(), true);
    EXPECT_EQ(out_meta.layout_type, KVCacheLayoutType::GQA);
}
```

- [ ] **Step 2: Add GetKVCache declaration to client_service.h**

```cpp
/**
 * @brief Retrieves KVCache data with architecture-aware deserialization
 * @param object_key Key to retrieve
 * @param slices Vector of slices to store the retrieved data
 * @param metadata Output architecture-specific metadata
 * @param expected_type Expected layout type for validation
 * @return ErrorCode indicating success/failure
 */
tl::expected<void, ErrorCode> GetKVCache(
    const std::string& object_key,
    std::vector<Slice>& slices,
    KVCacheMetadataBase& metadata,
    KVCacheLayoutType expected_type = KVCacheLayoutType::UNKNOWN);
```

- [ ] **Step 3: Implement GetKVCache in client_service.cpp**

- [ ] **Step 4: Run test to verify it passes**

- [ ] **Step 5: Commit**

---

## Phase 4: Integration Testing (P1)

### Task 10: Integration Test with vLLM Mock

**Files:**
- Create: `tests/integration/kvcache_vllm_integration_test.cpp`

- [ ] **Step 1: Write integration test simulating vLLM prefill-decode flow**

- [ ] **Step 2: Run integration test**

- [ ] **Step 3: Fix any issues**

- [ ] **Step 4: Commit**

---

### Task 11: Performance Benchmark

**Files:**
- Create: `mooncake-store/benchmarks/kvcache_benchmark.cpp`

- [ ] **Step 1: Create benchmark for GQA vs MHA size comparison**

- [ ] **Step 2: Create benchmark for MLA compression ratio**

- [ ] **Step 3: Create benchmark for serialize/deserialize throughput**

- [ ] **Step 4: Run benchmarks and document results**

- [ ] **Step 5: Commit**

---

## Phase 5: Documentation and Polish (P2)

### Task 12: Update Documentation

**Files:**
- Update: `docs/source/getting_started/kvcache-layouts.md`
- Update: `README.md` (add feature mention)

- [ ] **Step 1: Create KVCache layout documentation**

- [ ] **Step 2: Add API usage examples**

- [ ] **Step 3: Add architecture decision records**

- [ ] **Step 4: Commit**

---

## Summary

| Phase | Tasks | Priority | Estimated Complexity |
|-------|-------|----------|---------------------|
| Phase 1: Core Abstraction | Tasks 1-4 | P0 | Medium |
| Phase 2: MLA Handler | Tasks 5-7 | P1 | Medium |
| Phase 3: Client API | Tasks 8-9 | P0 | High |
| Phase 4: Integration | Tasks 10-11 | P1 | Medium |
| Phase 5: Documentation | Task 12 | P2 | Low |

**Key Files to Create:**
- `mooncake-store/include/kvcache_layout.h`
- `mooncake-store/include/kvcache_layout_handler.h`
- `mooncake-store/include/gqa_layout_handler.h`
- `mooncake-store/include/mla_layout_handler.h`
- `mooncake-store/include/hybrid_layout_handler.h`
- `mooncake-store/src/gqa_layout_handler.cpp`
- `mooncake-store/src/mla_layout_handler.cpp`
- `mooncake-store/src/hybrid_layout_handler.cpp`
- `mooncake-store/src/kvcache_layout_handler.cpp`

**Key Files to Modify:**
- `mooncake-store/include/client_service.h`
- `mooncake-store/src/client_service.cpp`
- `mooncake-store/CMakeLists.txt`

**Test Files to Create:**
- `mooncake-store/tests/kvcache_layout_test.cpp`
- `mooncake-store/tests/kvcache_layout_handler_test.cpp`
- `mooncake-store/tests/gqa_layout_handler_test.cpp`
- `mooncake-store/tests/mla_layout_handler_test.cpp`
- `mooncake-store/tests/hybrid_layout_handler_test.cpp`
- `mooncake-store/tests/client_kvcache_test.cpp`

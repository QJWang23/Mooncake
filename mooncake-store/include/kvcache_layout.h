#pragma once

#include <string>
#include <cstdint>
#include "ylt/struct_json/json_reader.h"
#include "ylt/struct_json/json_writer.h"

namespace mooncake {

/**
 * @brief Enum representing different KVCache layout types
 *
 * Different LLM architectures use different KVCache formats:
 * - MHA: Traditional Multi-Head Attention (all heads stored separately)
 * - GQA: Grouped Query Attention (query heads share KV, e.g., GLM-4, Qwen)
 * - MLA: Multi-Head Latent Attention (compressed latent vectors, e.g., DeepSeek)
 * - HYBRID: Sliding Window or mixed attention patterns (e.g., Qwen with local attention)
 */
enum class KVCacheLayoutType : int32_t {
    UNKNOWN = 0,
    MHA = 1,      // Traditional Multi-Head Attention
    GQA = 2,      // Grouped Query Attention (GLM-4, Qwen)
    MLA = 3,      // Multi-Head Latent Attention (DeepSeek)
    HYBRID = 4,   // Hybrid/Sliding Window Attention
};

/**
 * @brief Convert KVCacheLayoutType to string representation
 */
inline std::string toString(KVCacheLayoutType type) {
    switch (type) {
        case KVCacheLayoutType::MHA: return "MHA";
        case KVCacheLayoutType::GQA: return "GQA";
        case KVCacheLayoutType::MLA: return "MLA";
        case KVCacheLayoutType::HYBRID: return "HYBRID";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Parse KVCacheLayoutType from string
 */
inline KVCacheLayoutType fromString(const std::string& str) {
    if (str == "MHA") return KVCacheLayoutType::MHA;
    if (str == "GQA") return KVCacheLayoutType::GQA;
    if (str == "MLA") return KVCacheLayoutType::MLA;
    if (str == "HYBRID") return KVCacheLayoutType::HYBRID;
    return KVCacheLayoutType::UNKNOWN;
}

/**
 * @brief Base metadata structure for all KVCache types
 *
 * Contains common fields shared across all layout types.
 */
struct KVCacheMetadataBase {
    KVCacheLayoutType layout_type = KVCacheLayoutType::UNKNOWN;
    uint32_t num_layers = 0;
    uint32_t seq_len = 0;
};
YLT_REFL(KVCacheMetadataBase, layout_type, num_layers, seq_len);

/**
 * @brief GQA-specific KVCache metadata (GLM-4, Qwen)
 *
 * GQA stores K/V for groups of query heads, reducing memory compared to MHA.
 * Each KV group is shared by multiple query heads.
 */
struct GQAKVCacheMetadata {
    KVCacheLayoutType layout_type = KVCacheLayoutType::GQA;
    uint32_t num_layers = 0;
    uint32_t seq_len = 0;
    uint32_t num_query_heads = 0;    // Total query heads
    uint32_t num_kv_groups = 0;      // Number of KV groups
    uint32_t group_size = 0;         // Query heads per group (num_query_heads / num_kv_groups)
    uint32_t head_dim = 0;           // Dimension per head
};
YLT_REFL(GQAKVCacheMetadata, layout_type, num_layers, seq_len,
         num_query_heads, num_kv_groups, group_size, head_dim);

/**
 * @brief MLA-specific KVCache metadata (DeepSeek)
 *
 * MLA stores compressed latent vectors instead of full K/V heads,
 * significantly reducing memory footprint while maintaining quality.
 */
struct MLAKVCacheMetadata {
    KVCacheLayoutType layout_type = KVCacheLayoutType::MLA;
    uint32_t num_layers = 0;
    uint32_t seq_len = 0;
    uint32_t latent_dim = 0;         // Compressed latent dimension
    uint32_t num_heads = 0;          // Original number of heads
    uint32_t head_dim = 0;           // Original head dimension
    bool has_rope_embedding = false; // Whether RoPE is included in the latent
};
YLT_REFL(MLAKVCacheMetadata, layout_type, num_layers, seq_len,
         latent_dim, num_heads, head_dim, has_rope_embedding);

/**
 * @brief Hybrid/Sliding Window KVCache metadata (Qwen with local attention)
 *
 * Supports sliding window attention patterns where only a window of KV
 * values is stored, enabling efficient long-context handling.
 */
struct HybridKVCacheMetadata {
    KVCacheLayoutType layout_type = KVCacheLayoutType::HYBRID;
    uint32_t num_layers = 0;
    uint32_t seq_len = 0;
    uint32_t window_size = 0;        // Sliding window size
    uint32_t current_start = 0;      // Current window start position
    uint32_t num_kv_groups = 0;      // For GQA component
    uint32_t head_dim = 0;
    bool supports_random_access = false; // For sparse access patterns
};
YLT_REFL(HybridKVCacheMetadata, layout_type, num_layers, seq_len,
         window_size, current_start, num_kv_groups, head_dim, supports_random_access);

/**
 * @brief MHA-specific KVCache metadata (traditional attention)
 *
 * Traditional Multi-Head Attention where each head has its own K and V.
 * This is the baseline format used in original transformer architectures.
 */
struct MHAKVCacheMetadata {
    KVCacheLayoutType layout_type = KVCacheLayoutType::MHA;
    uint32_t num_layers = 0;
    uint32_t seq_len = 0;
    uint32_t num_heads = 0;          // Total number of attention heads
    uint32_t head_dim = 0;           // Dimension per head
};
YLT_REFL(MHAKVCacheMetadata, layout_type, num_layers, seq_len, num_heads, head_dim);

}  // namespace mooncake

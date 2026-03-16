#include "kvcache_layout.h"
#include <gtest/gtest.h>
#include <string>

namespace mooncake {

// Test KVCacheLayoutType enum and string conversion
TEST(KVCacheLayoutTest, LayoutTypeToString) {
    EXPECT_EQ(toString(KVCacheLayoutType::MHA), "MHA");
    EXPECT_EQ(toString(KVCacheLayoutType::GQA), "GQA");
    EXPECT_EQ(toString(KVCacheLayoutType::MLA), "MLA");
    EXPECT_EQ(toString(KVCacheLayoutType::HYBRID), "HYBRID");
    EXPECT_EQ(toString(KVCacheLayoutType::UNKNOWN), "UNKNOWN");
}

TEST(KVCacheLayoutTest, LayoutTypeFromString) {
    EXPECT_EQ(fromString("MHA"), KVCacheLayoutType::MHA);
    EXPECT_EQ(fromString("GQA"), KVCacheLayoutType::GQA);
    EXPECT_EQ(fromString("MLA"), KVCacheLayoutType::MLA);
    EXPECT_EQ(fromString("HYBRID"), KVCacheLayoutType::HYBRID);
    EXPECT_EQ(fromString("unknown"), KVCacheLayoutType::UNKNOWN);
    EXPECT_EQ(fromString("INVALID"), KVCacheLayoutType::UNKNOWN);
}

// Test KVCacheMetadataBase serialization
TEST(KVCacheLayoutTest, BaseMetadataSerialization) {
    KVCacheMetadataBase meta;
    meta.layout_type = KVCacheLayoutType::GQA;
    meta.num_layers = 32;
    meta.seq_len = 1024;

    std::string json;
    ylt::struct_json::to_json(meta, json);

    KVCacheMetadataBase parsed;
    ylt::struct_json::from_json(parsed, json);

    EXPECT_EQ(parsed.layout_type, KVCacheLayoutType::GQA);
    EXPECT_EQ(parsed.num_layers, 32);
    EXPECT_EQ(parsed.seq_len, 1024);
}

// Test GQA metadata serialization
TEST(KVCacheLayoutTest, GQAMetadataSerialization) {
    GQAKVCacheMetadata meta;
    meta.layout_type = KVCacheLayoutType::GQA;
    meta.num_layers = 32;
    meta.seq_len = 2048;
    meta.num_query_heads = 32;
    meta.num_kv_groups = 4;
    meta.group_size = 8;
    meta.head_dim = 128;

    std::string json;
    ylt::struct_json::to_json(meta, json);

    // Verify JSON contains expected fields
    EXPECT_NE(json.find("\"num_query_heads\":32"), std::string::npos);
    EXPECT_NE(json.find("\"num_kv_groups\":4"), std::string::npos);

    GQAKVCacheMetadata parsed;
    ylt::struct_json::from_json(parsed, json);

    EXPECT_EQ(parsed.layout_type, KVCacheLayoutType::GQA);
    EXPECT_EQ(parsed.num_layers, 32);
    EXPECT_EQ(parsed.seq_len, 2048);
    EXPECT_EQ(parsed.num_query_heads, 32);
    EXPECT_EQ(parsed.num_kv_groups, 4);
    EXPECT_EQ(parsed.group_size, 8);
    EXPECT_EQ(parsed.head_dim, 128);
}

// Test MLA metadata serialization
TEST(KVCacheLayoutTest, MLAMetadataSerialization) {
    MLAKVCacheMetadata meta;
    meta.layout_type = KVCacheLayoutType::MLA;
    meta.num_layers = 32;
    meta.seq_len = 4096;
    meta.latent_dim = 512;
    meta.num_heads = 32;
    meta.head_dim = 128;
    meta.has_rope_embedding = true;

    std::string json;
    ylt::struct_json::to_json(meta, json);

    // Verify JSON contains expected fields
    EXPECT_NE(json.find("\"latent_dim\":512"), std::string::npos);
    EXPECT_NE(json.find("\"has_rope_embedding\":true"), std::string::npos);

    MLAKVCacheMetadata parsed;
    ylt::struct_json::from_json(parsed, json);

    EXPECT_EQ(parsed.layout_type, KVCacheLayoutType::MLA);
    EXPECT_EQ(parsed.num_layers, 32);
    EXPECT_EQ(parsed.seq_len, 4096);
    EXPECT_EQ(parsed.latent_dim, 512);
    EXPECT_EQ(parsed.num_heads, 32);
    EXPECT_EQ(parsed.head_dim, 128);
    EXPECT_TRUE(parsed.has_rope_embedding);
}

// Test Hybrid metadata serialization
TEST(KVCacheLayoutTest, HybridMetadataSerialization) {
    HybridKVCacheMetadata meta;
    meta.layout_type = KVCacheLayoutType::HYBRID;
    meta.num_layers = 32;
    meta.seq_len = 8192;
    meta.window_size = 4096;
    meta.current_start = 100;
    meta.num_kv_groups = 8;
    meta.head_dim = 128;
    meta.supports_random_access = true;

    std::string json;
    ylt::struct_json::to_json(meta, json);

    // Verify JSON contains expected fields
    EXPECT_NE(json.find("\"window_size\":4096"), std::string::npos);
    EXPECT_NE(json.find("\"supports_random_access\":true"), std::string::npos);

    HybridKVCacheMetadata parsed;
    ylt::struct_json::from_json(parsed, json);

    EXPECT_EQ(parsed.layout_type, KVCacheLayoutType::HYBRID);
    EXPECT_EQ(parsed.num_layers, 32);
    EXPECT_EQ(parsed.seq_len, 8192);
    EXPECT_EQ(parsed.window_size, 4096);
    EXPECT_EQ(parsed.current_start, 100);
    EXPECT_EQ(parsed.num_kv_groups, 8);
    EXPECT_EQ(parsed.head_dim, 128);
    EXPECT_TRUE(parsed.supports_random_access);
}

// Test MHA metadata serialization
TEST(KVCacheLayoutTest, MHAMetadataSerialization) {
    MHAKVCacheMetadata meta;
    meta.layout_type = KVCacheLayoutType::MHA;
    meta.num_layers = 32;
    meta.seq_len = 2048;
    meta.num_heads = 32;
    meta.head_dim = 128;

    std::string json;
    ylt::struct_json::to_json(meta, json);

    MHAKVCacheMetadata parsed;
    ylt::struct_json::from_json(parsed, json);

    EXPECT_EQ(parsed.layout_type, KVCacheLayoutType::MHA);
    EXPECT_EQ(parsed.num_layers, 32);
    EXPECT_EQ(parsed.seq_len, 2048);
    EXPECT_EQ(parsed.num_heads, 32);
    EXPECT_EQ(parsed.head_dim, 128);
}

// Test GQA metadata default values
TEST(KVCacheLayoutTest, GQAMetadataDefaults) {
    GQAKVCacheMetadata meta;

    EXPECT_EQ(meta.layout_type, KVCacheLayoutType::GQA);
    EXPECT_EQ(meta.num_layers, 0);
    EXPECT_EQ(meta.seq_len, 0);
    EXPECT_EQ(meta.num_query_heads, 0);
    EXPECT_EQ(meta.num_kv_groups, 0);
    EXPECT_EQ(meta.group_size, 0);
    EXPECT_EQ(meta.head_dim, 0);
}

// Test MLA metadata default values
TEST(KVCacheLayoutTest, MLAMetadataDefaults) {
    MLAKVCacheMetadata meta;

    EXPECT_EQ(meta.layout_type, KVCacheLayoutType::MLA);
    EXPECT_EQ(meta.num_layers, 0);
    EXPECT_EQ(meta.seq_len, 0);
    EXPECT_EQ(meta.latent_dim, 0);
    EXPECT_EQ(meta.num_heads, 0);
    EXPECT_EQ(meta.head_dim, 0);
    EXPECT_FALSE(meta.has_rope_embedding);
}

// Test Hybrid metadata default values
TEST(KVCacheLayoutTest, HybridMetadataDefaults) {
    HybridKVCacheMetadata meta;

    EXPECT_EQ(meta.layout_type, KVCacheLayoutType::HYBRID);
    EXPECT_EQ(meta.num_layers, 0);
    EXPECT_EQ(meta.seq_len, 0);
    EXPECT_EQ(meta.window_size, 0);
    EXPECT_EQ(meta.current_start, 0);
    EXPECT_EQ(meta.num_kv_groups, 0);
    EXPECT_EQ(meta.head_dim, 0);
    EXPECT_FALSE(meta.supports_random_access);
}

// Test that GQA group calculation is consistent
TEST(KVCacheLayoutTest, GQAGroupConsistency) {
    GQAKVCacheMetadata meta;
    meta.num_query_heads = 32;
    meta.num_kv_groups = 4;
    meta.group_size = 8;  // 32 / 4 = 8

    // Verify the relationship: num_query_heads = num_kv_groups * group_size
    EXPECT_EQ(meta.num_query_heads, meta.num_kv_groups * meta.group_size);
}

// Test JSON round-trip for all metadata types
TEST(KVCacheLayoutTest, JSONRoundTripAllTypes) {
    // Test GQA
    {
        GQAKVCacheMetadata original;
        original.layout_type = KVCacheLayoutType::GQA;
        original.num_layers = 16;
        original.seq_len = 512;
        original.num_query_heads = 16;
        original.num_kv_groups = 2;
        original.group_size = 8;
        original.head_dim = 64;

        std::string json;
        ylt::struct_json::to_json(original, json);

        GQAKVCacheMetadata restored;
        ylt::struct_json::from_json(restored, json);

        EXPECT_EQ(original.layout_type, restored.layout_type);
        EXPECT_EQ(original.num_layers, restored.num_layers);
        EXPECT_EQ(original.seq_len, restored.seq_len);
        EXPECT_EQ(original.num_query_heads, restored.num_query_heads);
        EXPECT_EQ(original.num_kv_groups, restored.num_kv_groups);
        EXPECT_EQ(original.group_size, restored.group_size);
        EXPECT_EQ(original.head_dim, restored.head_dim);
    }

    // Test MLA
    {
        MLAKVCacheMetadata original;
        original.layout_type = KVCacheLayoutType::MLA;
        original.num_layers = 24;
        original.seq_len = 1024;
        original.latent_dim = 256;
        original.num_heads = 16;
        original.head_dim = 64;
        original.has_rope_embedding = false;

        std::string json;
        ylt::struct_json::to_json(original, json);

        MLAKVCacheMetadata restored;
        ylt::struct_json::from_json(restored, json);

        EXPECT_EQ(original.layout_type, restored.layout_type);
        EXPECT_EQ(original.num_layers, restored.num_layers);
        EXPECT_EQ(original.seq_len, restored.seq_len);
        EXPECT_EQ(original.latent_dim, restored.latent_dim);
        EXPECT_EQ(original.num_heads, restored.num_heads);
        EXPECT_EQ(original.head_dim, restored.head_dim);
        EXPECT_EQ(original.has_rope_embedding, restored.has_rope_embedding);
    }

    // Test Hybrid
    {
        HybridKVCacheMetadata original;
        original.layout_type = KVCacheLayoutType::HYBRID;
        original.num_layers = 28;
        original.seq_len = 2048;
        original.window_size = 1024;
        original.current_start = 512;
        original.num_kv_groups = 4;
        original.head_dim = 96;
        original.supports_random_access = false;

        std::string json;
        ylt::struct_json::to_json(original, json);

        HybridKVCacheMetadata restored;
        ylt::struct_json::from_json(restored, json);

        EXPECT_EQ(original.layout_type, restored.layout_type);
        EXPECT_EQ(original.num_layers, restored.num_layers);
        EXPECT_EQ(original.seq_len, restored.seq_len);
        EXPECT_EQ(original.window_size, restored.window_size);
        EXPECT_EQ(original.current_start, restored.current_start);
        EXPECT_EQ(original.num_kv_groups, restored.num_kv_groups);
        EXPECT_EQ(original.head_dim, restored.head_dim);
        EXPECT_EQ(original.supports_random_access, restored.supports_random_access);
    }
}

}  // namespace mooncake

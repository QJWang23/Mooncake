# Mooncake x 灵衡软件实现指南

## 文档版本
- **版本**: v1.0
- **日期**: 2025年1月26日
- **配套文档**: Mooncake_TieredCache_LingQu_Architecture_Evolution.md

---

## 目录

1. [开发环境搭建](#1-开发环境搭建)
2. [核心组件实现](#2-核心组件实现)
3. [集成与测试](#3-集成与测试)
4. [性能调优](#4-性能调优)
5. [故障排查](#5-故障排查)

---

## 1. 开发环境搭建

### 1.1 系统要求

```bash
# 操作系统
OS: CentOS 7.9+ / Ubuntu 20.04+

# 硬件要求
- 昇腾A3 NPU (至少1张)
- 灵衡交换机支持
- RDMA网卡 (推荐200Gbps+)
- 内存: 至少256GB
- 存储: 至少2TB NVMe SSD

# 软件依赖
- CANN 8.0+
- Python 3.10+
- CUDA 12.1+ (可选)
- RDMA驱动
- 灵衡驱动
```

### 1.2 依赖安装

```bash
#!/bin/bash
# install_lingqu_deps.sh

# 1. 安装CANN
# 请参考华为官方文档

# 2. 安装RDMA驱动
apt-get install -y rdma-core libibverbs-dev librdmacm-dev

# 3. 安装灵衡驱动
# 请联系华为获取灵衡驱动包

# 4. 安装MemFabric Hybrid
git clone https://github.com/your-org/memfabric-hybrid.git
cd memfabric-hybrid
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install

# 5. 安装Mooncake依赖
cd /path/to/Mooncake
bash dependencies.sh

# 6. 编译Mooncake (启用灵衡支持)
mkdir build && cd build
cmake -DWITH_LINGQU=ON -DWITH_STORE=ON ..
make -j$(nproc)
sudo make install
```

### 1.3 配置文件

```json
// mooncake_lingqu_config.json
{
  "metadata_service": {
    "type": "etcd",
    "conn_string": "http://localhost:2379"
  },
  "lingqu": {
    "enabled": true,
    "transport_type": "HCCP",
    "net_dev": "ib0",
    "net_addr": "192.168.1.100",
    "port": 12345,
    "enable_rdma": true,
    "enable_fabric_mem": true
  },
  "gva": {
    "hbm_base": "0x100000000000",
    "hbm_size": "0x4000000000",    // 256GB
    "dram_base": "0x180000000000",
    "dram_size": "0x800000000",    // 32GB
    "host_swap_size": "0x80000000", // 2GB
    "device_swap_size": "0x8000000" // 128MB
  },
  "tiered_cache": {
    "tiers": [
      {
        "id": 0,
        "type": "LOCAL_HBM",
        "priority": 0,
        "capacity": 64424509434    // 64GB
      },
      {
        "id": 1,
        "type": "LINGQU_HBM",
        "priority": 1,
        "capacity": 1099511627776   // 1TB
      },
      {
        "id": 2,
        "type": "LOCAL_DRAM",
        "priority": 2,
        "capacity": 2147483648     // 2GB
      },
      {
        "id": 3,
        "type": "DISTRIBUTED",
        "priority": 3,
        "capacity": 0               // 无限制
      }
    ]
  }
}
```

---

## 2. 核心组件实现

### 2.1 灵衡CacheTier实现

```cpp
// mooncake-store/src/tiered_cache/lingqu_tier.cpp

#include "tiered_cache/lingqu_tier.h"
#include "gva_manager.h"
#include "lingqu_transport.h"

namespace mooncake {
namespace store {

bool LingQuCacheTier::Init(TieredBackend* backend, TransferEngine* engine) {
    backend_ = backend;
    engine_ = engine;

    // 1. 初始化GVA管理器
    if (!gva_manager_.Init(config_.etcd_url, config_.local_rank_id)) {
        LOG(ERROR) << "Failed to initialize GVA manager";
        return false;
    }

    // 2. 初始化交换缓冲区 (2MB对齐)
    host_swap_buffer_ = aligned_alloc(2 * 1024 * 1024, config_.host_swap_size);
    if (!host_swap_buffer_) {
        LOG(ERROR) << "Failed to allocate host swap buffer";
        return false;
    }

    // 3. 注册交换缓冲区到传输层
    if (!engine_->registerLocalMemory(host_swap_buffer_,
                                     config_.host_swap_size)) {
        LOG(ERROR) << "Failed to register host swap buffer";
        return false;
    }

    // 4. 初始化设备交换缓冲区
    if (config_.enable_fabric_mem) {
        // 使用Ascend API分配设备内存
        auto ret = aclrtMalloc(&device_swap_buffer_, config_.device_swap_size);
        if (ret != ACL_ERROR_NONE) {
            LOG(ERROR) << "Failed to allocate device swap buffer";
            return false;
        }

        // 注册到灵衡传输层
        if (!engine_->registerLocalMemory(device_swap_buffer_,
                                         config_.device_swap_size)) {
            LOG(ERROR) << "Failed to register device swap buffer";
            return false;
        }
    }

    // 5. 初始化RH2D传输
    if (!transport_.Init(&gva_manager_, host_swap_buffer_,
                        device_swap_buffer_)) {
        LOG(ERROR) << "Failed to initialize RH2D transport";
        return false;
    }

    LOG(INFO) << "LingQuCacheTier initialized successfully";
    return true;
}

std::optional<TieredLocation> LingQuCacheTier::Allocate(size_t size) {
    // 1. 对齐到2MB边界 (灵衡要求)
    size_t aligned_size = (size + 2 * 1024 * 1024 - 1) &
                         ~(2 * 1024 * 1024 - 1);

    // 2. 从GVA管理器分配内存
    auto gva = gva_manager_.AllocateGVA(aligned_size,
                                       HYBM_MEM_TYPE_HOST);
    if (!gva) {
        LOG(WARNING) << "Failed to allocate GVA memory, size=" << size;
        return std::nullopt;
    }

    // 3. 返回TieredLocation
    TieredLocation loc;
    loc.tier_id = tier_id_;
    loc.data.ptr = reinterpret_cast<void*>(*gva);
    loc.data.size = aligned_size;
    loc.data.type = MemoryType::DRAM;

    return loc;
}

bool LingQuCacheTier::WriteAt(uint64_t offset, const DataSource& source) {
    // 1. 确定数据源类型
    if (source.type == MemoryType::DRAM) {
        // DRAM → 灵衡DRAM
        return transport_.ExecuteGL2H(
            reinterpret_cast<uint64_t>(source.ptr),
            offset + gva_base_,
            source.size
        );
    } else if (source.type == MemoryType::HBM) {
        // HBM → 灵衡DRAM
        // 需要通过Host中转
        auto ret = aclrtMemcpy(host_swap_buffer_, source.size,
                              source.ptr, source.size,
                              ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_ERROR_NONE) {
            LOG(ERROR) << "Failed to copy from device to host";
            return false;
        }

        return transport_.ExecuteGL2H(
            reinterpret_cast<uint64_t>(host_swap_buffer_),
            offset + gva_base_,
            source.size
        );
    }

    LOG(ERROR) << "Unsupported source type: " << static_cast<int>(source.type);
    return false;
}

std::optional<DataSource> LingQuCacheTier::AsDataSource(
    const std::string& key) {

    // 1. 从元数据中查找key的GVA地址
    auto metadata = metadata_index_.Get(key);
    if (!metadata) {
        return std::nullopt;
    }

    // 2. 构造DataSource
    DataSource source;
    source.ptr = reinterpret_cast<void*>(metadata->gva);
    source.size = metadata->size;
    source.type = MemoryType::DRAM;  // 灵衡内存作为DRAM类型

    return source;
}

void LingQuCacheTier::BindKey(const std::string& key,
                             uint64_t offset,
                             size_t size) {
    // 更新元数据索引
    metadata_index_.Put(key, {
        .gva = offset + gva_base_,
        .size = size,
        .tier_id = tier_id_
    });
}

void LingQuCacheTier::Delete(const std::string& key) {
    // 从元数据中删除
    metadata_index_.Delete(key);
}

} // namespace store
} // namespace mooncake
```

### 2.2 智能路径选择器实现

```cpp
// mooncake-transfer-engine/src/path_selector.cpp

#include "path_selector.h"
#include "lingqu_transport.h"

namespace mooncake {
namespace transfer {

IntelligentPathSelector::TransportPath
IntelligentPathSelector::SelectOptimalPath(const SelectionContext& ctx) {
    // 1. 检查是否为本地访问
    if (ctx.local_rank == ctx.remote_rank) {
        return TransportPath::LOCAL_ACCESS;
    }

    // 2. 检查是否在同一超节点
    bool in_same_superpod = IsInSameSuperPod(ctx.local_rank, ctx.remote_rank);

    // 3. 获取各路径的性能指标
    auto lingqu_hccp_metrics = GetPathMetrics(TransportPath::LINGQU_HCCP);
    auto lingqu_hcom_metrics = GetPathMetrics(TransportPath::LINGQU_HCOM);
    auto rdma_metrics = GetPathMetrics(TransportPath::TRADITIONAL_RDMA);

    // 4. 决策逻辑
    if (in_same_superpod) {
        // 超节点内，优先使用灵衡HCCP
        if (IsDeviceDirectAvailable(ctx.local_addr, ctx.remote_gva)) {
            // 设备直连可用
            if (lingqu_hccp_metrics &&
                lingqu_hccp_metrics->availability > 0.95) {
                return TransportPath::LINGQU_HCCP;
            }
        }

        // 回退到HCOM
        if (lingqu_hcom_metrics &&
            lingqu_hcom_metrics->availability > 0.95) {
            return TransportPath::LINGQU_HCOM;
        }
    }

    // 5. 跨节点或灵衡不可用，使用传统RDMA
    if (rdma_metrics && rdma_metrics->availability > 0.9) {
        return TransportPath::TRADITIONAL_RDMA;
    }

    // 6. 最后回退到TCP
    return TransportPath::TCP_FALLBACK;
}

bool IntelligentPathSelector::IsInSameSuperPod(uint32_t rank1,
                                              uint32_t rank2) {
    // 通过etcd查询节点所属超节点
    // 简化实现: 假设rank高8位为超节点ID
    return (rank1 >> 24) == (rank2 >> 24);
}

bool IntelligentPathSelector::IsDeviceDirectAvailable(
    const void* local_addr,
    uint64_t remote_gva) {

    // 1. 检查本地地址是否为设备内存
    // (通过/proc/self/maps或其他方式判断)

    // 2. 检查远程GVA是否为HBM空间
    // (GVA地址范围判断)

    // 3. 检查是否启用Fabric Memory模式
    bool fabric_mem_enabled = GetFabricMemConfig();

    return fabric_mem_enabled &&
           IsDeviceMemory(local_addr) &&
           IsHBMGVA(remote_gva);
}

void IntelligentPathSelector::UpdatePathMetrics(
    TransportPath path,
    const PathMetrics& metrics) {

    std::lock_guard<std::mutex> lock(metrics_mutex_);

    // 指数移动平均平滑
    auto& existing = path_metrics_[path];
    const double alpha = 0.3;  // 平滑因子

    if (existing.avg_latency_us == 0) {
        // 首次更新
        existing = metrics;
    } else {
        // EMA更新
        existing.avg_latency_us =
            alpha * metrics.avg_latency_us +
            (1 - alpha) * existing.avg_latency_us;
        existing.p99_latency_us =
            alpha * metrics.p99_latency_us +
            (1 - alpha) * existing.p99_latency_us;
        existing.bandwidth_gbps =
            alpha * metrics.bandwidth_gbps +
            (1 - alpha) * existing.bandwidth_gbps;
        existing.availability =
            alpha * metrics.availability +
            (1 - alpha) * existing.availability;
    }
}

} // namespace transfer
} // namespace mooncake
```

### 2.3 统一传输管理器实现

```cpp
// mooncake-transfer-engine/src/unified_transport_manager.cpp

#include "unified_transport.h"

namespace mooncake {
namespace transfer {

bool UnifiedTransportManager::Transfer(const TransferRequest& request) {
    auto start = std::chrono::high_resolution_clock::now();

    // 1. 选择传输引擎
    IntelligentPathSelector::SelectionContext ctx;
    ctx.local_rank = local_rank_id_;
    ctx.remote_rank = ExtractRankFromGVA(request.remote_gva);
    ctx.local_addr = request.local_addr;
    ctx.remote_gva = request.remote_gva;
    ctx.data_size = request.size;
    ctx.is_latency_critical = (request.size < 1024 * 1024);  // <1MB视为延迟敏感
    ctx.is_bandwidth_critical = (request.size > 64 * 1024 * 1024);  // >64MB视为带宽敏感

    auto engine_type = engine_selector_.SelectEngine(ctx);

    // 2. 获取传输引擎
    auto engine = engine_selector_.GetEngine(engine_type);
    if (!engine) {
        LOG(ERROR) << "Failed to get transport engine, type="
                   << static_cast<int>(engine_type);
        return false;
    }

    // 3. 执行传输
    bool success = false;
    if (request.direction == TransferRequest::READ) {
        success = engine->Read(request.local_addr,
                              reinterpret_cast<const void*>(request.remote_gva),
                              request.size);
    } else {
        success = engine->Write(request.local_addr,
                               reinterpret_cast<void*>(request.remote_gva),
                               request.size);
    }

    // 4. 更新统计信息
    auto end = std::chrono::high_resolution_clock::now();
    double latency_us =
        std::chrono::duration<double, std::micro>(end - start).count();

    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.total_transfers++;
    stats_.total_bytes += request.size;
    stats_.avg_latency_us =
        (stats_.avg_latency_us * (stats_.total_transfers - 1) + latency_us) /
        stats_.total_transfers;

    // 5. 更新路径选择器的性能指标
    IntelligentPathSelector::PathMetrics metrics;
    metrics.avg_latency_us = latency_us;
    metrics.bandwidth_gbps = (request.size * 8.0 / 1e9) / (latency_us / 1e6);
    metrics.availability = success ? 1.0 : 0.0;

    path_selector_.UpdatePathMetrics(
        static_cast<IntelligentPathSelector::TransportPath>(engine_type),
        metrics
    );

    return success;
}

bool UnifiedTransportManager::BatchTransfer(
    const std::vector<TransferRequest>& requests) {

    // 1. 聚合请求
    auto aggregated = optimizer_.AggregateRequests(requests);

    // 2. 执行聚合传输
    bool all_success = true;
    for (const auto& agg : aggregated) {
        if (!optimizer_.ExecuteAggregatedTransfer(agg)) {
            all_success = false;
            LOG(WARNING) << "Failed to execute aggregated transfer";
        }
    }

    return all_success;
}

} // namespace transfer
} // namespace mooncake
```

### 2.4 Python绑定

```python
# mooncake-wheel/mooncake/lingqu_backend.py

import ctypes
import logging
from typing import Optional, List, Dict
import numpy as np

logger = logging.getLogger(__name__)

class LingQuTier:
    """灵衡缓存层Python接口"""

    def __init__(self, config: Dict):
        """
        初始化灵衡缓存层

        Args:
            config: 配置字典，包含:
                - local_rank_id: 本地Rank ID
                - etcd_url: etcd地址
                - host_swap_size: 主机交换缓冲区大小 (bytes)
                - device_swap_size: 设备交换缓冲区大小 (bytes)
                - enable_fabric_mem: 是否启用Fabric Memory
        """
        self.config = config

        # 加载C++库
        try:
            self._lib = ctypes.CDLL("libmooncake_lingqu.so")
        except OSError as e:
            logger.error(f"Failed to load libmooncake_lingqu.so: {e}")
            raise

        # 初始化
        self._handle = self._init()

    def _init(self) -> int:
        """初始化灵衡缓存层"""
        init_func = self._lib.LingQuCacheTier_Init
        init_func.argtypes = [
            ctypes.c_uint32,  # local_rank_id
            ctypes.c_char_p,  # etcd_url
            ctypes.c_size_t,  # host_swap_size
            ctypes.c_size_t,  # device_swap_size
            ctypes.c_bool,    # enable_fabric_mem
        ]
        init_func.restype = ctypes.c_void_p

        handle = init_func(
            self.config['local_rank_id'],
            self.config['etcd_url'].encode(),
            self.config['host_swap_size'],
            self.config['device_swap_size'],
            self.config['enable_fabric_mem']
        )

        if not handle:
            raise RuntimeError("Failed to initialize LingQuCacheTier")

        return handle

    def allocate(self, size: int) -> Optional[int]:
        """
        分配内存

        Args:
            size: 分配大小 (bytes)

        Returns:
            GVA地址，失败返回None
        """
        alloc_func = self._lib.LingQuCacheTier_Allocate
        alloc_func.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        alloc_func.restype = ctypes.c_uint64

        gva = alloc_func(self._handle, size)
        if gva == 0:
            return None

        return gva

    def write(self, gva: int, data: bytes) -> bool:
        """
        写入数据

        Args:
            gva: 目标GVA地址
            data: 要写入的数据

        Returns:
            是否成功
        """
        write_func = self._lib.LingQuCacheTier_WriteAt
        write_func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,  # offset
            ctypes.c_void_p,  # data
            ctypes.c_size_t   # size
        ]
        write_func.restype = ctypes.c_bool

        # 创建临时缓冲区
        buf = ctypes.create_string_buffer(data)

        return write_func(self._handle, gva, buf, len(data))

    def read(self, gva: int, size: int) -> Optional[bytes]:
        """
        读取数据

        Args:
            gva: 源GVA地址
            size: 读取大小

        Returns:
            读取的数据，失败返回None
        """
        read_func = self._lib.LingQuCacheTier_Read
        read_func.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,  # gva
            ctypes.c_void_p,  # buffer
            ctypes.c_size_t   # size
        ]
        read_func.restype = ctypes.c_bool

        # 创建缓冲区
        buf = ctypes.create_string_buffer(size)

        success = read_func(self._handle, gva, buf, size)
        if not success:
            return None

        return buf.raw

    def free(self, gva: int) -> bool:
        """
        释放内存

        Args:
            gva: 要释放的GVA地址

        Returns:
            是否成功
        """
        free_func = self._lib.LingQuCacheTier_Free
        free_func.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        free_func.restype = ctypes.c_bool

        return free_func(self._handle, gva)

    def __del__(self):
        """清理资源"""
        if hasattr(self, '_handle') and self._handle:
            shutdown_func = self._lib.LingQuCacheTier_Shutdown
            shutdown_func.argtypes = [ctypes.c_void_p]
            shutdown_func(self._handle)


class LingQuTieredBackend:
    """灵衡分层缓存后端Python接口"""

    def __init__(self, config: Dict):
        """
        初始化分层缓存后端

        Args:
            config: 配置字典
        """
        self.lingqu_tier = LingQuTier(config['lingqu'])

    def put(self, key: str, data: bytes, tier: int = 1) -> bool:
        """
        存储数据

        Args:
            key: 键
            data: 数据
            tier: 目标层 (0=本地HBM, 1=灵衡HBM, 2=本地DRAM, 3=分布式)

        Returns:
            是否成功
        """
        # 分配内存
        gva = self.lingqu_tier.allocate(len(data))
        if gva is None:
            logger.error(f"Failed to allocate memory for key={key}")
            return False

        # 写入数据
        if not self.lingqu_tier.write(gva, data):
            logger.error(f"Failed to write data for key={key}")
            self.lingqu_tier.free(gva)
            return False

        # 绑定键
        if not self._bind_key(key, gva, len(data)):
            logger.error(f"Failed to bind key={key}")
            self.lingqu_tier.free(gva)
            return False

        return True

    def get(self, key: str) -> Optional[bytes]:
        """
        获取数据

        Args:
            key: 键

        Returns:
            数据，不存在返回None
        """
        # 查询GVA地址
        gva, size = self._query_key(key)
        if gva is None:
            return None

        # 读取数据
        return self.lingqu_tier.read(gva, size)

    def delete(self, key: str) -> bool:
        """
        删除数据

        Args:
            key: 键

        Returns:
            是否成功
        """
        return self._delete_key(key)

    def _bind_key(self, key: str, gva: int, size: int) -> bool:
        """绑定键到GVA地址"""
        # 实���键绑定逻辑
        # 可以使用etcd存储元数据
        pass

    def _query_key(self, key: str) -> tuple[Optional[int], Optional[int]]:
        """查询键的GVA地址和大小"""
        # 实现键查询逻辑
        # 从etcd读取元数据
        pass

    def _delete_key(self, key: str) -> bool:
        """删除键"""
        # 实现键删除逻辑
        # 从etcd删除元数据
        pass
```

---

## 3. 集成与测试

### 3.1 单元测试

```cpp
// mooncake-store/tests/lingqu_tier_test.cpp

#include <gtest/gtest.h>
#include "tiered_cache/lingqu_tier.h"

class LingQuCacheTierTest : public ::testing::Test {
protected:
    void SetUp() override {
        LingQuCacheTier::Config config;
        config.local_rank_id = 0;
        config.etcd_url = "http://localhost:2379";
        config.host_swap_size = 2ULL * 1024 * 1024 * 1024;  // 2GB
        config.device_swap_size = 128 * 1024 * 1024;        // 128MB
        config.enable_fabric_mem = true;

        tier_ = std::make_unique<LingQuCacheTier>(config);
        ASSERT_TRUE(tier_->Init(nullptr, nullptr));
    }

    void TearDown() override {
        tier_->Shutdown();
    }

    std::unique_ptr<LingQuCacheTier> tier_;
};

TEST_F(LingQuCacheTierTest, AllocateSuccess) {
    const size_t size = 4 * 1024 * 1024;  // 4MB
    auto location = tier_->Allocate(size);

    ASSERT_TRUE(location.has_value());
    EXPECT_NE(location->data.ptr, nullptr);
    EXPECT_EQ(location->data.size, size);
}

TEST_F(LingQuCacheTierTest, AllocateAlignment) {
    // 测试2MB对齐
    const std::vector<size_t> sizes = {
        1024,              // 1KB
        4 * 1024 * 1024,   // 4MB
        7 * 1024 * 1024,   // 7MB
    };

    for (auto size : sizes) {
        auto location = tier_->Allocate(size);
        ASSERT_TRUE(location.has_value());

        // 检查对齐
        uint64_t addr = reinterpret_cast<uint64_t>(location->data.ptr);
        EXPECT_EQ(addr % (2 * 1024 * 1024), 0);
    }
}

TEST_F(LingQuCacheTierTest, WriteAtSuccess) {
    const size_t size = 4 * 1024 * 1024;  // 4MB
    auto location = tier_->Allocate(size);
    ASSERT_TRUE(location.has_value());

    // 准备测试数据
    std::vector<char> test_data(size, 'X');
    DataSource source;
    source.ptr = test_data.data();
    source.size = size;
    source.type = MemoryType::DRAM;

    // 写入
    EXPECT_TRUE(tier_->WriteAt(location->tier_id, source));
}

TEST_F(LingQuCacheTierTest, BindKeyAndRetrieve) {
    const std::string test_key = "test_key";
    const size_t size = 4 * 1024 * 1024;

    // 分配并写入
    auto location = tier_->Allocate(size);
    ASSERT_TRUE(location.has_value());

    std::vector<char> test_data(size, 'Y');
    DataSource source;
    source.ptr = test_data.data();
    source.size = size;
    source.type = MemoryType::DRAM;

    ASSERT_TRUE(tier_->WriteAt(location->tier_id, source));

    // 绑定键
    tier_->BindKey(test_key, location->tier_id, size);

    // 检索
    auto retrieved = tier_->AsDataSource(test_key);
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ(retrieved->ptr, location->data.ptr);
    EXPECT_EQ(retrieved->size, size);
}

TEST_F(LingQuCacheTierTest, DeleteKey) {
    const std::string test_key = "test_key_delete";
    const size_t size = 4 * 1024 * 1024;

    // 分配并绑定
    auto location = tier_->Allocate(size);
    ASSERT_TRUE(location.has_value());

    tier_->BindKey(test_key, location->tier_id, size);

    // 删除
    tier_->Delete(test_key);

    // 验证删除
    auto retrieved = tier_->AsDataSource(test_key);
    EXPECT_FALSE(retrieved.has_value());
}

TEST_F(LingQuCacheTierTest, RH2DPerformance) {
    // 性能测试
    const std::vector<size_t> sizes = {
        1024,                  // 1KB
        1024 * 1024,           // 1MB
        64 * 1024 * 1024,      // 64MB
        256 * 1024 * 1024,     // 256MB
    };

    for (auto size : sizes) {
        auto start = std::chrono::high_resolution_clock::now();

        // 执行RH2D传输
        // ...

        auto end = std::chrono::high_resolution_clock::now();
        double latency_us =
            std::chrono::duration<double, std::micro>(end - start).count();

        double bandwidth_gbps = (size * 8.0 / 1e9) / (latency_us / 1e6);

        LOG(INFO) << "Size=" << size
                  << " Latency=" << latency_us << "us"
                  << " Bandwidth=" << bandwidth_gbps << "GB/s";

        // 性能断言
        if (size < 1024 * 1024) {
            // 小数据延迟目标
            EXPECT_LT(latency_us, 5.0);  // <5μs
        } else {
            // 大数据带宽目标
            EXPECT_GT(bandwidth_gbps, 50.0);  // >50GB/s
        }
    }
}
```

### 3.2 集成测试

```python
# mooncake-wheel/tests/test_lingqu_integration.py

import pytest
import numpy as np
from mooncake.lingqu_backend import LingQuTieredBackend

@pytest.fixture
def lingqu_backend():
    """创建灵衡后端"""
    config = {
        'lingqu': {
            'local_rank_id': 0,
            'etcd_url': 'http://localhost:2379',
            'host_swap_size': 2 * 1024**3,
            'device_swap_size': 128 * 1024**2,
            'enable_fabric_mem': True,
        }
    }
    return LingQuTieredBackend(config)

def test_put_get(lingqu_backend):
    """测试基本的Put/Get操作"""
    key = "test_key_1"
    data = b"Hello, LingQu!" * 1000

    # Put
    assert lingqu_backend.put(key, data)

    # Get
    retrieved = lingqu_backend.get(key)
    assert retrieved == data

def test_multi_tier(lingqu_backend):
    """测试多层级缓存"""
    key = "test_key_2"
    data = b"Multi-tier test" * 10000

    # 测试各层
    for tier in [0, 1, 2, 3]:
        assert lingqu_backend.put(key, data, tier=tier)
        retrieved = lingqu_backend.get(key)
        assert retrieved == data

        # 清理
        assert lingqu_backend.delete(key)

def test_large_data(lingqu_backend):
    """测试大数据传输"""
    key = "test_key_large"
    # 64MB数据
    data = np.random.bytes(64 * 1024**2)

    assert lingqu_backend.put(key, data)

    retrieved = lingqu_backend.get(key)
    assert retrieved == data

def test_performance(lingqu_backend):
    """性能测试"""
    sizes = [1024, 1024**2, 64*1024**2]  # 1KB, 1MB, 64MB

    for size in sizes:
        key = f"perf_test_{size}"
        data = np.random.bytes(size)

        import time
        start = time.time()

        assert lingqu_backend.put(key, data)
        retrieved = lingqu_backend.get(key)

        elapsed = time.time() - start
        latency_us = elapsed * 1e6
        bandwidth_gbps = (size * 8) / (elapsed * 1e9)

        print(f"Size={size} bytes, "
              f"Latency={latency_us:.2f} us, "
              f"Bandwidth={bandwidth_gbps:.2f} GB/s")

        # 性能断言
        if size < 1024**2:
            assert latency_us < 10.0  # <10μs for small data
        else:
            assert bandwidth_gbps > 40.0  # >40GB/s for large data
```

### 3.3 性能基准测试

```python
# mooncake-wheel/tests/benchmark_lingqu.py

import time
import numpy as np
import matplotlib.pyplot as plt
from mooncake.lingqu_backend import LingQuTieredBackend

def benchmark_throughput(backend, sizes, iterations=10):
    """吞吐量基准测试"""
    results = {}

    for size in sizes:
        latencies = []
        bandwidths = []

        for _ in range(iterations):
            key = f"bench_{size}_{_}"
            data = np.random.bytes(size)

            # 写入测试
            start = time.time()
            backend.put(key, data)
            write_time = time.time() - start

            # 读取测试
            start = time.time()
            retrieved = backend.get(key)
            read_time = time.time() - start

            total_time = write_time + read_time

            latencies.append(total_time * 1e6)  # 转换为微秒
            bandwidths.append((size * 2 * 8) / (total_time * 1e9))  # 读写双向

        results[size] = {
            'avg_latency_us': np.mean(latencies),
            'p99_latency_us': np.percentile(latencies, 99),
            'avg_bandwidth_gbps': np.mean(bandwidths),
        }

    return results

def plot_results(results):
    """绘制结果"""
    sizes = list(results.keys())
    latencies = [results[s]['avg_latency_us'] for s in sizes]
    bandwidths = [results[s]['avg_bandwidth_gbps'] for s in sizes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 延迟图
    ax1.plot(sizes, latencies, 'o-')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Size (bytes)')
    ax1.set_ylabel('Latency (us)')
    ax1.set_title('Transfer Latency')
    ax1.grid(True)

    # 带宽图
    ax2.plot(sizes, bandwidths, 's-')
    ax2.set_xscale('log')
    ax2.set_xlabel('Size (bytes)')
    ax2.set_ylabel('Bandwidth (GB/s)')
    ax2.set_title('Transfer Bandwidth')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('lingqu_benchmark.png')

if __name__ == '__main__':
    backend = LingQuTieredBackend({
        'lingqu': {
            'local_rank_id': 0,
            'etcd_url': 'http://localhost:2379',
            'host_swap_size': 2 * 1024**3,
            'device_swap_size': 128 * 1024**2,
            'enable_fabric_mem': True,
        }
    })

    sizes = [
        4 * 1024,          # 4KB
        64 * 1024,         # 64KB
        1024 * 1024,       # 1MB
        16 * 1024 * 1024,  # 16MB
        64 * 1024 * 1024,  # 64MB
        256 * 1024 * 1024, # 256MB
    ]

    results = benchmark_throughput(backend, sizes)

    print("\n=== Benchmark Results ===")
    for size, metrics in results.items():
        print(f"Size: {size:>12} bytes")
        print(f"  Avg Latency:  {metrics['avg_latency_us']:>8.2f} us")
        print(f"  P99 Latency:  {metrics['p99_latency_us']:>8.2f} us")
        print(f"  Avg Bandwidth: {metrics['avg_bandwidth_gbps']:>8.2f} GB/s")
        print()

    plot_results(results)
```

---

## 4. 性能调优

### 4.1 内存调优

```bash
# 1. 启用大页内存
echo 1024 > /proc/sys/vm/nr_hugepages
echo "vm.nr_hugepages = 1024" >> /etc/sysctl.conf

# 2. 配置NUMA策略
numactl --cpunodebind=0 --membind=0 ./your_application

# 3. 调整交换缓冲区大小
# 在配置文件中:
# host_swap_size: 根据并发度调整 (1-4GB)
# device_swap_size: 根据数据大小调整 (128-512MB)
```

### 4.2 网络调优

```bash
# 1. 调整RDMA参数
echo "options ib_uverbs max_sg_entries=4096" >> /etc/modprobe.d/ib_uverbs.conf

# 2. 增加QP队列深度
echo "net.core.rmem_max = 268435456" >> /etc/sysctl.conf
echo "net.core.wmem_max = 268435456" >> /etc/sysctl.conf

# 3. 启用多QP
# 在配置文件中:
# max_qp_count: 16  # 每个连接的最大QP数
```

### 4.3 传输优化

```cpp
// 批量传输配置
struct BatchConfig {
    size_t max_batch_size;       // 最大批量大小 (建议64MB)
    size_t max_batch_count;      // 最大批量数量 (建议32)
    bool enable_coalescing;      // 启用合并
    bool enable_pipelining;      // 启用流水线
};

// 预取配置
struct PrefetchConfig {
    size_t prefetch_distance;    // 预取距离 (建议4-8个token)
    bool enable_adaptive;        // 自适应预取
    double prefetch_threshold;   // 预取阈值 (命中率)
};
```

---

## 5. 故障排查

### 5.1 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| GVA分配失败 | 内存不足、对齐错误 | 检查内存配置、确保2MB对齐 |
| RH2D传输超时 | 网络问题、QP错误 | 检查RDMA状态、重新建立QP |
| 性能不达预期 | 路径选择错误、配置不当 | 检查路径选择器、优化配置 |
| 灵衡不可用 | 驱动未加载、配置错误 | 检查驱动、验证配置 |

### 5.2 调试工具

```bash
# 1. 查看GVA分配情况
python -m mooncake.tools.gva_inspector

# 2. 查看传输统计
python -m mooncake.tools.transport_stats

# 3. 性能分析
python -m mooncake.tools.profiler

# 4. 日志分析
tail -f /var/log/mooncake/lingqu.log | grep ERROR
```

### 5.3 性能分析

```python
# 性能分析脚本
import cProfile
import pstats

def profile_your_code():
    profiler = cProfile.Profile()
    profiler.enable()

    # 你的代码
    backend = LingQuTieredBackend(config)
    backend.put("key", data)
    result = backend.get("key")

    profiler.disable()

    # 输出统计
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime')
    stats.print_stats(20)
```

---

**文档结束**

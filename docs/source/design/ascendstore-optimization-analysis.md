# AscendStore 优化点识别与共创 -- Mooncake 多级缓存与元数据架构需求拆解

> 基于 Mooncake 最新代码（main 分支）分析  
> 编制：Qingjun Wang  
> 日期：2026-08-13  
> 工作量换算基准：3K LOC = 1 人月

---

## 1. 分析范围与代码基线

| 模块 | 关键文件 | 代码行数 |
|------|----------|----------|
| Master Service | master_service.cpp / .h | 2220 + 968 |
| Storage Backend | storage_backend.cpp / .h | 2574 + 982 |
| File Storage | file_storage.cpp / .h | 510 + 99 |
| Real Client | real_client.cpp / .h | 2331 + 555 |
| Master Client | master_client.cpp / .h | 777 + 458 |
| Posix File | posix_file.cpp | 133 |
| HA Helper | ha_helper.cpp / .h | 194 + 87 |
| HF3FS | hf3fs_file.cpp, hf3fs_resource_manager.cpp | ~400 |

---

## 2. 需求一：L4 共享盘存储缓存适配（仅昇腾）

### 2.1 现状分析（代码级）

当前 Mooncake SSD 存储能力存在明确的 Local SSD 限制：

**(1) Replica 层 -- 仅支持本地路径**

`replica.h:116-125` 定义了三种 Replica 类型：
- `MemoryReplicaData` -- 内存副本，含 `AllocatedBuffer`
- `DiskReplicaData` -- 磁盘副本，仅含 `file_path`（字符串路径）+ `object_size`
- `LocalDiskReplicaData` -- 本地磁盘副本，含 `client_id` + `transport_endpoint`

`DiskReplicaData.file_path` 是纯本地文件系统路径，无共享存储寻址能力。`LocalDiskReplicaData` 绑定了特定 `client_id`，数据必须通过该 client 的 RPC 服务访问，无法跨节点直接读取共享盘。

**(2) Storage Backend -- POSIX 文件 I/O，无共享存储抽象**

`posix_file.cpp` 封装了最基础的 POSIX I/O（`open`/`write`/`read`/`close`），无分布式文件系统接口抽象。

`storage_backend.cpp:95-97` 明确区分了 3FS 模式：
```cpp
#ifdef USE_3FS
    // Eviction is only enabled for local storage, not for 3FS
    return !is_3fs_dir_;
```
3FS（HF3FS）虽已集成（`hf3fs/` 目录），但编译时需 `USE_3FS` 开关且依赖 `hf3fs_api_shared` 库（`CMakeLists.txt:42-46`），且 3FS 模式下**禁用了 eviction**（`storage_backend.cpp:202-205`），意味着 3FS 当前仅作为无淘汰的写入路径，不具备完整的 L4 缓存能力。

**(3) Master Service -- root_fs_dir 为单一本地路径**

`master_config.h:33` 定义 `root_fs_dir`，`master_service.cpp:1429` 拼接路径为 `root_fs_dir_ + "/" + cluster_id_`，是单一本地目录，无共享盘挂载点管理。

**(4) Ascend 平台支持现状**

Ascend 传输层已集成（`mooncake-transfer-engine/include/transport/ascend_transport/`，含 `ascend_direct_transport`、`heterogeneous_rdma_transport`、`hccl_transport`）。`client_service.cpp:273-369` 支持 `protocol == "ascend"` 传输协议安装。`real_client.cpp:294-295` 支持 Ascend 内存段分配（`AscendSegmentDeleter` 调用 `free_memory("ascend", ptr)`）。

但 Ascend 平台的**磁盘存储路径**（L4 缓存）尚无适配，当前 POSIX 文件 I/O 未针对昇腾环境做优化。

### 2.2 需求拆解

| 子任务 | 描述 | 涉及文件 |
|--------|------|----------|
| 1A. 共享盘 Replica 类型扩展 | 新增 `SharedDiskReplicaData`，支持共享文件系统路径寻址 + 节点无关访问 | replica.h, types.h |
| 1B. StorageBackend 共享盘适配 | 抽象 `StorageFileInterface`，支持 NFS/共享盘后端，启用 eviction | storage_backend.h/.cpp, posix_file.cpp |
| 1C. Master 路径管理扩展 | `root_fs_dir` 支持多挂载点注册与容量管理 | master_service.cpp/.h, master_config.h |
| 1D. Ascend 平台 I/O 优化 | 针对昇腾环境适配 `posix_file` 的 I/O 路径（可选 aio/io_uring） | posix_file.cpp, 新增 ascend_file.cpp |
| 1E. 测试与验证 | 共享盘 L4 缓存端到端测试（put/get/evict/offload） | tests/ |

### 2.3 工作量估算

| 子任务 | 新增/修改代码（LOC） | 人月 |
|--------|---------------------|------|
| 1A. Replica 类型扩展 | ~600 | 0.20 |
| 1B. StorageBackend 适配 | ~2400 | 0.80 |
| 1C. Master 路径管理 | ~900 | 0.30 |
| 1D. Ascend I/O 优化 | ~1500 | 0.50 |
| 1E. 测试 | ~1200 | 0.40 |
| **合计** | **~6600** | **2.20** |

> 对齐需求标注：1.5~2 人，按 3K/人月换算约 2 人月，与估算吻合。

### 2.4 构建节奏

| 阶段 | 周期 | 交付件 |
|------|------|--------|
| P1: 设计与接口定义 | 第 1-2 周 | RFC 文档 + `SharedDiskReplicaData` 接口定义 + `StorageFileInterface` 抽象层 |
| P2: 核心实现 | 第 3-6 周 | StorageBackend 共享盘适配 + Master 多路径管理 + Ascend I/O 优化 |
| P3: 集成测试 | 第 7-8 周 | E2E 测试通过 + 性能基线（对比 Local SSD） |

---

## 3. 需求二：SSD 预取功能

### 3.1 现状分析（代码级）

**(1) 无任何 SSD 层预取机制**

全仓库搜索 `prefetch`/`Prefetch`/`readahead` 关键词，仅在 `cachelib_memory_allocator/AllocationClass.h:161-177` 找到 CPU L1 cache 级别的 `__builtin_prefetch`（用于 slab 遍历优化），与 SSD I/O 预取完全无关。

**(2) 当前 SSD 读取路径**

`FileStorage::BatchGet`（`file_storage.h:25-27`）→ `StorageBackend::BatchLoad`（`storage_backend.h:127-128`）→ `PosixFile::read`（同步 `::read` 系统调用）。全链路同步阻塞 I/O，无异步预读、无预测性加载。

**(3) IO bound 场景下的收益缺失**

当 KVCache miss 命中 SSD 层时，需同步从 SSD 读取数据到内存，再通过 TransferEngine 传输到目标节点。无预取意味着：
- 首次访问延迟 = SSD 读延迟 + 传输延迟（串行）
- 无法利用请求间的时序模式做预测性加载
- 在多轮对话场景（prefill→decode 切换）中，SSD 缓存命中率提升无法转化为 TTFT 改善

**(4) 已有社区进展**

需求标注"已有 draft PR 和社区 RFC"，说明社区已识别此问题并有初步方案。本工作需基于社区 draft PR 进行深化和昇腾适配。

### 3.2 需求拆解

| 子任务 | 描述 | 涉及文件 |
|--------|------|----------|
| 2A. 预取策略引擎 | 实现访问模式预测（LRU-aware / 序列感知 / 显式 hint） | 新增 prefetch_manager.h/.cpp |
| 2B. 异步 I/O 路径 | 将 `PosixFile::read` 升级为异步 I/O（io_uring / aio） | posix_file.cpp, 新增 async_file.cpp |
| 2C. StorageBackend 集成 | 在 `BatchGet`/`BatchLoad` 路径中集成预取触发与异步管道 | file_storage.cpp, storage_backend.cpp |
| 2D. Master 协同 | 预取 hint 从 Master 下发（基于全局访问模式）或 Client 本地决策 | master_service.cpp, real_client.cpp |
| 2E. 测试与基准 | 预取命中率、IO bound 场景 TTFT/TPS 基准 | tests/, benchmarks/ |

### 3.3 工作量估算

| 子任务 | 新增/修改代码（LOC） | 人月 |
|--------|---------------------|------|
| 2A. 预取策略引擎 | ~2100 | 0.70 |
| 2B. 异步 I/O 路径 | ~1800 | 0.60 |
| 2C. StorageBackend 集成 | ~1500 | 0.50 |
| 2D. Master 协同 | ~900 | 0.30 |
| 2E. 测试与基准 | ~1200 | 0.40 |
| **合计** | **~7500** | **2.50** |

> 对齐需求标注：2~3 人，按 3K/人月换算约 2.5 人月，与估算吻合。

### 3.4 构建节奏

| 阶段 | 周期 | 交付件 |
|------|------|--------|
| P1: 社区 RFC 对齐 + 设计 | 第 1-2 周 | 预取策略 RFC + 异步 I/O 接口设计 |
| P2: 核心实现 | 第 3-7 周 | 预取引擎 + 异步 I/O + StorageBackend 集成 |
| P3: 优化与测试 | 第 8-10 周 | IO bound 基准测试 + 预取命中率调优 |

---

## 4. 需求三：MooncakeStore 去中心化元数据架构

### 4.1 现状分析（代码级）

**(1) 单一中心化 Master 架构**

`MasterService` 是唯一的元数据中心。所有操作（`PutStart`/`PutEnd`/`GetReplicaList`/`MountSegment`/`Remove`/`Ping` 等）均通过 `MasterClient` → `coro_rpc` → `MasterService` 单点路径。

`master_client.h:20` 定义 `kDefaultMasterAddress = "localhost:50051"`，Client 连接单一地址。`MasterClient::RpcClientAccessor`（`master_client.h:422-441`）管理 RPC 连接池，但所有连接指向同一 Master。

**(2) 内部分片但不分布**

`master_service.h:655` 定义 `kNumShards = 1024`，元数据通过 `std::hash<std::string>{}(key) % kNumShards` 分片。但这是**单进程内的内存分片**（`std::array<MetadataShard, kNumShards> metadata_shards_`），不跨节点分布。

**(3) HA 模式为 Active-Standby**

`ha_helper.h:22-64` 定义 `MasterViewHelper`，通过 etcd leader election 实现 HA。`ha_helper.cpp:27-80` 的 `ElectLeader` 是竞争式选举：同一时刻只有一个 Master 作为 leader 服务。这是**故障转移式 HA**，不是横向扩展。

**(4) Client-Master 频繁交互**

Client 与 Master 的交互路径包括：
- `Ping`（心跳，TTL 续约）-- `master_service.h:307`
- `OffloadObjectHeartbeat`（offload 状态同步）-- `master_service.h:336-337`
- `FetchTasks`（任务拉取）-- `master_service.h:378-379`
- `MountSegment`/`ReMountSegment`（段注册）-- `master_service.h:58-73`
- 每次 `PutStart`/`GetReplicaList` 均需 Master 介入

在大集群（数百节点）场景下，单一 Master 的 RPC 吞吐、元数据内存占用、锁竞争均会成为瓶颈。

**(5) 现有缓解措施不足**

- `MetadataShard` 使用 `SharedMutex`（读写锁），但锁粒度仍为单进程
- `client_ping_queue_` 使用 `boost::lockfree::queue`（`master_service.h:901`），容量 `128 * 1024`，大集群下可能溢出
- Eviction 线程、Task cleanup 线程均为单线程，无法并行处理

### 4.2 需求拆解

此需求涉及 MooncakeStore V3 架构演进，是三个需求中复杂度最高的。提出两种可行路径：

#### 路径 A：层级化元数据架构（Hierarchical）

| 子任务 | 描述 |
|--------|------|
| 3A-1. Region Master 层 | 引入 Region Master，每个 Region Master 管理一组节点的元数据 |
| 3A-2. Global Coordinator | 全局协调器管理 Region 映射，处理跨 Region 操作 |
| 3A-3. Client 路由层 | Client 根据 key 前缀/哈希路由到对应 Region Master |
| 3A-4. 跨 Region 操作 | 跨 Region 的 replica 复制、迁移协议 |
| 3A-5. 元数据同步 | Region 间元数据最终一致性 / 强一致性协议选择 |

#### 路径 B：去中心化元数据架构（Distributed）

| 子任务 | 描述 |
|--------|------|
| 3B-1. 分布式元数据存储 | 基于 etcd/Redis 的元数据分片存储，Master 变为无状态代理 |
| 3B-2. 一致性哈希路由 | Client 直接通过一致性哈希定位元数据分片 |
| 3B-3. 分布式锁与事务 | 跨分片操作的分布式锁（etcd lease / Redis RedLock） |
| 3B-4. 缓存与回填 | Client 侧元数据缓存 + 失效通知（watch 机制） |
| 3B-5. 故障恢复 | 分片迁移、Rebalance、节点宕机处理 |

### 4.3 工作量估算

**路径 A（层级化，推荐先行验证）:**

| 子任务 | 新增/修改代码（LOC） | 人月 |
|--------|---------------------|------|
| 3A-1. Region Master | ~3000 | 1.00 |
| 3A-2. Global Coordinator | ~2400 | 0.80 |
| 3A-3. Client 路由层 | ~1800 | 0.60 |
| 3A-4. 跨 Region 操作 | ~2700 | 0.90 |
| 3A-5. 元数据同步 | ~2100 | 0.70 |
| 测试 | ~3000 | 1.00 |
| **合计** | **~15000** | **5.00** |

**路径 B（去中心化，长期演进）:**

| 子任务 | 新增/修改代码（LOC） | 人月 |
|--------|---------------------|------|
| 3B-1. 分布式元数据存储 | ~3600 | 1.20 |
| 3B-2. 一致性哈希路由 | ~1500 | 0.50 |
| 3B-3. 分布式锁与事务 | ~2400 | 0.80 |
| 3B-4. 缓存与回填 | ~2100 | 0.70 |
| 3B-5. 故障恢复 | ~3000 | 1.00 |
| 测试 | ~3600 | 1.20 |
| **合计** | **~16200** | **5.40** |

### 4.4 构建节奏

| 阶段 | 周期 | 交付件 |
|------|------|--------|
| P1: 架构 RFC + 原型验证 | 第 1-4 周 | V3 架构 RFC + 路径 A/B 对比原型 + 基准测试 |
| P2: 核心架构实现 | 第 5-16 周 | Region Master / 分布式元数据存储 + Client 路由 + 跨分片操作 |
| P3: 兼容性与迁移 | 第 17-20 周 | V2→V3 平滑迁移方案 + 兼容模式 |
| P4: 大规模测试 | 第 21-24 周 | 200+ 节点集群测试 + 性能基线 |

---

## 5. 总览

| 需求 | 代码量（LOC） | 人月 | 建议人数 | 周期 |
|------|-------------|------|---------|------|
| 1. L4 共享盘缓存适配 | ~6600 | 2.20 | 2 人 | 8 周 |
| 2. SSD 预取功能 | ~7500 | 2.50 | 2~3 人 | 10 周 |
| 3. 去中心化元数据架构 | ~15000~16200 | 5.0~5.4 | 5~6 人 | 24 周 |
| **合计** | **~29100~30300** | **9.7~10.1** | -- | -- |

---

## 6. 优先级与依赖关系

```
需求 1（L4 共享盘）  ──┐
                       ├──> 需求 2（SSD 预取）依赖 L4 存储层接口
                       │
需求 3（元数据架构） ──┘ 独立可并行，但 L4 扩容后元数据压力更大，需协同
```

**建议执行顺序:**
1. 需求 1 + 需求 2 可并行启动（不同团队/人员）
2. 需求 3 独立启动架构 RFC，但实现阶段需与需求 1 的存储接口对齐
3. 需求 2 的异步 I/O 路径（2B）可复用需求 1 的 Ascend I/O 优化成果（1D）

---

## 7. 风险与建议

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 共享盘 eviction 正确性 | 数据丢失 | 参考 3FS 模式的设计教训（当前 3FS 禁用 eviction），需设计共享盘专用的 eviction 协议 |
| 预取策略误判 | 内存浪费 / 性能回退 | 引入置信度阈值 + 自适应策略，支持运行时关闭 |
| V3 架构迁移风险 | 兼容性破坏 | 保留 V2 兼容模式，灰度迁移，双写过渡期 |
| 社区 RFC 对齐 | 重复工作 / 方向分歧 | 需求 2 的预取功能须基于社区 draft PR 深化，避免重复造轮子 |
| 昇腾环境测试覆盖 | 生产环境缺陷 | 需在真实昇腾集群验证，CI 增加 Ascend 专用测试矩阵 |

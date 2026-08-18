# AscendStore 优化点识别与共创 -- Mooncake 多级缓存与元数据架构需求拆解

> 基于 Mooncake 最新代码（main 分支）分析
> 编制：Qingjun Wang
> 日期：2026-08-13，更新：2026-08-14
> 工作量换算基准：3K LOC = 1 人月

---

## 1. 分析范围与代码基线

| 模块 | 关键文件 | 代码行数 |
|------|----------|----------|
| Master Service | master_service.cpp / .h | 2220 + 2786 |
| Storage Backend | storage_backend.cpp / .h | 2574 + 982 |
| Distributed Storage | distributed_storage_backend.h, dfs_global_allocator.h, posix_fs_adapter.h, fs_adapter.h, hf3fs_adapter.h, object_storage_adapter.h | ~600 |
| File Storage | file_storage.cpp / .h | 510 + 202 |
| Real Client | real_client.cpp / .h | 2331 + 555 |
| Master Client | master_client.cpp / .h | 777 + 458 |
| Replica | replica.h | 824 |
| Segment | segment.h | 602 |
| Local Hot Cache | local_hot_cache.h | 331 |
| NVMe KV Backend | nvme_kv_backend.h, nvme_kv_connector.h, nvme_kv_executor.h | ~300 |
| Device Abstraction | device/accelerator_device.h, accelerator_registry.h, runtime_accelerator.h, cuda_ipc_buffer.h | ~400 |
| HA Components | ha/ directory, ha_metric_manager.h, hot_standby_service.h, standby_state_machine.h, master_snapshot_manager.h | ~800 |

---

## 2. 需求一：L4 共享盘存储缓存适配（仅昇腾）

### 2.1 现状分析（代码级）

> ★ 重大更新：Mooncake main 分支已新增 `DfsReplicaData` + `DistributedStorageBackend` + `DfsGlobalAllocator`，L4 共享盘存储的框架级抽象**已存在**，但昇腾环境未穿刺验证。原报告中"无共享存储寻址能力"的结论已过时。

**(1) Replica 层 -- 已有分布式文件系统副本类型**

`replica.h:222-234` 定义了 `DistributedFSDescriptor` 和 `DfsReplicaData`：
- `DistributedFSDescriptor`：含 `file_path` + `offset` + `object_size` + `transport_endpoint`（可选），支持共享文件系统路径寻址
- `DfsReplicaData`：封装 `DistributedFSDescriptor`，用于分布式文件系统上的 KVCache 副本
- `Replica` 类构造函数（`replica.h:305-307`）支持 `DistributedFSDescriptor` 直接构造 DFS 副本
- `Replica::is_dfs_replica()`（`replica.h:429`）和 `get_dfs_descriptor()`（`replica.h:436-441`）提供类型检查和描述符访问

当前 Replica 类型体系（5 种）：
- `MemoryReplicaData` -- 内存副本（HBM/DRAM）
- `NoFReplicaData` -- NVMe-oF SSD 副本（带 buffer）
- `DiskReplicaData` -- 本地磁盘副本（file_path + object_size）
- `LocalDiskReplicaData` -- 本地磁盘副本（client_id + transport_endpoint）
- **`DfsReplicaData` -- 分布式文件系统副本（DistributedFSDescriptor）← 已存在**

**(2) DistributedStorageBackend -- 已有共享盘存储后端抽象**

`include/storage/distributed/distributed_storage_backend.h` 定义 `DistributedStorageBackend` 类：
- 支持两种存储模式：`DistributedStorageMode::kFileSystem`（分布式文件系统）和 `kObjectStorage`（对象存储）
- `BatchOffload()` 接口支持批量 offload 到分布式存储
- `IsEnableOffloading()` 控制是否启用 offload
- 配合 `FileSystemAdapter` 抽象接口（`fs_adapter.h`），`PosixFsAdapter`（`posix_fs_adapter.h`）为 POSIX 实现

`DfsGlobalAllocator`（`dfs_global_allocator.h`）提供分布式文件系统全局内存分配：
- `Allocate()` 返回 `DistributedFSDescriptor`（含 file_path + offset）
- 内部使用 `OffsetAllocator` 管理分配偏移
- 支持 `PendingEviction` 机制

**(3) NoFSegmentManager -- NVMe-oF SSD 段管理已独立**

`segment.h:529-598` 定义 `NoFSegmentManager`：
- 管理 NVMe-oF SSD 段（`NoFSegment`），独立于主 `SegmentManager`
- `ScopedNoFSegmentAccess` 提供 RAII 式段访问
- `GetMountedSegments()` 返回 `MountedNoFSegmentSnapshot`（含 segment_id + base + size + te_endpoint）
- `master_service.h:197-300` 中 `MountNoFSegment`/`ReMountNoFSegment`/`UnmountNoFSegment`/`GetAllNoFSegments` 已集成到 MasterService

**(4) Master Service -- 已有分布式存储集成和外部元数据服务**

`master_service.h:67` 前向声明 `DfsGlobalAllocator`，`master_service.h:2571` 持有 `std::unique_ptr<DfsGlobalAllocator> dfs_allocator_` 成员。

`master_service.h:2545-2562` 已集成 `HttpMetadataServer`：
- `http_metadata_server_` 指针，支持外部 HTTP 元数据服务
- `http_metadata_remote_`（`MetadataStoragePlugin`）远程元数据客户端
- `http_metadata_cleanup_thread_` 清理线程
- `master.yaml` 中 `enable_http_metadata_server` + `http_metadata_server_port` 配置项

**(5) Ascend 平台支持现状**

Ascend 传输层已集成（`mooncake-transfer-engine/include/transport/ascend_transport/`）。`real_client.cpp:294-295` 支持 Ascend 内存段分配。

`include/device/` 目录提供设备抽象框架：
- `accelerator_device.h` / `accelerator_registry.h` -- 加速器设备抽象与注册
- `runtime_accelerator.h` -- 运行时加速器接口
- `cuda_ipc_buffer.h` -- CUDA IPC buffer（GPU 直接访问）
- AscendCacheTier 是 `CacheTier` 的昇腾实现（`-DUSE_ASCEND_CACHE_TIER=ON`）

**但昇腾平台的 L4 分布式存储路径（`DistributedStorageBackend` + `DfsGlobalAllocator`）尚未穿刺验证。**

**(6) 上游 PR 进展**

| PR | 状态 | 内容 |
|-----|------|------|
| #3427 | open | Extract LocalSSD management from SegmentManager -- 将 LocalSSD 运行时状态独立为 `LocalSsdManager` |
| #3467 | open | Bucket: add MAX_PHYSICAL_BYTES cap on real shared-disk usage -- 按实际磁盘用量（stat.st_blocks）限制共享盘用量 |
| #3491 | open | Add GDS offload for mooncake-store -- 基于 GDS transport 直接存储访问 |
| #3479 | open | fix(store): deduplicate SSD carryover keys -- 修复 `GroupOffloadingKeysByBucket()` carryover key 重复 |
| #3488 | open | perf(store): batch io_uring bucket reads -- `UringFile::batch_read()` 批量对齐读取 |

### 2.2 需求拆解（更新后）

| 子任务 | 描述 | 上游现状 | InferNex 适配 |
|--------|------|----------|-------------|
| 1A. ~~共享盘 Replica 类型扩展~~ | ~~新增 SharedDiskReplicaData~~ | ★ **已存在** `DfsReplicaData` + `DistributedFSDescriptor`（`replica.h:222-234`） | 无需新增，直接使用 |
| 1B. StorageBackend 共享盘适配 | 抽象共享盘后端，启用 eviction | ★ **已存在** `DistributedStorageBackend`（kFileSystem/kObjectStorage）+ `DfsGlobalAllocator` + `PosixFsAdapter` | 昇腾环境穿刺验证 + eviction 策略适配 |
| 1C. Master 路径管理扩展 | 多挂载点注册与容量管理 | ☆ `DfsGlobalAllocator` 已在 MasterService 中集成（`dfs_allocator_`），但 root_fs_dir 仍为单一配置 | 评估多挂载点需求，可能需扩展配置 |
| 1D. Ascend 平台 I/O 优化 | 昇腾环境 I/O 路径优化 | ☆ `posix_file.cpp` 基础 POSIX I/O，PR #3488 batch io_uring 可参考 | 评估 io_uring/aio 在昇腾环境的兼容性 |
| 1E. cache-indexer L3 索引扩展 | L3 索引增加 SSD/DFS 层查询 | ☆ 当前 cache-indexer 轮询 Mooncake Master `/get_all_keys`（不含 SSD 层 key） | 需评估 Master 是否返回 DFS 副本信息 |
| 1F. 测试与验证 | 共享盘 L4 缓存端到端测试 | ☆ 无昇腾环境测试 | E2E 测试（put/get/evict/offload on DFS） |

### 2.3 工作量估算（更新后）

| 子任务 | 新增/修改代码（LOC） | 人月 |
|--------|---------------------|------|
| 1A. ~~Replica 类型扩展~~ | ~~已存在，0~~ | 0 |
| 1B. StorageBackend 昇腾适配 | ~600（仅昇腾适配+eviction策略） | 0.20 |
| 1C. Master 配置扩展 | ~300（多挂载点配置） | 0.10 |
| 1D. Ascend I/O 优化 | ~1500（io_uring适配） | 0.50 |
| 1E. cache-indexer L3 扩展 | ~600（SSD/DFS层查询适配） | 0.20 |
| 1F. 测试 | ~1200 | 0.40 |
| **合计** | **~4200** | **1.40** |

> ★ 工作量从原估算 2.20 人月下调至 1.40 人月（上游框架已存在，仅需昇腾适配和验证）。
> 对齐需求标注：1.5~2 人，按 3K/人月换算约 0.5~0.67 人月 -- 上游框架成熟度超出预期，实际工作量集中在昇腾适配和测试验证。

### 2.4 构建节奏（更新后）

| 阶段 | 周期 | 交付件 |
|------|------|--------|
| P1: 昇腾环境穿刺 | 第 1-2 周 | `DistributedStorageBackend` + `DfsGlobalAllocator` 在昇腾环境编译+运行验证 |
| P2: eviction 策略适配 | 第 3-4 周 | 共享盘 eviction 协议适配（参考 3FS 禁用 eviction 的教训） |
| P3: cache-indexer L3 扩展 | 第 5-6 周 | L3 索引增加 DFS 副本查询 |
| P4: 集成测试 | 第 7-8 周 | E2E 测试 + 性能基线（对比 Local SSD） |

---

## 3. 需求二：SSD 预取功能

### 3.1 现状分析（代码级，更新后）

> ★ 重大更新：上游已有 RFC #3417 + draft PR #2646，预取框架**已存在 draft 实现**。

**(1) 上游 RFC 与 draft PR**

| 编号 | 类型 | 内容 |
|------|------|------|
| RFC #3417 | RFC | Explicit SSD->DRAM Prefetch Trigger -- router 在请求进入引擎前显式触发 SSD->DRAM 搬移，将 IO 移出 `get` 关键路径。数据落入专用 transit buffer（按 pipeline depth 分配），enqueue 时授予 lease。窗口错过或内存压力超限则回退到当前 SSD 读。 |
| PR #2646 | draft PR | Prefetch SSD-Only Objects to DRAM on Exist -- `is_exist`/`batch_is_exist` 调用时 `ExistOptions.prefetch_to_memory=true`，异步提升 SSD-only keys（LOCAL_DISK，无 MEMORY）到 DRAM。核心变更：专用 prefetch RPC 路径（`GetReplicaListForPrefetch`/`BatchGetReplicaListForPrefetch`/`RegisterPrefetchTask`），chunked batch query（128 keys/chunk），bounded prefetch_pool_（4 线程），PrefetchThrottle（dedup TTL + DRAM-pressure cooldown）。 |

**(2) 当前 SSD 读取路径（file_storage.h 代码级）**

`FileStorage`（`file_storage.h:14-192`）封装了完整的 SSD offload 路径：
- `OffloadObjects()` -- 异步 offload 对象到 SSD
- `BatchLoad()`（`file_storage.h:164`）-- 同步批量加载
- `AllocateBatch()` / `LoadBatch()` -- 批量分配和加载接口
- `RunDiskWatermarkEviction()` -- 水位驱动 eviction
- `NotifyEvictedDiskReplicas()` -- eviction 通知
- `ReRegisterOffloadedObjects()` -- Master 重启后 SSD 元数据同步
- `IsPerBucketSoftOffloadError()` -- per-bucket 软错误分类

`LocalDiskSegment`（`segment.h:90-115`）维护 SSD offload 状态：
- `offloading_objects` -- 待 offload 对象映射
- `promotion_objects` -- 待提升对象队列（`TryPushPromotionQueue` 在 get 命中 LOCAL_DISK-only key 时触发）
- `pending_remove_all` -- 全量 SSD 清除标志

**(3) 已有的提升机制（promotion-on-hit）**

`segment.h:98-102` 中 `promotion_objects` 队列已实现"命中时提升"：当 `get` 命中 LOCAL_DISK-only key 时，`TryPushPromotionQueue` 将其加入提升队列。但这是**同步被动提升**（命中后才触发），非**异步预取**（请求到达前预加载）。

**(4) IO bound 场景下的收益缺失**

当前提升路径（命中时触发）无法消除首次访问的 SSD 读延迟。RFC #3417 的核心价值在于将预取从"命中后被动提升"变为"路由器主动预取"，将 IO 移出 `get` 关键路径。

### 3.2 需求拆解（更新后）

| 子任务 | 描述 | 上游现状 | InferNex 适配 |
|--------|------|----------|-------------|
| 2A. 预取策略引擎 | 访问模式预测 + 显式 hint | ☆ RFC #3417 提出 router 显式触发；PR #2646 实现 `is_exist` 触发 | 对接 InferNex cache-indexer：router 是第一个知道请求将命中 SSD 的组件，可触发 prefetch |
| 2B. 异步 I/O 路径 | io_uring / aio 异步预读 | ☆ `file_storage.h` 已有 `BatchLoad`/`AllocateBatch`/`LoadBatch` 同步接口；PR #3488 `UringFile::batch_read()` 批量异步读取 | 评估 PR #3488 在昇腾环境兼容性 |
| 2C. StorageBackend 集成 | prefetch 触发与异步管道 | ☆ PR #2646 实现了专用 prefetch RPC 路径 | 评估 PR #2646 合入状态，适配昇腾 |
| 2D. Master 协同 | 全局访问模式下发 | ☆ PR #2646 支持 `prefetch_offload_object` RPC 跨节点 holder 委托 | 评估是否需 Master 全局模式感知 |
| 2E. 测试与基准 | 预取命中率 + IO bound TTFT/TPS | ☆ PR #2646 含测试 | 昇腾环境基准 |

### 3.3 工作量估算（更新后）

| 子任务 | 新增/修改代码（LOC） | 人月 |
|--------|---------------------|------|
| 2A. 预取策略对接 | ~600（对接 cache-indexer + router 触发） | 0.20 |
| 2B. 异步 I/O 适配 | ~900（PR #3488 昇腾适配） | 0.30 |
| 2C. PR #2646 适配 | ~1200（昇腾适配+集成） | 0.40 |
| 2D. Master 协同 | ~300（可选全局模式） | 0.10 |
| 2E. 测试与基准 | ~1200 | 0.40 |
| **合计** | **~4200** | **1.40** |

> ★ 工作量从原估算 2.50 人月下调至 1.40 人月（上游已有 RFC + draft PR + BatchLoad 接口，核心工作变为对接和昇腾适配）。
> 对齐需求标注：2~3 人，按 3K/人月换算约 0.67~1.0 人月 -- 上游 RFC 和 draft PR 成熟度超出预期。

### 3.4 构建节奏（更新后）

| 阶段 | 周期 | 交付件 |
|------|------|--------|
| P1: 社区 RFC 对齐 + PR 评估 | 第 1-2 周 | RFC #3417 + PR #2646 评估报告 + 昇腾适配方案 |
| P2: 核心适配 | 第 3-5 周 | cache-indexer 触发 + 异步 I/O 昇腾适配 + PR #2646 集成 |
| P3: 优化与测试 | 第 6-8 周 | IO bound 基准测试 + 预取命中率调优 |

---

## 4. 需求三：MooncakeStore 去中心化元数据架构

### 4.1 现状分析（代码级，更新后）

> ★ 重大更新：上游已有 `HttpMetadataServer` + `MetadataStoragePlugin` + `NoFSegmentManager` + `MasterSnapshotManager` + `hot_standby_service` + `enable_oplog`，分层元数据和 HA 的基础设施**已初步存在**。

**(1) 外部元数据服务（HttpMetadataServer）**

`master_service.h:2545-2562`：
- `HttpMetadataServer* http_metadata_server_` -- 外部 HTTP 元数据服务指针
- `MetadataStoragePlugin http_metadata_remote_` -- 远程元数据存储插件
- `http_metadata_prefix_` -- HTTP 元数据 key 前缀
- `http_metadata_cleanup_thread_` / `http_metadata_cleanup_queue_` -- 清理线程和队列
- `setHttpMetadataServer()` / `setHttpMetadataRemoteUrl()` -- 配置接口
- `master.yaml`: `enable_http_metadata_server: false` + `http_metadata_server_port: 8080`

**意义：Master 已支持将元数据存储卸载到外部 HTTP 服务，为去中心化元数据提供基础设施。**

**(2) NoFSegmentManager 独立段管理**

`segment.h:529-598` + `master_service.h:197-300`：
- `NoFSegmentManager` 独立于 `SegmentManager`，管理 NVMe-oF SSD 段
- MasterService 中 `nof_segment_manager_` 成员独立于主 segment 管理
- `MountNoFSegment` / `ReMountNoFSegment` / `UnmountNoFSegment` / `GetAllNoFSegments` 独立 RPC 接口
- `NoFHeartbeatState` + `nof_heartbeat_thread_` 独立心跳管理
- `NoFBatchEvict` 独立 eviction 路径

**意义：SSD 段管理已从主 SegmentManager 中分离，为层级化元数据管理提供初步基础。**

**(3) HA 热备与快照**

新增头文件和组件：
- `hot_standby_service.h` -- 热备服务
- `standby_state_machine.h` -- 备机状态机
- `master_snapshot_manager.h` / `master_snapshot_repository.h` -- Master 快照管理和存储
- `master_service.h:2247-2277` `MetadataSerializer` -- 元数据序列化/反序列化（支持 fork serialize）
- `master.yaml`: `enable_oplog: false` + `oplog_poll_interval_ms: 1000` -- OpLog HA 热备复制
- PR #3493 -- simplify P2P client HA recovery for Redis HA mode

**意义：HA 已从简单的 etcd Active-Standby 升级为支持 OpLog 复制和快照恢复的完整热备体系。**

**(4) 元数据分片（不变）**

`master_service.h:1574` `kNumShards = 1024`，`std::array<MetadataShard, kNumShards> metadata_shards_`，仍为单进程内分片。但 `MetadataAccessorRW`/`MetadataAccessorRO`（`master_service.h:2054-2348`）提供了完整的 RAII 式元数据访问封装，为未来跨进程分片提供了接口基础。

**(5) 分布式锁与租约**

- `master_service.h:891-906` `setHttpMetadataServer()` / `setHttpMetadataRemoteUrl()` 支持远程元数据
- `deadline_scheduler.h` -- 截止时间调度器
- `k8s_lease_helper.h` -- K8s lease 辅助
- `pinned_buffer_pool.h` / `pinned_host_buffer.h` -- 缓冲区固定与租约

**(6) 上游 RFC 与社区进展**

| 编号 | 类型 | 内容 |
|------|------|------|
| RFC #2117 | RFC | Hierarchical Arch for Intra/Inter Data Center -- AntGroup 10,000 卡规模 Master 瓶颈实践 |
| Issue #3452 | Bug | mooncake-master OOM restart under long-context PD-separation |
| Issue #1883 | Roadmap | Milestone 2: Store V3 Evolution (TE & Store Decoupling) |
| PR #3493 | open | simplify P2P client HA recovery for Redis HA mode |

### 4.2 需求拆解（更新后）

#### 路径 A：层级化元数据架构（Hierarchical）

| 子任务 | 描述 | 上游现状 |
|--------|------|----------|
| 3A-1. Region Master 层 | 引入 Region Master | ☆ `HttpMetadataServer` + `MetadataStoragePlugin` 可作为 Region 元数据服务的实现基础 |
| 3A-2. Global Coordinator | 全局协调器 | ☆ 当前 etcd leader election 可复用 |
| 3A-3. Client 路由层 | key 前缀/哈希路由 | ☆ `MasterClient` 当前连接单一地址，需扩展路由 |
| 3A-4. 跨 Region 操作 | 跨 Region replica | ☆ `DfsGlobalAllocator` + `DistributedStorageBackend` 可支撑跨 Region 存储 |
| 3A-5. 元数据同步 | Region 间一致性 | ☆ `MetadataSerializer` + `master_snapshot_manager` 提供快照序列化基础 |

#### 路径 B：去中心化元数据架构（Distributed）

| 子任务 | 描述 | 上游现状 |
|--------|------|----------|
| 3B-1. 分布式元数据存储 | Master 变为无状态代理 | ★ `HttpMetadataServer` + `MetadataStoragePlugin` 已支持外部元数据存储 |
| 3B-2. 一致性哈希路由 | Client 直连元数据分片 | ☆ 需新增 Client 路由层 |
| 3B-3. 分布式锁与事务 | 跨分片操作 | ☆ `deadline_scheduler` + `k8s_lease_helper` 可参考 |
| 3B-4. 缓存与回填 | Client 侧缓存 | ☆ `local_hot_cache.h` 的 LRU 机制可参考 |
| 3B-5. 故障恢复 | 分片迁移 | ★ `hot_standby_service` + `master_snapshot_manager` + `enable_oplog` 已提供 HA 基础 |

### 4.3 工作量估算（更新后）

**路径 A（层级化）：**

| 子任务 | 新增/修改代码（LOC） | 人月 |
|--------|---------------------|------|
| 3A-1. Region Master | ~1800（基于 HttpMetadataServer 扩展） | 0.60 |
| 3A-2. Global Coordinator | ~1500（基于 etcd 扩展） | 0.50 |
| 3A-3. Client 路由层 | ~1200 | 0.40 |
| 3A-4. 跨 Region 操作 | ~1800 | 0.60 |
| 3A-5. 元数据同步 | ~1200（基于 MetadataSerializer） | 0.40 |
| 测试 | ~1800 | 0.60 |
| **合计** | **~9300** | **3.10** |

**路径 B（去中心化）：**

| 子任务 | 新增/修改代码（LOC） | 人月 |
|--------|---------------------|------|
| 3B-1. 分布式元数据存储 | ~600（HttpMetadataServer 已存在，仅需配置） | 0.20 |
| 3B-2. 一致性哈希路由 | ~1500 | 0.50 |
| 3B-3. 分布式锁与事务 | ~1800 | 0.60 |
| 3B-4. 缓存与回填 | ~1200（参考 local_hot_cache） | 0.40 |
| 3B-5. 故障恢复 | ~600（hot_standby + oplog 已存在） | 0.20 |
| 测试 | ~2400 | 0.80 |
| **合计** | **~8100** | **2.70** |

> ★ 工作量从原估算 5.0~5.4 人月下调至 2.7~3.1 人月（上游已有 HttpMetadataServer / hot_standby / oplog / MetadataSerializer 基础设施）。

### 4.4 构建节奏（更新后）

| 阶段 | 周期 | 交付件 |
|------|------|--------|
| P1: 架构 RFC + 上游对齐 | 第 1-3 周 | V3 架构 RFC + 路径 A/B 对比 + 上游 HttpMetadataServer 评估 |
| P2: 核心架构实现 | 第 4-12 周 | Region Master / 分布式元数据 + Client 路由 + 跨分片操作 |
| P3: 兼容性与迁移 | 第 13-16 周 | V2->V3 兼容模式 |
| P4: 大规模测试 | 第 17-20 周 | 200+ 节点集群测试 |

---

## 5. 总览（更新后）

| 需求 | 代码量（LOC） | 人月 | 建议人数 | 周期 | 原估算 | 变化 |
|------|-------------|------|---------|------|-------|------|
| 1. L4 共享盘缓存适配 | ~4200 | 1.40 | 1.5~2 人 | 8 周 | 2.20 | ★ -36%（上游框架已存在） |
| 2. SSD 预取功能 | ~4200 | 1.40 | 2 人 | 8 周 | 2.50 | ★ -44%（RFC + draft PR 已存在） |
| 3. 去中心化元数据架构 | ~8100~9300 | 2.7~3.1 | 3 人 | 20 周 | 5.0~5.4 | ★ -44%（HA + 元数据基础设施已存在） |
| **合计** | **~16500~17700** | **5.5~5.9** | -- | -- | 9.7~10.1 | **★ -42%** |

---

## 6. 优先级与依赖关系（不变）

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

## 7. 风险与建议（更新后）

| 风险 | 影响 | 缓解措施 | 状态变化 |
|------|------|----------|---------|
| 共享盘 eviction 正确性 | 数据丢失 | 参考 3FS 模式教训；`DistributedStorageBackend` 已有 `IsEnableOffloading` 控制 | ★ 上游已有框架，风险降低 |
| 预取策略误判 | 内存浪费 / 性能回退 | RFC #3417 设计了 lease + 回退机制；PR #2646 有 PrefetchThrottle | ★ 上游已有设计，风险降低 |
| V3 架构迁移风险 | 兼容性破坏 | `HttpMetadataServer` + `MetadataStoragePlugin` 支持渐进式卸载；`enable_http_metadata_server` 默认关闭 | ★ 上游已有兼容路径 |
| 社区 RFC 对齐 | 重复工作 / 方向分歧 | 需求 2 基于 RFC #3417 + PR #2646 深化；需求 3 基于 RFC #2117 | ★ 上游 RFC 已对齐 |
| 昇腾环境测试覆盖 | 生产环境缺陷 | 需在真实昇腾集群验证 | ☆ 不变，仍需昇腾实机测试 |
| 上游 PR 合入时间 | 适配依赖未合入 | PR #3427/#3467/#3488/#3491/#2646 均 open 状态 | ☆ 新增风险：需跟踪合入时间 |

---

## 8. 上游社区最新代码分析附录（新增）

### 8.1 Mooncake main 分支新增能力（vs 原报告基线）

| 能力 | 关键文件/PR | 原报告结论 | 最新代码结论 |
|------|-----------|----------|------------|
| 分布式文件系统副本 | `replica.h:222-234` `DfsReplicaData` + `DistributedFSDescriptor` | "无共享存储寻址能力" | ★ **已存在**，含 file_path + offset + object_size + transport_endpoint |
| 分布式存储后端 | `distributed_storage_backend.h` `DistributedStorageBackend` | "POSIX 文件 I/O，无共享存储抽象" | ★ **已存在**，kFileSystem/kObjectStorage 两种模式 |
| 分布式 FS 分配器 | `dfs_global_allocator.h` `DfsGlobalAllocator` | 未提及 | ★ **已存在**，`Allocate()` 返回 `DistributedFSDescriptor` |
| POSIX FS 适配器 | `posix_fs_adapter.h` `PosixFsAdapter` | "posix_file.cpp 封装最基础 POSIX I/O" | ★ **已抽象**为 `FileSystemAdapter` 接口实现 |
| NVMe-oF 段管理 | `segment.h:529-598` `NoFSegmentManager` | 未提及 | ★ **已存在**，独立于主 SegmentManager |
| 外部元数据服务 | `master_service.h:2545-2562` `HttpMetadataServer` | "单一中心化 Master" | ★ **已支持外部元数据**，`enable_http_metadata_server` 配置 |
| 元数据存储插件 | `MetadataStoragePlugin` | 未提及 | ★ **已存在**，`http_metadata_remote_` 远程元数据客户端 |
| HA 热备 | `hot_standby_service.h` + `enable_oplog` | "Active-Standby via etcd" | ★ **已升级**为 OpLog 复制 + 快照恢复 |
| Master 快照 | `master_snapshot_manager.h` / `master_snapshot_repository.h` | 未提及 | ★ **已存在** |
| 元数据序列化 | `master_service.h:2247-2277` `MetadataSerializer` | 未提及 | ★ **已存在**，支持 fork serialize |
| SSD 预取 RFC | RFC #3417 + PR #2646 | "无任何预取机制" | ★ **已有 RFC + draft PR**，专用 prefetch RPC 路径 + PrefetchThrottle |
| SSD 批量读取 | PR #3488 `UringFile::batch_read()` | "全链路同步阻塞 I/O" | ★ **已有批量异步读取** draft PR |
| SSD 去重 | PR #3479 | 未提及 | ★ **已有修复** PR |
| 本地热点缓存 | `local_hot_cache.h` `LocalHotCache` | 未提及 | ★ **已存在**（InferNex 贡献上游），LRU + 16MB 块 + LOCAL_MEMCPY |
| 设备抽象框架 | `device/accelerator_device.h` + `runtime_accelerator.h` | 未提及 | ★ **已存在**，AscendCacheTier 为昇腾实现 |
| GPU 直接访问 | `device/cuda_ipc_buffer.h` | 未提及 | ★ **已存在**，CUDA IPC buffer |

### 8.2 InferNex 上游贡献在 Mooncake main 中的状态

| InferNex 贡献 | Mooncake 中的位置 | 状态 |
|-------------|----------------|------|
| PR 2092 Dual RDMA 前向路径 | Store V3 双向传输 | ★ 已合入 |
| PR 2407 per-RPC 细粒度指标 | `client_metric.h` + `p2p_client_metric.h` | ★ 已合入 |
| PR 2429 Ascend DRAM Tier 适配 | `dram_tier.h` / `dram_tier.cpp` | ★ 已合入 |
| PR 1688 RealClient 压测 | `tests/real_client_stress_workload.py` | ☆ open |
| PR 2436 跨进程 P2P 测试 | `tests/peer_client_test.cpp` | ☆ open |
| AscendCacheTier (VRAM) | `USE_ASCEND_CACHE_TIER=ON` | ★ 已合入 |
| LocalHotCache 热点缓存 | `local_hot_cache.h` (331 lines) | ★ 已合入 |

# AscendStore 优化点识别与共创 -- Mooncake 多级缓存与元数据架构需求拆解

> 基于 Mooncake 最新代码（main 分支）分析
> 编制：Qingjun Wang
> 日期：2026-08-13，更新：2026-08-14
> 工作量换算基准：3K LOC = 1 人月
> 标注说明：★ = 上游已有实现或已合入；☆ = 需新增构建；◎ = 与已有条目有重复

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
| Pinned Buffer | pinned_buffer_pool.h, pinned_host_buffer.h, aligned_client_buffer.h, client_buffer.h | ~400 |
| K8s Lease | k8s_lease_helper.h | ~60 |
| Deadline Scheduler | deadline_scheduler.h | ~100 |

---

## 2. 需求一：Layerwise 池化传输与 DSA KV Offload 接口诉求

> 诉求来源：vLLM-Ascend 对 Mooncake 能力的诉求。当前为实现高性能分层 D2RH、RH2D 传输，主要涉及 Mooncake Layerwise 池化传输和 DSA KV Offload 场景，对 Mooncake 接口有新诉求。

### 2.1 现状分析（代码级）

**(1) 上游已有 Layerwise Session API（RFC #2887 + PR #2881）**

Mooncake main 分支已合入 Layerwise KV Cache Session-Based Ranged Transfer：
- RFC #2887 提出基于 session 的 ranged transfer API：Master 元数据查询一次，后续逐层传输不再重复查询
- PR #2881 实现 Store session APIs：`batch_get_session_start` -> `batch_get_into_multi_buffer_ranges` -> `batch_get_session_end` / `batch_put_session_start` -> `batch_put_from_multi_buffer_ranges` -> `batch_put_session_end` / `batch_put_session_revoke`
- Get session 缓存 `QueryResult`（含完整 memory replica），range 路径仅用缓存（无 Master RPC）
- Session ranged transfers 通过 `Client::BatchTransferReadRanges` / `BatchTransferWriteRanges`

**(2) 上游已有 Host Buffer 分配 API（PR #3019）**

PR #3019 添加 `mooncake.engine.allocate_host_buffer` / `free_host_buffer`：
- 为 decode offload + ACL graph capture 提供固定 host buffer 地址
- Ascend 构建（USE_ASCEND_DIRECT）：`ascend_allocate_memory`（fabric mem 或 `aclrtMallocHost`）
- 非 Ascend 构建：`aligned_alloc(4096, ...)` / `free`
- 无状态 API，不注册 TE，调用方需单独 `register_memory`

**(3) 上游已有 Shared Host Segment RFC（RFC #3249）**

RFC #3249 提出 `create_shared_segment` API：one-writer-many-reader KV 布局（MLA sparse KV decode offload），TP 组内 KV payload 相同，单 rank 写入 + 全组读取，避免 per-rank 内存浪费（TP16 下节省 16x）。

**(4) 上游已有 GDS Offload 支持（Issue #2731 + PR #3491）**

Issue #2731 描述 GDS（GPU Direct Storage）架构，支持 standalone-store 模式下 SSD offload 直接到 GPU/NPU。

### 2.2 需求拆解

| 子任务 | 描述 | 上游现状 | InferNex 适配 | 工作量（人月） |
|--------|------|----------|-------------|-------------|
| 1A. batch_alloc_buffer() | 手动申请 DRAM BUFFER | ★ PR #3019 `allocate_host_buffer` 已合入；`pinned_buffer_pool.h` / `aligned_client_buffer.h` 已有 buffer 分配框架 | ☆ 昇腾环境适配验证（aclrtMallocHost 路径） | 0.30 |
| 1B. batch_get_key_info() | 查询命中时直接获取 key 对应 GVA 地址信息 | ★ PR #2881 session API 已合入 `batch_get_session_start` 缓存 QueryResult（含 replica descriptor + GVA 地址）；`replica.h:222` DistributedFSDescriptor 含 transport_endpoint | ☆ 昇腾 GVA 地址映射适配 | 0.40 |
| 1C. batch_copy_with_gva() | 直接使用 GVA 地址进行 D2RH、RH2D 拷贝 | ☆ PR #2881 `batch_get_into_multi_buffer_ranges` 支持 ranged transfer，但未直接暴露 GVA-based copy 接口；`device/cuda_ipc_buffer.h` 有 GPU IPC 机制但非 Ascend | ☆ 需新增 Ascend GVA-based batch copy 接口 | 0.60 |
| 1D. batch_copy_finished() | 结束拷贝 | ☆ session API 的 `batch_get_session_end` / `batch_put_session_end` 已有 finalize 语义 | ☆ 适配 GVA copy 的 finalize | 0.20 |
| 1E. batch_add_lease() / batch_remove_lease() | 对 key 添加/释放租约 | ☆ `deadline_scheduler.h` + `pinned_buffer_pool.h` 有 lease 相关机制；session API 的 `batch_put_session_revoke` 有类似语义 | ☆ 需新增 key-level lease batch API | 0.40 |
| 1F. 端到端集成测试 | Layerwise 池化传输 + DSA Offload 端到端验证 | ☆ 无昇腾环境 E2E 测试 | ☆ 昇腾环境 E2E | 0.30 |
| **合计** | | | | **~2.20** |

> 对齐需求标注：~3 人月。上游已有 session API + allocate_host_buffer + shared segment RFC，实际工作集中在 GVA-based batch copy 接口新增和昇腾适配，估算 2.2 人月，偏差在合理范围。

### 2.3 构建节奏

| 阶段 | 周期 | 交付件 |
|------|------|--------|
| P1: 上游 API 评估 + 昇腾适配 | 第 1-3 周 | session API / allocate_host_buffer 昇腾环境验证 + GVA 地址映射方案 |
| P2: GVA batch copy 接口 | 第 4-7 周 | batch_copy_with_gva / batch_copy_finished / batch_add_lease 接口实现 |
| P3: 端到端测试 | 第 8-10 周 | Layerwise + DSA Offload E2E + 性能基线 |

---

## 3. 需求二：MooncakeStore 高可用加固

> 诉求来源：当前 MooncakeStore 高可用模式能力较不成熟，需要加固以支撑商用。

### 3.1 现状分析（代码级）

**(1) HA 模式基于 ETCD，K8s 原生支持不成熟**

当前 HA 通过 etcd leader election 实现（`ha_helper.h` Active-Standby）。上游 Issue #2643 指出：etcd 和 K8s HA 后端互斥构建（两个不同 Go 模块），导致 pip 包需按 HA 后端分版本，K8s HA 后端难以与 SGLang/vLLM 集成。Issue #1856 提交 K8s-native Leader Election PR（基于 `coordination.k8s.io/v1` Lease），已拆分为 Go shared library（PR #1910）+ C++ coordinator（PR #1956），但尚未合入。

**(2) Master 主备切换不支持恢复元数据**

RFC #1150 明确指出：Master HA 只能保证"快速启动"但无法解决缓存数据连续性问题。主备切换后内存池数据信息丢失。当前 `master_snapshot_manager.h` / `master_snapshot_repository.h` 提供快照管理框架，`MetadataSerializer` 提供序列化能力，但端到端恢复流程未闭环。Issue #2971 指出 etcd OpLog 隐式启用但缺乏 retention 和完整元数据复制。Issue #2807 质疑 etcd leader/follower 数据一致性同步路线图。

**(3) HIXL 传输接口 RAS 场景 core dump**

Issue #2440 报告 mooncake+hixl 在 final destruction phase core dump，stack trace 指向 `libascend_trace.so` 的 `free()` 调用。RAS 场景（对 P/D 节点注入故障）下 HIXL 接口可靠性不足。

### 3.2 需求拆解

| 序号 | 子任务 | 上游现状 | InferNex 适配 | 工作量（人月） |
|------|--------|----------|-------------|-------------|
| 2A | HA 模式直接基于 K8s（支持更多 RAS 场景异常处理能力） | ☆ Issue #1856 K8s-native Leader Election 已提交 PR 但未合入；Issue #2643 etcd+k8s HA-wrapper 合并方案推进中；`k8s_lease_helper.h` 已有 K8s lease 辅助 | ☆ 跟踪 PR #1910/#1956 合入，适配 InferNex 部署 | ~1.5（对齐需求 1.5 人） |
| 2B | Master 主备切换后增加重建元数据能力 | ☆ RFC #1150 提出持久化和恢复方案；`master_snapshot_manager` 框架已有但恢复流程未闭环；Issue #2971 OpLog retention 缺失；Issue #2807 数据一致性路线图未明确 | ☆ 需推进 RFC #1150 实现，完善快照恢复+OpLog retention | ?（涉及 Master 架构演进） |
| 2C | HIXL 传输接口 RAS 场景端到端加固 | ☆ Issue #2440 core dump（libascend_trace.so free 路径）；HIXL 传输在故障注入场景下接口可靠性不足 | ☆ 需端到端加固 HIXL 接口（deregistration/析构路径） | 1~2（对齐需求） |

### 3.3 构建节奏

| 阶段 | 周期 | 交付件 |
|------|------|--------|
| P1: K8s HA 跟踪+适配 | 2026 Q3~Q4 | 跟踪 PR #1910/#1956/#2643 合入；InferNex Helm chart 适配 K8s HA 后端 |
| P2: 元数据恢复方案 | 2026 Q4~2027 Q1 | 推进 RFC #1150；快照恢复 + OpLog retention 完善 |
| P3: HIXL 接口加固 | 2026 Q4 | Issue #2440 根因修复 + RAS 故障注入测试覆盖 |

---

## 4. 需求三：存算分离分离式部署

> 诉求来源：当前 MooncakeStore 存算分离部署能力在昇腾上尚不成熟，无法很好做到和推理引擎解耦，推理引擎挂死会导致本地池化系统一起挂死，数据丢失。

### 4.1 现状分析（代码级）

当前 Mooncake Client 与推理引擎运行在同一进程中（`real_client.h` 的 MooncakeDistributedStore 作为 Python 扩展被 vLLM 加载）。推理引擎挂死时，Mooncake Client 所在进程也挂死，本地池化的内存段（HBM/DRAM）数据随之丢失。

上游 `real_client.h` 已有独立二进制部署模式（`setup_p2p_real_client` 支持 `master_server_addr` + `client_rpc_port` 独立启动），但 vLLM-Ascend 集成路径仍为进程内嵌模式（`MooncakeLayerwiseConnector` / `MooncakeConnectorV1` 在引擎进程内运行）。

Issue #2731 GDS 架构提到 standalone-store 模式（独立 store 进程拥有 SSD 池，vLLM 作为纯请求者），为存算分离提供了架构参考，但当前仅针对 SSD offload，未覆盖 HBM/DRAM 层。

### 4.2 需求拆解

| 序号 | 子任务 | 上游现状 | InferNex 适配 | 工作量（人月） |
|------|--------|----------|-------------|-------------|
| 3A | RDMA（A2, A3, A5）, UB UBOE（A5）场景存算分离 | ☆ `real_client.h` 支持独立二进制部署（`setup_p2p_real_client`）；Issue #2731 standalone-store 模式提供架构参考；但 vLLM-Ascend 集成路径仍为进程内嵌 | ☆ 需改造 vLLM-Ascend Mooncake Connector 为独立进程模式，引擎通过 RPC + RDMA 访问独立 Store 进程 | ~3（对齐需求） |
| 3B | HCCS（A3）场景存算分离 | ☆ HCCS 场景下 Mooncake 传输依赖 HCCS 总线，进程间分离需额外处理 HCCS 设备亲和性 | ☆ 需评估 HCCS 场景下进程间共享 HCCS 设备的可行性 | ~1（对齐需求） |

### 4.3 构建节奏

| 阶段 | 周期 | 交付件 |
|------|------|--------|
| P1: 架构设计 | 第 1-4 周 | 存算分离架构 RFC（RDMA/HCCS 两种场景）+ Connector 改造方案 |
| P2: RDMA 场景实现 | 第 5-12 周 | 独立 Store 进程 + vLLM-Ascend Connector RPC 化改造 + RDMA 传输验证 |
| P3: HCCS 场景实现 | 第 13-16 周 | HCCS 设备亲和性 + 进程间共享方案 |
| P4: 端到端测试 | 第 17-18 周 | 引擎挂死后 Store 数据不丢失验证 + 性能基线 |

---

## 5. 需求四：4 级缓存，更大的内存池，更高的命中率，更优的 SSD 性能

> 诉求来源：当前 Mooncake 多级缓存能力存在待适配问题。

### 5.1 现状分析（代码级）

> ★ 重大更新：Mooncake main 分支已新增 `DfsReplicaData` + `DistributedStorageBackend` + `DfsGlobalAllocator`，L4 共享盘存储的框架级抽象**已存在**，但昇腾环境未穿刺验证。

**(1) Replica 层 -- 已有分布式文件系统副本类型**

`replica.h:222-234` 定义了 `DistributedFSDescriptor`（file_path + offset + object_size + transport_endpoint）和 `DfsReplicaData`。当前 Replica 类型体系 5 种：MemoryReplicaData / NoFReplicaData / DiskReplicaData / LocalDiskReplicaData / **DfsReplicaData**。

**(2) DistributedStorageBackend -- 已有共享盘存储后端抽象**

`include/storage/distributed/distributed_storage_backend.h`：kFileSystem / kObjectStorage 两种模式，配合 `PosixFsAdapter`（`posix_fs_adapter.h`）和 `DfsGlobalAllocator`（`dfs_global_allocator.h`）。

**(3) 上游 PR 进展**

| PR | 状态 | 内容 |
|-----|------|------|
| #3427 | open | Extract LocalSSD management from SegmentManager |
| #3467 | open | Bucket: MAX_PHYSICAL_BYTES cap on shared-disk usage |
| #3491 | open | Add GDS offload for mooncake-store |
| #3479 | open | fix(store): deduplicate SSD carryover keys |
| #3488 | open | perf(store): batch io_uring bucket reads |
| #2827 | open | Bug: DSV4 SSD offload stress testing repeat offloading（功能性问题） |

**(4) SSD 预取上游已有 RFC + draft PR**

RFC #3417（Explicit SSD->DRAM Prefetch Trigger）+ PR #2646（Prefetch SSD-Only Objects to DRAM on Exist）。

### 5.2 需求拆解

| 序号 | 子任务 | 上游现状 | InferNex 适配 | 工作量（人月） |
|------|--------|----------|-------------|-------------|
| 4A | SSD 能力仅支持到 Local SSD，基于共享盘存储的 L4 缓存尚未穿刺（仅昇腾） | ★ `DfsReplicaData` + `DistributedStorageBackend` + `DfsGlobalAllocator` 已存在；PR #3427/#3467/#3491 推进中；Issue #2827 SSD 功能性问题待修复 | ☆ 昇腾环境穿刺验证 + eviction 策略适配 + cache-indexer L3 索引扩展（当前 `/get_all_keys` 不返回 SSD 层 key） | 1.5~2（对齐需求） |
| 4B | 当前 SSD 不支持预取功能，IO bound 场景收益小（已有 draft PR 和社区 RFC） | ★ RFC #3417 + PR #2646 已有 draft 实现；`file_storage.h` 已有 `BatchLoad`/`AllocateBatch` 接口；PR #3488 batch io_uring | ☆ 对接 InferNex cache-indexer + router 触发 prefetch；PR #2646 昇腾适配 | 2~3（对齐需求） |

### 5.3 构建节奏

| 阶段 | 周期 | 交付件 |
|------|------|--------|
| P1: L4 昇腾穿刺 + SSD 功能性加固 | 第 1-4 周 | DistributedStorageBackend 昇腾环境验证 + Issue #2827 联合修复 |
| P2: SSD 预取适配 | 第 5-10 周 | RFC #3417 + PR #2646 昇腾适配 + cache-indexer/router 触发 |
| P3: 集成测试 | 第 11-12 周 | L4 + 预取 E2E + IO bound 性能基线 |

---

## 6. 需求五：更大的内存池，更大的集群

> 诉求来源：当前 MooncakeStore 机制在扩容场景存在瓶颈，Mooncake Master Service 作为单一中心化服务，大集群场景容易成为瓶颈。

### 6.1 现状分析（代码级）

> ★ 重大更新：上游已有 `HttpMetadataServer` + `MetadataStoragePlugin` + `NoFSegmentManager` + `hot_standby_service` + `enable_oplog`，分层元数据和 HA 的基础设施**已初步存在**。

**(1) 外部元数据服务**

`master_service.h:2545-2562`：`HttpMetadataServer` + `MetadataStoragePlugin`（`http_metadata_remote_`），`master.yaml` 配置 `enable_http_metadata_server`。

**(2) NoFSegmentManager 独立段管理**

`segment.h:529-598`：NVMe-oF SSD 段管理独立于主 SegmentManager，`master_service.h` 有独立的 Mount/ReMount/Unmount/GetAllNoFSegments 接口。

**(3) HA 热备与快照**

`hot_standby_service.h` + `standby_state_machine.h` + `master_snapshot_manager.h` + `enable_oplog`（OpLog 复制）+ `MetadataSerializer`（fork serialize）。

**(4) 上游 RFC 与社区进展**

RFC #2117（Hierarchical Arch，AntGroup 10,000 卡 Master 瓶颈）+ Issue #3452（Master OOM）+ Issue #1883 Roadmap（Store V3 Evolution）。

### 6.2 需求拆解

| 序号 | 子任务 | 上游现状 | InferNex 适配 | 工作量（人月） |
|------|--------|----------|-------------|-------------|
| 5A | 探索去中心化分布式元数据架构或 hierarchical 元数据架构，支持大集群统一内存池 | ★ `HttpMetadataServer` + `MetadataStoragePlugin` 已支持外部元数据；`NoFSegmentManager` 已独立；`hot_standby` + `enable_oplog` 已有 HA 基础；RFC #2117 已提出层级化架构 | ☆ 需推进 RFC #2117 实现；InferNex cache-indexer 需适配新元数据架构（当前轮询 `/get_all_keys`，Master 瓶颈影响采集延迟） | ?（涉及 MooncakeStore V3 架构演进） |

### 6.3 构建节奏

| 阶段 | 周期 | 交付件 |
|------|------|--------|
| P1: 架构 RFC + 上游对齐 | 第 1-4 周 | V3 架构 RFC + 路径对比（hierarchical vs distributed）+ 上游 HttpMetadataServer 评估 |
| P2: 核心架构实现 | 第 5-16 周 | Region Master / 分布式元数据 + Client 路由 + 跨分片操作 |
| P3: 兼容性与迁移 | 第 17-20 周 | V2->V3 兼容模式 |
| P4: 大规模测试 | 第 21-24 周 | 200+ 节点集群测试 |

---

## 7. 需求六：社区软件栈加固和优化

> 诉求来源：AscendStore 优化点识别与共创。

### 7.1 需求拆解

| 序号 | 子任务 | 上游现状 | InferNex 现状 | 工作量（人月） | 构建节奏 | 备注 |
|------|--------|----------|-------------|-------------|---------|------|
| 6A | AscendDirectTransport 段错误修复（need_update_metadata_segs_ 状态管理） | ☆ Issue #10532 已贡献上游，Mooncake v0.3.9 随机段错误（std::_Hashtable::erase） | ★ InferNex 已贡献 Issue 10532 | 跟踪验证（0.5） | 持续 | ◎ 与需求二 2C HIXL 接口加固有重叠（均为 Ascend 传输层可靠性） |
| 6B | K8s 环境 NPU 设备 ID 兼容修复 | ☆ Issue #2557 已贡献上游，torch_npu 逻辑 ID 与 ASCEND_RT_VISIBLE_DEVICES 物理 ID 不匹配 | ★ InferNex 已贡献 Issue 2557 | 跟踪验证（0.5） | 持续 | 需确认 vLLM-Ascend v0.18.0 是否已含 PR 2541 |
| 6C | V1 Mooncake Store Connector register_buffer | ☆ Issue #5044 已贡献上游 | ★ InferNex 已贡献 Issue 5044 | 跟踪验证（0.3） | 持续 | 需确认 Helm chart connectorConfig |
| 6D | Mooncake P2P HA 恢复优化 | ☆ PR #3493（simplify P2P client HA recovery for Redis HA）推进中 | ☆ InferNex PD 分离部署依赖 P2P HA | 跟踪+适配（0.5） | 2026 Q4 | ◎ 与需求二 2A/2B HA 加固有重叠 |
| 6E | Mooncake SSD 去重修复 | ☆ PR #3479（deduplicate SSD carryover keys） | ☆ 影响 L4 缓存正确性 | 跟踪验证（0.3） | 2026 Q4 | 依赖需求四 4A |
| 6F | Mooncake batch io_uring reads 性能优化 | ☆ PR #3488（batch io_uring bucket reads） | ☆ 影响 SSD 缓存读取性能 | 跟踪验证（0.3） | 2026 Q4 | 依赖需求四 4A。`file_storage.h` 已有 `BatchLoad` 接口 |
| 6G | Mooncake EGM-backed Store pool over NVLink | ☆ RFC #2914 三件套（PR #2966/#3335/#3431），NVLink 直读远端 GPU 内存免 CPU 中转 | ☆ 昇腾无 NVLink，但灵衢/RoCE 可参考 | 评估调研（0.5） | 2027 Q1~Q2 | Mooncake `device/` 已有设备抽象框架 |

---

## 8. 总览

| 需求 | 代码量（LOC） | 人月 | 建议人数 | 周期 |
|------|-------------|------|---------|------|
| 1. Layerwise 池化传输 + DSA KV Offload 接口 | ~6600 | 2.20 | 3 人 | 10 周 |
| 2. MooncakeStore 高可用加固 | ~6000+ | 1.5+?+1~2 | 1.5~3.5 人 | Q3~Q4 持续 |
| 3. 存算分离分离式部署 | ~9000+ | 3+1 | 4 人 | 18 周 |
| 4. 4 级缓存（L4 共享盘 + SSD 预取） | ~8400 | 1.5~2 + 2~3 | 3.5~5 人 | 12 周 |
| 5. Master 去中心化元数据架构 | ~8100~9300 | 2.7~3.1 | 3 人 | 24 周 |
| 6. 社区软件栈加固和优化 | ~跟踪验证 | 2.1+ | 1~2 人（持续跟踪） | 持续 |
| **合计** | **~38000+** | **~13~16+** | -- | -- |

---

## 9. 优先级与依赖关系

```
需求 1（Layerwise 接口）──> 需求 3（存算分离）依赖传输接口
                              │
需求 2（HA 加固）─────────────┤ 2A/2B 与需求 5 元数据架构协同
                              │
需求 4（L4 + 预取）───────────┤ 4A 依赖需求 3 存算分离（独立 Store 进程才有 L4 意义）
                              │
需求 5（Master 去中心化）──────┘ 独立可并行，但 L4 扩容后元数据压力更大
                              │
需求 6（社区加固）─────────────┘ 持续跟踪，与各需求交叉
```

**建议执行顺序:**
1. 需求 1（Layerwise 接口）+ 需求 2（HA 加固）并行启动
2. 需求 3（存算分离）依赖需求 1 接口，紧随其后
3. 需求 4（L4 + 预取）依赖需求 3 存算分离（独立 Store 进程），第三批启动
4. 需求 5（Master 去中心化）独立启动架构 RFC，实现阶段与需求 4 协同
5. 需求 6（社区加固）持续跟踪，与各需求交叉推进

---

## 10. 风险与建议

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Layerwise session API 昇腾适配 | GVA 地址映射不兼容 | 需评估 AscendDirectTransport GVA 路径与 cuda_ipc_buffer 差异 |
| HA K8s 后端 PR 未合入 | 需求 2A 阻塞 | 跟踪 PR #1910/#1956/#2643；必要时 cherry-pick |
| Master 元数据恢复方案未闭环 | 需求 2B 阻塞 | 推进 RFC #1150；完善 master_snapshot_manager 恢复流程 |
| HIXL RAS core dump | 需求 2C 阻塞 | Issue #2440 根因修复（libascend_trace.so free 路径） |
| 存算分离改造影响 vLLM-Ascend 集成 | 需求 3 兼容性 | Connector RPC 化需 vLLM-Ascend 团队协同；保留进程内嵌模式作为兼容 |
| 共享盘 eviction 正确性 | 数据丢失 | 参考 3FS 禁用 eviction 教训；设计共享盘专用 eviction 协议 |
| 上游 PR 合入时间 | 多个依赖 PR 均 open | PR #3427/#3467/#3488/#3491/#2646/#3493 跟踪合入 |
| 昇腾环境测试覆盖 | 生产环境缺陷 | 需真实昇腾集群验证；CI 增加 Ascend 专用测试矩阵 |

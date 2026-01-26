# A3代际灵衢(UB)架构技术白皮书

## 文档概述

本文档基于MemFabric Hybrid开源内存池化软件，详细阐述昇腾A3代际灵衢(LingQu/UB)硬件架构、RH2D等传输通信路径、统一编址(GVA)以及对应关键灵衢底层系统服务的技术原理。旨在为分布式KVCache管理、权重分发加速等典型场景提供基于对等互联架构实现加速的设计参考。

---

## 1. A3灵衢硬件架构

### 1.1 架构概述

昇腾A3超节点采用灵衢(LingQu Computing Network)高速互联网络，实现多节点间内存池化和高性能数据访问。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        A3 SuperPod 架构                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐      │
│  │  Server 1 │    │  Server 2 │    │  Server N │    │  Server M │      │
│  │           │    │           │    │           │    │           │      │
│  │  ┌─────┐  │    │  ┌─────┐  │    │  ┌─────┐  │    │  ┌─────┐  │      │
│  │  │NPU 0│  │    │  │NPU 0│  │    │  │NPU 0│  │    │  │NPU 0│  │      │
│  │  │ HBM │  │    │  │ HBM │  │    │  │ HBM │  │    │  │ HBM │  │      │
│  │  └──┬──┘  │    │  └──┬──┘  │    │  └──┬──┘  │    │  └──┬──┘  │      │
│  │     │     │    │     │     │    │     │     │    │     │     │      │
│  │  ┌─────┐  │    │  ┌─────┐  │    │  ┌─────┐  │    │  ┌─────┐  │      │
│  │  │NPU 1│  │    │  │NPU 1│  │    │  │NPU 1│  │    │  │NPU 1│  │      │
│  │  │ HBM │  │    │  │ HBM │  │    │  │ HBM │  │    │  │ HBM │  │      │
│  │  └──┬──┘  │    │  └──┬──┘  │    │  └──┬──┘  │    │  └──┬──┘  │      │
│  └─────┼──────┘    └─────┼──────┘    └─────┼──────┘    └─────┼──────┘      │
│        │                  │                  │                  │            │
│        └──────────────────┼──────────────────┼──────────────────┘            │
│                           │                  │                               │
│                    ┌──────┴──────┐    ┌──────┴──────┐                        │
│                    │   L1 交换   │    │   L1 交换   │                        │
│                    │   (灵衢)    │    │   (灵衢)    │                        │
│                    └──────┬──────┘    └──────┬──────┘                        │
│                           └──────────────────┘                               │
│                                   │                                         │
│                           ┌──────┴──────┐                                   │
│                           │  L2 交换矩阵  │                                   │
│                           │  (56个L2)    │                                   │
│                           └─────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 灵衢互联特性

**灵衢(UB) vs RoCE对比：**

| 特性 | 灵衢 UB 1.0 | RoCE |
|------|------------|------|
| 传输层级 | 设备级直连 | 网络级传输 |
| 延迟 | 亚微秒级 | 微秒-毫秒级 |
| 带宽 | >100GB/s | 取决于网络配置 |
| 内存访问 | 设备直接内存访问 | 主机参与 |
| 适用场景 | A3超节点内 | 跨机柜/通用场景 |
| 硬件依赖 | 需要灵衡交换机 | 标准以太网交换机 |

### 1.3 传输类型定义

```cpp
// src/hybm/csrc/transport/hybm_transport_common.h
enum TransportType {
    TT_HCCP = 0,     // 设备传输 (基于灵衢的HCCP协议)
    TT_HCOM,         // 主机传输 (基于RoCE)
    TT_COMPOSE,      // 组合传输
    TT_BUTT,
};
```

### 1.4 硬件组件关系

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MemFabric 传输管理层                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐      ┌──────────────────┐                    │
│  │ RdmaTransportMgr │      │ HcomTransportMgr │                    │
│  │   (设备/UB)      │      │   (主机/RoCE)    │                    │
│  └────────┬─────────┘      └────────┬─────────┘                    │
│           │                         │                              │
│           └──────────┬──────────────┘                              │
│                      ▼                                             │
│         ┌──────────────────────┐                                  │
│         │ ComposeTransportMgr  │                                  │
│         │    (组合传输管理)     │                                  │
│         └──────────────────────┘                                  │
│                      │                                             │
│         ┌────────────┴────────────┐                               │
│         │                         │                               │
│    ┌────▼────┐              ┌────▼────┐                          │
│    │QP管理器 │              │流管理器  │                          │
│    └─────────┘              └─────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
                      │           │
                      ▼           ▼
              ┌──────────┐  ┌──────────┐
              │ HCCP协议 │  │  RDMA    │
              └──────────┘  └──────────┘
```

---

## 2. RH2D通信路径详解

### 2.1 RH2D定义

**RH2D (Remote Host to Device)**: 远程主机DRAM → 本地设备HBM的数据传输路径。

这是A3架构中最关键的跨节点数据访问路径之一，典型应用场景包括：
- 分布式KV Cache访问
- 模型权重分发
- 梯度聚合

### 2.2 数据流向图

```
RH2D数据流：远程主机DRAM池 → 本地设备HBM

┌───────────────────┐                           ┌───────────────────┐
│   远程节点         │                           │   本地节点         │
├───────────────────┤                           ├───────────────────┤
│                   │                           │                   │
│  ┌─────────────┐  │    ①RDMA Read            │  ┌─────────────┐  │
│  │ 远程Host    │  │ ──────────────────────→  │  │ 交换缓冲区   │  │
│  │ DRAM        │  │   (灵衡/PCIe)           │  │ (Swap Buffer)│  │
│  │             │  │                          │  │             │  │
│  │ GVA:0xXXXX  │  │                          │  │ LVA:0xYYYY  │  │
│  └─────────────┘  │                          │  └──────┬──────┘  │
│                   │                          │         │         │
│                   │                          │  ②Host→Device    │
│                   │                          │         │         │
│                   │                          │  ┌──────▼──────┐  │
│                   │                          │  │  本地NPU    │  │
│                   │                          │  │  HBM        │  │
│                   │                          │  │             │  │
│                   │                          │  │ LVA:0xZZZZ  │  │
│                   │                          │  └─────────────┘  │
└───────────────────┘                          └───────────────────┘
```

### 2.3 RH2D传输实现

**关键代码路径：** `src/hybm/csrc/data_operation/host/hybm_data_op_host_rdma.h`

```cpp
// RH2D传输类型定义
enum hybm_data_copy_direction {
    HYBM_GLOBAL_HOST_TO_LOCAL_DEVICE,   // GH→LD (RH2D)
    HYBM_LOCAL_HOST_TO_GLOBAL_DEVICE,   // LH→GD
    HYBM_GLOBAL_DEVICE_TO_LOCAL_HOST,   // GD→LH
    HYBM_LOCAL_DEVICE_TO_GLOBAL_HOST,   // LD→GH
    // ... 其他方向
};
```

**两阶段传输流程：**

```cpp
// 阶段1: 远程主机 → 本地主机的RDMA Read
Result CopyGH2LH(void *srcVA, void *dstVA, uint64_t length) {
    // 获取远程内存key
    uint32_t remoteRankId = GetRankIdByGVA(srcVA);
    uint64_t remoteGVA = ConvertToGVA(srcVA);

    // 使用交换缓冲区
    void *swapBuffer = GetHostSwapBuffer();

    // RDMA Read操作
    transportManager_->ReadRemote(remoteRankId,
                                  swapBuffer,      // 本地地址
                                  remoteGVA,       // 远程地址
                                  length);

    return Result::Success();
}

// 阶段2: 本地主机 → 本地设备的DMA
Result CopyLH2LD(void *srcVA, void *dstVA, uint64_t length) {
    // 使用ACL API进行Host→Device传输
    DlAclApi::AclrtMemcpy(dstVA,           // 设备地址
                         length,
                         srcVA,           // 主机地址
                         length,
                         ACL_MEMCPY_HOST_TO_DEVICE);

    return Result::Success();
}
```

### 2.4 交换缓冲区设计

```
┌─────────────────────────────────────────────────────────────────┐
│                     交换缓冲区(Swap Buffer)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Host Swap Buffer: 1GB                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  [Block 0] [Block 1] [Block 2] ... [Block N-1]          │  │
│  │  (2MB对齐，支持RDMA注册)                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Device Swap Buffer: 128MB                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  [Block 0] [Block 1] [Block 2] ... [Block M-1]          │  │
│  │  (HBM空间，用于设备内中转)                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.5 完整RH2D数据流时序图

```
┌─────┐                ┌─────┐                ┌─────┐
│本地NPU│                │本地Host│               │远程Host│
└──┬──┘                └──┬──┘                └──┬──┘
   │                       │                      │
   │ ①发起RH2D请求         │                      │
   ├──────────────────────►│                      │
   │ (包含GVA地址)          │                      │
   │                       │                      │
   │                       │ ②查询远程Rank映射    │
   │                       ├─────────────────────►│
   │                       │                      │
   │                       │ ③返回远程memory key  │
   │                       │◄─────────────────────┤
   │                       │                      │
   │                       │ ④RDMA Read (异步)    │
   │                       ├──────────────��──────►│
   │                       │                      │
   │                       │ ⑤数据到达交换缓冲区  │
   │                       │◄─────────────────────┤
   │                       │                      │
   │ ⑥Host→Device DMA     │                      │
   │◄──────────────────────┤                      │
   │                       │                      │
   │ ⑦DMA完成通知          │                      │
   ├──────────────────────►│                      │
   │                       │                      │
```

---

## 3. 统一编址GVA

### 3.1 GVA地址空间布局

GVA (Global Virtual Address) 是全局统一虚拟地址空间，实现跨节点内存的透明访问。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       GVA地址空间布局 (256TB)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  0x100000000000 (16T) ─────┐                                           │
│                            ├─ HBM空间 (8TB)                             │
│                            │  ┌──────────────────────────────────────┐  │
│  0x180000000000 (24T) ─────┘  │  设备HBM池化空间                      │  │
│                               │  每个设备128GB，支持64个设备          │  │
│  ...                          └──────────────────────────────────────┘  │
│                                                                         │
│  0x280000000000 (160T) ────┐                                           │
│                             ├─ GVM空间 (128TB)                          │
│  0xA80000000000 (288T) ─────┘  ┌──────────────────────────────────────┐ │
│                                │  全局虚拟内存空间                      │ │
│  ...                           │  DRAM池化 + HBM池化                   │ │
│                                └──────────────────────────────────────┘ │
│                                                                         │
│  0x30000000000 (48T) ─────┐                                           │
│                            ├─ Host连接空间 (16TB)                       │
│  0x130000000000 (64T) ─────┘  ┌──────────────────────────────────────┐ │
│                                │  主机连接内存                          │ │
│  ...                           │  用于跨节点通信                       │ │
│                                └──────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 GVA核心数据结构

**内存类型定义：**

```cpp
// src/hybm/csrc/common/hybm_define.h
enum hybm_mem_type {
    HYBM_MEM_TYPE_DEVICE = 1U << 0,  // 设备HBM内存
    HYBM_MEM_TYPE_HOST = 1U << 1,     // 主机DRAM内存
};
```

**GVA分配信息结构：**

```cpp
// src/hybm/csrc/mm/hybm_va_manager.h
struct BaseAllocatedGvaInfo {
    uint64_t gva;              // 全局虚拟地址
    uint64_t size;             // 内存大小
    hybm_mem_type memType;     // 内存类型(HBM/DRAM)
    uint64_t lva;              // 本地虚拟地址
};

struct AllocatedGvaInfo : BaseAllocatedGvaInfo {
    bool registered;           // 是否为本地分配内存
    uint32_t localRankId;      // 本地Rank ID
    uint32_t importedRankId;   // 导入内存的源Rank ID
};
```

### 3.3 VA管理器架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HybmVaManager (GVA管理器)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  核心映射表                                                   │ │
│  │  ┌────────────────────────────────────────────────────────┐ │ │
│  │  │ allocatedLookupMapByGva_  : map<uint64_t, AllocatedGvaInfo>│ │ │
│  │  │ allocatedLookupMapByLva_  : map<uint64_t, AllocatedGvaInfo>│ │ │
│  │  │ reservedLookupMapByGva_   : map<uint64_t, ReservedGvaInfo> │ │ │
│  │  │ allocatedLookupMapByMemType_: map<memType, vector<Info>>  │ │ │
│  │  └────────────────────────────────────────────────────────┘ │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  核心操作                                                     │ │
│  │  • AllocateMemory()    - 分配GVA内存                          │ │
│  │  • RegisterMemory()    - 注册现有内存为GVA                    │ │
│  │  • ImportMemory()      - 导入远程内存                         │ │
│  │  • GvaToLva()          - GVA到LVA转换                         │ │
│  │  • QueryMemInfo()      - 查询内存信息                         │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.4 GVA地址转换流程

```
用户请求: 读取Rank 2的HBM内存，大小1MB

┌─────────┐
│ 用户请求 │ "读取 GVA=0x10000200000, 1MB"
└────┬────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: GVA解析                                                     │
│ • 解析GVA地址所属内存类型 (HBM/DRAM)                                 │
│ • 计算内存块偏移: offset = GVA - segment.baseGVA                    │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: 查询GVA映射表                                                │
│ • 在allocatedLookupMapByGva_中查找                                  │
│ • 获取AllocatedGvaInfo: {gva, size, memType, lva, rankId}          │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: 确定数据路径                                                  │
│ if (rankId == localRankId) {                                        │
│     → 本地内存访问: LVA = baseLVA + offset                          │
│ } else if (registered) {                                            │
│     → 直接RDMA: 使用远程memory key                                  │
│ } else {                                                            │
│     → 两阶段传输: 远程→交换缓冲区→本地                              │
│ }                                                                   │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────┐
│ 执行传输 │
└─────────┘
```

### 3.5 页表管理

**页大小配置：**

```cpp
constexpr uint64_t SMALL_PAGE_SIZE = 4U * KB;          // 4KB小页
constexpr uint64_t HYBM_LARGE_PAGE_SIZE = 2UL * MB;    // 2MB大页 (灵衡)
```

**页表类型：**

```cpp
enum DevPageType {
    DEVMM_NORMAL_PAGE_TYPE = 0x0,    // 普通页
    DEVMM_HUGE_PAGE_TYPE,            // 大页 (灵衡必需)
    DEVMM_PAGE_TYPE_MAX
};
```

### 3.6 GVA一致性保证

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GVA一致性机制                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 全局元数据共享                                                  │
│     • 所有进程通过共享内存访问GVA元数据                              │
│     • 使用acc_links TCP通道进行元数据同步                           │
│                                                                     │
│  2. 地址预分配                                                      │
│     • 初始化时为每个节点预分配GVA范围                               │
│     • 保证GVA空间不重叠                                             │
│                                                                     │
│  3. 内存导入/导出                                                  │
│     • Export: 发布本地内存为GVA                                     │
│     • Import: 获取远程内存的GVA映射                                 │
│                                                                     │
│  4. 版本控制                                                        │
│     enum HybmGvaVersion {                                          │
│         HYBM_GVA_V1 = 0,                                           │
│         HYBM_GVA_V2 = 1,                                           │
│         HYBM_GVA_V3 = 2,                                           │
│         HYBM_GVA_V4 = 3,                                           │
│     };                                                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 关键系统服务

### 4.1 传输管理服务

**TransportManager接口：**

```cpp
// src/hybm/csrc/transport/hybm_transport_manager.h
class TransportManager {
public:
    // 设备初始化
    virtual Result OpenDevice(const TransportOptions &options) = 0;

    // 内存区域注册
    virtual Result RegisterMemoryRegion(const TransportMemoryRegion &mr) = 0;

    // 传输准备
    virtual Result Prepare(const HybmTransPrepareOptions &options) = 0;

    // 建立连接
    virtual Result Connect() = 0;

    // 远程读操作
    virtual Result ReadRemote(uint32_t rankId,
                              uint64_t lAddr,
                              uint64_t rAddr,
                              uint64_t size) = 0;

    // 远程写操作
    virtual Result WriteRemote(uint32_t rankId,
                               uint64_t lAddr,
                               uint64_t rAddr,
                               uint64_t size) = 0;
};
```

### 4.2 QP连接管理

**设备QP管理器：**

```cpp
// src/hybm/csrc/transport/device/device_qp_manager.h
struct UserQpInfo {
    void *qpHandle{nullptr};           // QP句柄
    std::atomic<uint32_t> ref{0};      // 引用计数
};

struct ConnectionChannel {
    sockaddr_in remoteNet;             // 远端网络地址
    void *socketHandle;                // Socket句柄
    void *qpHandle{nullptr};           // QP句柄
    bool qpConnectCalled{false};       // 连接状态
    int qpStatus{-1};                  // QP状态
};
```

**连接建立流程：**

```
┌─────────┐      ┌─────────┐      ┌─────────┐
│ Rank 0  │      │ Rank 1  │      │ Rank 2  │
└────┬────┘      └────┬────┘      └────┬────┘
     │                │                │
     │ ①TCP连接        │                │
     ├───────────────►│                │
     │                │                │
     │ ②创建QP         │                │
     ├───────────────►│                │
     │                │                │
     │ ③交换QP信息     │                │
     │◄───────────────┤                │
     │                │                │
     │ ④QP连接         │                │
     ├───────────────►│                │
     │                │                │
     │ ⑤连接完成       │                │
     │◄───────────────┤                │
     │                │                │
```

### 4.3 内存注册服务

**内存区域结构：**

```cpp
struct TransportMemoryRegion {
    uint64_t addr = 0;                                   // 虚拟地址
    uint64_t size = 0;                                   // 大小
    int32_t access = REG_MR_ACCESS_FLAG_BOTH_READ_WRITE; // 访问权限
    uint32_t flags = 0;                                  // 标志(DRAM/HBM)
};
```

**访问权限定义：**

```cpp
constexpr int32_t REG_MR_ACCESS_FLAG_LOCAL_WRITE = 0x1;
constexpr int32_t REG_MR_ACCESS_FLAG_REMOTE_WRITE = 0x2;
constexpr int32_t REG_MR_ACCESS_FLAG_REMOTE_READ = 0x4;
constexpr int32_t REG_MR_ACCESS_FLAG_BOTH_READ_WRITE = 0x7;
```

### 4.4 DMA/Xcopy服务

**数据操作工厂：**

```cpp
// src/hybm/csrc/data_operation/hybm_data_op_factory.h
class DataOperatorFactory {
public:
    // 创建SDMA数据操作符
    static DataOperatorPtr CreateSdmaDataOperator();

    // 创建设备RDMA操作符
    static DataOperatorPtr CreateDevRdmaDataOperator(
        uint32_t rankId,
        const transport::TransManagerPtr &tm);

    // 创建主机RDMA操作符
    static DataOperatorPtr CreateHostRdmaDataOperator(
        uint32_t rankId,
        const transport::TransManagerPtr &tm);
};
```

### 4.5 acc_links控制通道

**acc_links架构：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                      acc_links 控制通信层                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  每个节点运行的TCP Server:                                          │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  AccTcpServer                                                 │ │
│  │  ├── 监听端口: 接收远端连接                                    │ │
│  │  ├── 连接管理: 维护与所有其他节点的连接                        │ │
│  │  ├── 请求处理: 处理控制命令                                    │ │
│  │  └── 响应发送: 返回操作结果                                    │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  消息类型:                                                          │
│  ├── ACC_CONN_REQ: 连接请求                                        │
│  ├── ACC_CONN_RESP: 连接响应                                       │
│  ├── ACC_DATA_REQ: 数据请求                                        │
│  └── ACC_DATA_RESP: 数据响应                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. 关键接口说明

### 5.1 BM API

**初始化接口：**

```c
// src/smem/include/host/smem_bm.h
int32_t smem_bm_init(const char *storeURL,      // 元数据存储URL
                     uint32_t worldSize,         // 集群大小
                     uint16_t deviceId,          // 设备ID
                     const smem_bm_config_t *config);  // 配置参数
```

**创建BM实例：**

```c
smem_bm_t smem_bm_create(uint32_t id,                // BM ID
                         uint32_t memberSize,        // 成员大小
                         smem_bm_data_op_type type,  // 数据操作类型
                         uint64_t localDRAMSize,     // 本地DRAM大小
                         uint64_t localHBMSize,      // 本地HBM大小
                         uint32_t flags);            // 标志位
```

**获取内存指针：**

```c
void *smem_bm_ptr_by_mem_type(smem_bm_t handle,
                              smem_bm_mem_type memType,  // HBM/DRAM
                              uint16_t peerRankId);      // 对端Rank
```

**数据拷贝：**

```c
int32_t smem_bm_copy(smem_bm_t handle,
                     smem_copy_params *params,  // 拷贝参数
                     smem_bm_copy_type type,    // L2G/G2L/...
                     uint32_t flags);
```

**拷贝类型定义：**

```c
typedef enum {
    SMEM_COPY_TYPE_L2G = 0,  // Local→Global
    SMEM_COPY_TYPE_G2L,      // Global→Local
    SMEM_COPY_TYPE_H2G,      // Host→Global
    SMEM_COPY_TYPE_G2H,      // Global→Host
    SMEM_COPY_TYPE_G2G,      // Global→Global
} smem_bm_copy_type;
```

### 5.2 Python接口

**BM Python API：**

```python
import memfabric_hybrid as mf

# 初始化
mf.smem.bm.initialize(store_url, world_size, device_id, config)

# 创建BM实例
bm_handle = mf.smem.bm.create(bm_id, member_size,
                              data_op_type,
                              local_dram_size,
                              local_hbm_size)

# 获取远端内存指针
peer_ptr = bm_handle.peer_rank_ptr(mem_type, peer_rank_id)

# 拷贝数据
bm_handle.copy_data(src, dest, size, copy_type)
```

### 5.3 传输配置接口

```c
typedef struct {
    uint32_t rankId;              // Rank ID
    char *netDev;                 // 网络设备名称
    char *netAddr;                // 网络地址
    uint32_t port;                // 端口号
    uint32_t tcpFlag;             // TCP标志
} smem_trans_config_t;
```

---

## 6. 关键架构和实现技术

### 6.1 零拷贝技术

**RDMA零拷贝：**

```
传统拷贝路径:
[远程内存] → [网卡] → [内核缓冲] → [用户缓冲] → [设备内存]
    (6次内存拷贝，4次上下文切换)

RDMA零拷贝路径:
[远程内存] → [网卡] → [设备内存]
    (1次内存拷贝，0次上下文切换)
```

**实现要点：**

1. **内存注册(Memory Registration)**
   ```cpp
   // 内存注册流程
   // 1. 分配内存
   void *buffer = malloc_aligned(size, 2MB);

   // 2. 注册到RDMA设备
   transportManager_->RegisterMemoryRegion({
       .addr = (uint64_t)buffer,
       .size = size,
       .access = REG_MR_ACCESS_FLAG_BOTH_READ_WRITE,
       .flags = HYBM_MEM_TYPE_HOST
   });
   ```

2. **直接内存访问**
   ```cpp
   // 直接RDMA读，无需CPU参与
   transportManager_->ReadRemote(remoteRank,
                                  localAddr,
                                  remoteAddr,
                                  size);
   // DMA引擎直接完成数据传输
   ```

### 6.2 内存对齐

**2MB对齐要求：**

```cpp
// 分配2MB对齐的内存
void *ptr = nullptr;
posix_memalign(&ptr, 2 * 1024 * 1024, size);

// 或者使用huge pages
void *huge_ptr = mmap(NULL, size,
                      PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS |
                      MAP_HUGETLB | MAP_HUGE_2MB,
                      -1, 0);
```

### 6.3 批量操作

**批量拷贝优化：**

```cpp
// 单次拷贝 vs 批量拷贝
for (int i = 0; i < n; i++) {
    copy_single(src[i], dst[i], size[i]);  // n次RDMA操作
}

// 批量拷贝 - 1次RDMA操作
batch_copy(src_list, dst_list, size_list, n);
```

### 6.4 异步操作

**异步传输模式：**

```cpp
// 异步RDMA操作
struct AsyncRequest {
    std::promise<Result> result;
    std::future<Result> future;
    std::atomic<bool> completed{false};
};

// 发起异步操作
auto req = std::make_shared<AsyncRequest>();
transportManager_->ReadRemoteAsync(rankId, laddr, raddr, size,
    [req](Result r) {
        req->result.set_value(r);
        req->completed = true;
    });

// 继续其他计算...
do_other_work();

// 等待完成
req->future.wait();
```

### 6.5 流水线优化

**计算-通信重叠：**

```
时间轴:
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│Compute0 │ Comm0   │Compute1 │ Comm1   │Compute2 │
└─────────┴─────────┴─────────┴─────────┴─────────┘

传统串行:
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│Compute0 │Compute1 │Compute2 │ Comm0   │ Comm1   │
└─────────┴─────────┴─────────┴─────────┴─────────┘
```

---

## 7. 应用场景参考

### 7.1 分布式KV Cache管理

**场景描述：**
大模型推理时，KV Cache分布在多个节点的DRAM/HBM中，需要高效访问。

**架构设计：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  分布式KV Cache架构 (基于MemFabric)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐         │
│  │  推理节点0   │      │  推理节点1   │      │  内存节点    │         │
│  │             │      │             │      │             │         │
│  │ ┌─────────┐ │      │ ┌─────────┐ │      │ ┌─────────┐ │         │
│  │ │Model    │ │      │ │Model    │ │      │ │KV Cache │ │         │
│  │ │Weights  │ │      │ │Weights  │ │      │ │Pool     │ │         │
│  │ └────┬────┘ │      │ └────┬────┘ │      │ └────┬────┘ │         │
│  │      │      │      │      │      │      │      │      │         │
│  │ ┌────▼────┐ │      │ ┌────▼────┐ │      │ ┌────▼────┐ │         │
│  │ │Local KV │ │      │ │Local KV │ │      │ │Remote KV│ │         │
│  │ │Cache    │ │      │ │Cache    │ │      │ │Cache    │ │         │
│  │ └─────────┘ │      │ └─────────┘ │      │ └─────────┘ │         │
│  └──────┬──────┘      └──────┬──────┘      └──────┬──────┘         │
│         │                    │                    │                 │
│         └────────────────────┼────────────────────┘                 │
│                              │                                      │
│                    ┌─────────▼─────────┐                            │
│                    │  MemFabric GVA    │                            │
│                    │  统一地址空间      │                            │
│                    └───────────────────┘                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**关键代码示例：**

```python
import memfabric_hybrid as mf

# 初始化MemFabric
mf.smem.bm.initialize("etcd://localhost:2379", world_size=8,
                      device_id=0, config=None)

# 创建KV Cache池 (本地HBM + 远程DRAM)
kv_pool = mf.smem.bm.create(
    bm_id=1,
    member_size=world_size,
    data_op_type=mf.smem.bm.DataOpType.DEVICE_RDMA,
    local_dram_size=0,       # 不使用本地DRAM
    local_hbm_size=32 * GB,  # 本地HBM 32GB
    flags=0
)

# 获取远程KV Cache指针
remote_kv_ptr = kv_pool.peer_rank_ptr(
    mem_type=mf.smem.bm.MemType.DEVICE,
    peer_rank_id=memory_node_rank
)

# 读取远程KV Cache (RH2D: Remote Host→Local Device)
kv_pool.copy_data(
    src=remote_kv_ptr + offset,
    dst=local_kv_buffer,
    size=tokens * hidden_dim,
    copy_type=mf.smem.bm.CopyType.G2L  # Global→Local
)
```

### 7.2 权重分发加速

**场景描述：**
大模型加载时，权重需要从存储节点分发到所有计算节点。

**架构设计：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                   权重分发架构 (基于MemFabric)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐                                                   │
│  │  存储节点    │                                                   │
│  │             │              ┌─────────────┐                      │
│  │ ┌─────────┐ │              │  计算节点0   │                      │
│  │ │Model    │ │              │ ┌─────────┐ │                      │
│  │ │Weights  │ │              │ │NPU 0    │ │                      │
│  │ │(160GB)  │ │              │ │HBM      │ │                      │
│  │ └────┬────┘ │              │ └────┬────┘ │                      │
│  └──────┼──────┘              └──────┼──────┘                      │
│         │ G2G拷贝                     │                             │
│         ├─────────────────────────────┼──────┐                      │
│         │                             │      │                      │
│         │                        ┌────▼──────▼───┐                   │
│         │                        │  计算节点N     │                   │
│         │                        │ ┌───────────┐ │                   │
│         └───────────────────────→│ │NPU 0-NPU 7│ │                   │
│                                  │ │HBM Pool   │ │                   │
│                                  │ └───────────┘ │                   │
│                                  └───────────────┘                   │
│                                                                     │
│                    一次性G2G广播，所有节点并行接收                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.3 梯度聚合

**场景描述：**
分布式训练时，梯度需要从所有计算节点聚合到参数服务器。

**架构设计：**

```python
# 梯度聚合示例
def aggregate_gradients(gradients_list):
    # 使用MemFabric的G2G进行梯度聚合

    # 1. 创建聚合缓冲区
    agg_buffer = allocate_buffer(gradient_size)

    # 2. 从所有节点收集梯度
    for rank, grad in enumerate(gradients_list):
        bm.copy_data(
            src=grad,              # 本地梯度
            dst=agg_buffer + rank * gradient_size,
            size=gradient_size,
            copy_type=CopyType.L2G  # Local→Global
        )

    # 3. 执行AllReduce (使用MemFabric内置)
    bm.allreduce(agg_buffer, op=ReduceOp.SUM)

    # 4. 分发聚合后的梯度
    for rank in range(world_size):
        bm.copy_data(
            src=agg_buffer,
            dst=rank_gradient_buffer(rank),
            size=gradient_size,
            copy_type=CopyType.G2L  # Global→Local
        )
```

---

## 8. 性能优化建议

### 8.1 内存布局优化

```cpp
// 好的内存布局 - 连续GVA地址
Layout A: [Node0-HBM][Node1-HBM][Node2-HBM]...

// 不好的内存布局 - 交错地址
Layout B: [Node0-HBM0][Node1-HBM0][Node0-HBM1][Node1-HBM1]...
```

### 8.2 批量操作

```python
# 不好的做法 - 单次请求
for i in range(1000):
    bm.copy_data(src[i], dst[i], size, type)

# 好的做法 - 批量请求
params = [(src[i], dst[i], size) for i in range(1000)]
bm.batch_copy(params, type)
```

### 8.3 交换缓冲区调优

```c
// 根据业务模式调整交换缓冲区大小
config.host_swap_size = 2 * GB;     // 大文件传输
config.device_swap_size = 256 * MB; // 高并发小请求
```

### 8.4 预取策略

```python
# 异步预取下一批数据
async def prefetch_next_batch():
    future = bm.copy_async(next_batch_src, local_buffer, size, G2L)
    return future

# 在处理当前数据时，后台预取下一批
prefetch_future = prefetch_next_batch()
process_current_batch()
prefetch_future.wait()
```

---

## 9. 总结

本文档详细阐述了A3代际灵衢硬件架构的核心技术要点：

### 9.1 核心技术栈

| 层级 | 技术 | 作用 |
|------|------|------|
| 硬件层 | 灵衡互联 | 亚微秒级设备间通信 |
| 传输层 | HCCP/RoCE | 可靠数据传输 |
| 内存层 | GVA | 统一地址空间 |
| 接口层 | BM/SHM/TRANS API | 简易编程模型 |

### 9.2 关键设计原则

1. **零拷贝**: 使用RDMA直接内存访问
2. **统一编址**: GVA实现跨节点透明访问
3. **异步流水**: 计算-通信重叠
4. **内存对齐**: 2MB对齐支持灵衡大页

### 9.3 适用场景

- 分布式KV Cache管理
- 模型权重分发
- 梯度聚合
- 跨节点数据共享

---

**文档版本**: v1.0
**基于**: MemFabric Hybrid 开源项目
**日期**: 2025年1月

# Mooncake 上游席位获取策略分析

**日期**: 2026-03-26
**作者**: Xun Sun
**状态**: 策略规划

---

## 1. 目标定义

**目标**：获得 Mooncake 社区 **Store 模块 CODEOWNERS** 权限。

CODEOWNERS 权限意味着：
- 对 `/mooncake-store` 目录下的 PR 有 merge 权限
- 参与模块方向决策
- 社区技术影响力的正式认可

---

## 2. 当前社区结构分析

### 2.1 CODEOWNERS 人员组成

| 模块 | CODEOWNERS | 归属组织 | 提交活跃度(6个月) |
|------|-----------|---------|------------------|
| **Transfer Engine** | @alogfans, @doujiang24, @chestnut-Q | 9#AISoft, 社区 | 高 (Feng Ren: 94 commits) |
| **Store** | @ykwd, @stmatengss, @XucSh, @YiXR | Approaching AI, 阿里云 | 高 (JinYan Su: 100 commits) |
| **EP/PG** | @UNIDY2002, @ympcMark | 社区 | 中等 |
| **Integration/TE** | @ShangmingCai, @alogfans | 阿里云, 9#AISoft | 高 |
| **Integration/Store** | @ykwd, @stmatengss | Approaching AI, 阿里云 | 高 |

### 2.2 关键发现

1. **Store 模块开放性高**：已有 4 位 CODEOWNER，说明接纳新成员的门槛相对较低
2. **核心决策者**：
   - @ykwd (Ke Yang, Approaching AI) — Store 主负责人
   - @stmatengss (Teng Ma, 阿里云) — LLM 生态合作，对集成方向有话语权
3. **高贡献竞争者**：JinYan Su (100 commits/6mo)、Feng Ren (94 commits) 是主要"竞争对手"
4. **EP 模块门槛最低**：仅 2 位 CODEOWNER，但模块已标记 deprecated
5. **Transfer Engine 被 9#AISoft 主导**：外部贡献者难以突破

### 2.3 社区运作模式

基于 `CONTRIBUTING.md` 和实际 PR 流程：

| 规则 | 说明 |
|------|------|
| RFC 文化 | >500 LOC 必须先在 GitHub Issue 中发起 RFC 讨论 |
| PR 前缀 | `[Store]`、`[TransferEngine]` 等分类前缀 |
| Code Review | 至少 1 位 CODEOWNER approve 才能合并 |
| Pre-commit | clang-format/ruff 必过，否则不会进入 review |
| CI 通过 | 所有 review 前必须 CI 绿 |

---

## 3. 技术机会点分析

### 3.1 Store 模块前沿方向

| 方向 | 成熟度 | 竞争强度 | 推荐度 |
|------|--------|---------|--------|
| **KVCache Layout 优化** | 发展中 | 低 | ⭐⭐⭐⭐⭐ |
| Storage Backend Benchmark | 新兴 | 中 | ⭐⭐⭐⭐ |
| Batch API 支持 | 刚起步 | 低 | ⭐⭐⭐⭐ |
| TENT 集成 | 发展中 | 中 | ⭐⭐⭐ |
| 错误处理/重试 | 成熟 | 高 | ⭐⭐ |

### 3.2 KVCache Layout 优势分析

**为什么选择 KVCache Layout 作为切入点**：

1. **技术前沿**：Qwen3.5/DeepSeek 等 new models 采用 Hybrid Attention (GQA+MLA+GDN)
2. **影响力大**：直接影响所有推理框架 (vLLM/SGLang/TensorRT-LLM) 的集成效果
3. **竞争少**：目前仅 @ykwd 深度理解这块，缺乏第二位专家
4. **已有基础**：我已经完成了 GQA/MLA/Hybrid Layout Handler 的代码框架

### 3.3 当前工作基础

已有的未提交文件：
```
mooncake-store/include/gqa_layout_handler.h
mooncake-store/include/hybrid_layout_handler.h
mooncake-store/include/kvcache_layout_handler.h
mooncake-store/include/mha_layout_handler.h
mooncake-store/include/mla_layout_handler.h
mooncake-store/src/gqa_layout_handler.cpp
mooncake-store/src/hybrid_layout_handler.cpp
mooncake-store/src/kvcache_layout_handler.cpp
mooncake-store/src/mha_layout_handler.cpp
mooncake-store/src/mla_layout_handler.cpp
mooncake-store/tests/gqa_layout_handler_test.cpp
mooncake-store/tests/kvcache_layout_handler_test.cpp
```

设计文档：
- `docs/superpowers/specs/2026-03-16-qwen35-hybrid-attention-kvcache-optimization.md`
- `docs/superpowers/plans/2026-03-17-qwen35-hybrid-attention-kvcache-implementation.md`

---

## 4. 策略路径选择

### 4.1 三条路径对比

| 路径 | 模块 | 优势 | 风险 | 时间线 |
|------|------|------|------|--------|
| **A：深耕 Store** | mooncake-store | 社区投资最大、演进最活跃、已有基础 | 竞争者多 | 3-6 个月 |
| B：切入 EP/PG | mooncake-ep/pg | CODEOWNER 少，门槛低 | 模块已 deprecated | 2-4 个月 |
| C：NVLink/TENT | mooncake-transfer-engine | 先发优势 | 需硬件资源，架构不稳定 | 4-8 个月 |

### 4.2 选定路径：A — 深耕 Store 模块

**理由**：
1. Store 模块社区投资最大、演进最活跃
2. KVCache layout 直接影响所有推理框架的集成效果
3. 我已经在这个方向有实质性产出
4. 与我的 RDMA + 存储系统背景匹配

---

## 5. 执行计划

### 第一阶段：建立可见度（1-2 个月）

**目标**：完成 Hybrid Attention KVCache 工作，获得 2-3 次高质量合并

| 周次 | 行动项 | 交付物 |
|------|--------|--------|
| W1-W2 | 完善代码实现 | gqa/mla/hybrid_layout_handler 完整实现 |
| W2-W3 | 发起 RFC Issue | RFC: KVCache Layout Handler for Hybrid Attention |
| W3-W4 | 根据 feedback 修改 | RFC 定稿 |
| W4-W6 | 提交 PR | `[Store] feat: Add KVCache layout handlers for GQA/MLA/Hybrid attention` |
| W6-W8 | 补齐测试和文档 | 完整测试覆盖 + 使用文档 |

**RFC Issue 模板**：
```markdown
# RFC: KVCache Layout Handler for Hybrid Attention (GQA+MLA+GDN)

## Motivation
Qwen3.5 等新模型采用混合注意力架构，当前 Store 的 layout 抽象
无法高效表达 GDN 线性注意力的状态矩阵存储需求。

## Proposed Design
参考：docs/superpowers/specs/2026-03-16-qwen35-hybrid-attention-kvcache-optimization.md

## Impact
- 支持 128K-256K 超长上下文
- 传输量减少约 75%
- 存储效率提升 4 倍

## Request for Comments
@ykwd @stmatengss — 请帮忙 review 这个方向是否符合社区路线
```

### 第二阶段：建立信任（2-4 个月）

**目标**：持续贡献 Store 模块，开始 review 他人的 PR

| 月份 | 贡献目标 | 信任建设 |
|------|---------|---------|
| M2 | 2-3 个功能/优化 PR | 主动评论 Store 相关 RFC |
| M3 | 2-3 个功能/优化 PR | Review 2-3 个他人的 Store PR |
| M4 | 1-2 个重大 PR | 获得 @ykwd 或 @stmatengss 的公开认可 |

**高价值贡献方向**：

1. **Storage Backend 性能优化**
   - 延续 #1388 storage backend benchmark suite 的工作
   - 为不同 backend (posix/hf3fs/cachelib) 添加性能对比

2. **Batch API 扩展**
   - #1417 刚加入 batch query keys
   - 可扩展为 batch put/get

3. **错误处理完善**
   - #1328 刚加入 retry logic
   - 可扩展为指数退避、熔断等

4. **TENT 集成**
   - #1398 TENT/Store integration improvements
   - 可跟进 metrics 暴露

### 第三阶段：正式申请（4-6 个月）

**目标**：被提议加入 CODEOWNERS

**触发条件**（需同时满足）：
- [ ] 累计 20+ Store 相关 commits
- [ ] 获得 @ykwd 或 @stmatengss 的公开认可
- [ ] 有 3+ 个重大 PR 作为代表作：
  1. Hybrid Attention KVCache Layout Handler
  2. Storage Backend 性能优化
  3. Batch API 扩展 或 TENT 集成
- [ ] 持续 review 他人的 PR (5+ 次)

**申请方式**（通常由现有 CODEOWNER 提议）：
```markdown
# PR: Add @your-username to CODEOWNERS for /mooncake-store

Rationale: @your-username has been consistently contributing to Store module
over the past X months, with Y commits covering:
- KVCache layout handler for hybrid attention models
- Storage backend performance optimization
- Batch API extension

Co-authored-by: @ykwd
```

---

## 6. 人际关系策略

### 6.1 关键人物

| 人物 | 角色 | 建立联系方式 |
|------|------|-------------|
| @ykwd (Ke Yang) | Store 主负责人 | RFC 主动 @，PR 中展示设计深度 |
| @stmatengss (Teng Ma) | 阿里云 LLM 生态 | 在 Integration 相关 PR 中协作 |
| @XucSh | Store CODEOWNER | 可作为第二 reviewer |
| @JinYan Su | 高贡献者（非 CODEOWNER） | 可作为技术合作者 |

### 6.2 沟通策略

1. **RFC 阶段**：主动 @ykwd 和 @stmatengss，展示设计文档的完整性
2. **PR 阶段**：确保每次 PR 都有清晰的 commit message 和测试
3. **Review 阶段**：对他们的 feedback 给出及时、专业的回应
4. **日常**：在 GitHub Discussion 或 Issue 中主动回答 Store 相关问题

### 6.3 避免的行为

- ❌ 在没有 RFC 的情况下提交大型 PR
- ❌ 与现有 CODEOWNER 发生技术争执（可私下讨论）
- ❌ 提交低质量或测试不足的 PR
- ❌ 忽视 pre-commit 钩子导致的 CI 失败

---

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| @ykwd 不认可设计方向 | 中 | 高 | RFC 阶段充分沟通，接受修改 |
| JinYan Su 等高贡献者竞争 | 高 | 中 | 专注差异化方向 (KVCache Layout) |
| 组织关系影响决策 | 低 | 高 | 利用已合作组织背景 |
| 时间投入不足 | 中 | 高 | 确保 5-15h/周 的稳定投入 |
| 代码质量问题 | 中 | 高 | 严格遵循 pre-commit 和测试规范 |

---

## 8. 成功指标

| 指标 | 当前 | 3个月目标 | 6个月目标 |
|------|------|----------|----------|
| Store commits | ~10 | 20+ | 40+ |
| Merged PRs | ~5 | 10+ | 20+ |
| Reviewed PRs | 0 | 3+ | 10+ |
| RFC 参与 | 0 | 2+ | 5+ |
| CODEOWNERS 认可 | 无 | 1位认可 | 正式加入 |

---

## 9. 下一步行动

1. **本周**：完善 Hybrid Layout Handler 代码实现
2. **下周**：发起 RFC Issue
3. **持续**：每周至少 1 个有质量的 commit 或 PR review

---

## 附录：参考资料

- [MAINTAINERS.md](../../MAINTAINERS.md)
- [CONTRIBUTING.md](../../CONTRIBUTING.md)
- [.github/CODEOWNERS](../../.github/CODEOWNERS)
- [KVCache Architecture Optimization Design](./2026-03-10-kvcache-architecture-optimization.md)
- [Qwen3.5 Hybrid Attention KVCache Optimization](../specs/2026-03-16-qwen35-hybrid-attention-kvcache-optimization.md)

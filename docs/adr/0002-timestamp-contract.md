# ADR-0002 · Timestamp Contract（`ts` / `message_ts` / `ingest_ts` 精度 SLA）

- **Status**: Accepted
- **Date**: 2026-05-11
- **Issue**: [#23](https://github.com/phead198708/BiGan/issues/23)
- **Milestone**: `mvp-v1`
- **Owners**: data-ingestion

---

## 1. 决策摘要（TL;DR）

| 字段 | 含义 | 来源 | 精度 | NULL 允许？ |
|---|---|---|---|---|
| `ts` | **Canonical 事件时间** — 下游所有特征/标签/回测的唯一时间基准 | 上游数据源服务器时间（Polymarket WS `timestamp`，Coinbase WS `time`，Chainlink `updatedAt`，等） | `int64` UTC ms epoch | **No** |
| `message_ts` | 消息在协议层携带的原始时间戳；用于多源对齐与诊断 | WS payload 原字段（Polymarket 等同 `ts`；多源场景下可能与 `ts` 差异） | `int64` UTC ms epoch | **No** |
| `ingest_ts` | 本地 BiGan 服务收到/解析消息并落入 NDJSON sink 的时间 | `int(time.time() * 1000)` 在 `clob_ws._dispatch` 内计算 | `int64` UTC ms epoch | **No** |

所有时间戳：

- **UTC**，禁止任何带时区偏移的字符串
- **ms 精度**（不是 s）— REST/HF 等以 s 为单位的源在 ingestion 边界即转换
- **单调性**：单个 asset 内 `ts` 不保证严格单调（exchange 可能乱序），但 `ingest_ts` 必须单调（本地时钟）
- **延迟预算**：正常情况下 `ingest_ts - message_ts ≤ 500ms`；超过即触发 warn log + Prometheus 告警

---

## 2. 背景与动机

多数据源时代（#22 / #24）即将到来。一旦同时接入 Polymarket / Binance / Coinbase / Chainlink，时间戳来源会变得异构：

- Polymarket WS 发送的是**交易所服务器墙钟**
- Binance WS 的 `E` 字段是**事件推送时间**，trade 类消息另有 `T` 交易时间
- Coinbase WS 的 `event_time` / trade `time` 字段是**撮合引擎时钟**
- Chainlink 的 `updatedAt` 是**链上 oracle round 落账时间**
- 本地接收时刻是**部署机器系统时钟**

如果不在 schema 层强约定，下游会出现三类典型故障：

1. **时区污染** — 任一源不小心存了带 `+08:00` 的 ISO 字符串，与 UTC ms 混用时静默错位 8 小时
2. **数据泄漏（lookahead bias）** — 特征任务误用 `ingest_ts` 作为事件时间，把模型在 `t` 时刻的预测耦合到 `t + δ` 才到达的数据
3. **跨源对齐失败** — Polymarket tick 用 server time, Coinbase tick 用 engine time，`as-of join` 输出错误的"最近一笔现货价"

本 ADR 把三个时间戳的语义、来源、精度、SLA 写死为合同（contract），ingestion / ETL / 校验 / 监控全部围绕它实现。

---

## 3. 字段定义

### 3.1 `ts` — Canonical 事件时间

- **来源优先级**：原始 payload 的协议时间戳 > 上游 API 的 `updatedAt` > 不可恢复时使用 fallback（Polymarket 的某些 `book` 消息缺 `timestamp`，使用 `ingest_ts` 替代）
- **含义**：「这件事在它的源系统里发生于这一刻」
- **用途**：特征工程的 `as-of` join 键、标签生成的窗口边界、回测的 walk-forward 切折锚点
- **不变量**：
  - `ts > 0`
  - `ts ≤ ingest_ts + FUTURE_GRACE_MS`（防止上游时钟前漂导致泄漏）
  - `ingest_ts - ts ≤ STALE_THRESHOLD_MS`（防止远古数据被当作实时灌入）

### 3.2 `message_ts` — 协议层原始时间戳

- **来源**：WS payload 中标识本条消息的字段，原样照搬，不做语义解释
- **与 `ts` 的关系**：
  - 单源（Polymarket）下 `message_ts == ts`
  - 多源（Coinbase/Kraken/Chainlink）下二者可能有 ms 级差异（撮合 vs 推送）
- **用途**：跨源诊断「ts 到底是谁的钟」、与上游 SDK 输出对账

### 3.3 `ingest_ts` — 本地接收时间

- **来源**：`int(time.time() * 1000)` 在 `clob_ws._dispatch` 内对每条消息单独计算
- **用途**：
  - 监控（`bigan_ingest_lag_seconds = (ingest_ts - message_ts) / 1000`）
  - 断流检测（#5 的 `GapDetector` 内部使用 `ingest_ts` 作为「最后一次收到该 asset 消息」的判定）
  - 重放（NDJSON `receive_time` 字段就是 `ingest_ts`）

---

## 4. 校验与监控

### 4.1 验证规则（`bigan.canonical.validation`）

| 规则 | 触发条件 | 行为 |
|---|---|---|
| `EMPTY_TIME` | `ts` 为 NULL 或 `≤ 0` | 隔离到 quarantine 表 |
| `TS_IN_FUTURE` | `ts > ingest_ts + FUTURE_GRACE_MS` | 隔离到 quarantine 表 |
| `TS_TOO_STALE` | `ingest_ts - ts > STALE_THRESHOLD_MS` | 隔离到 quarantine 表 |

阈值通过 `RowValidator(future_grace_ms=..., stale_threshold_ms=...)` 注入；ETL `run_etl_batch` 暴露 `timestamp_future_grace_seconds` / `timestamp_stale_threshold_seconds` 参数。

### 4.2 Prometheus 指标

- `bigan_ingest_lag_seconds{source,event_type}` (Histogram) — `(ingest_ts - message_ts) / 1000`，每条 WS 消息观察一次。Bucket 默认覆盖 `1ms` ~ `10s`。
- 现有 `bigan_last_event_receive_time_seconds` (Gauge) 继续用作 liveness alarm（最后一条事件多久前到达）。

### 4.3 结构化日志

`ingest_lag.high` — 当单条消息 lag 超过 `BIGAN_INGEST_LAG_WARN_SECONDS`（默认 500ms）时发一次 warn log，extras 含 `asset_id`, `event_type`, `lag_ms`。

---

## 5. 跨源映射（#24 落地时填表）

| 数据源 | `ts` 来源字段 | 单位 | `message_ts` 来源字段 | 备注 |
|---|---|---|---|---|
| Polymarket CLOB WS | `payload.timestamp` | ms string → ms | `payload.timestamp` | 单一时间戳，二者相等 |
| Binance Spot WS | `T`（trade / aggTrade 成交时间）或 `E`（无交易时间的行情事件） | ms（默认） | `E` | Binance 可用 URL 参数切到 microsecond；reader 必须统一降到 ms |
| Coinbase Advanced Trade WS | `updates[].event_time` / `trades[].time` (RFC 3339) | RFC 3339 → ms | envelope `timestamp` | `level2` 的 `event_time` 明确为交易引擎时间 |
| Polygon Chainlink (`latestRoundData`) | `updatedAt` (block seconds) | s → ms | `updatedAt` | 也可写入 `block_number` 作为额外溯源（在表层补充字段） |

### 5.1 上游字段确认

- Polymarket CLOB market channel：`book` / `price_change` / `last_trade_price` / `best_bid_ask` 示例都携带 `timestamp`，当前 reader 将其解析为 ms epoch。
- Binance Spot WS：官方文档说明所有 time/timestamp 字段默认是 milliseconds，除非显式请求 `timeUnit=MICROSECOND`。
- Coinbase Advanced Trade WS：消息 envelope 有 RFC 3339 `timestamp`；`market_trades` 使用 trade `time`，`level2` 使用 `event_time`，其中 `event_time` 是交易引擎记录的事件时间。
- Chainlink Data Feeds：`latestRoundData()` 返回 `updatedAt`，语义是 round 更新时刻；reader 必须从 seconds 转换成 UTC ms epoch。

**禁止：**

- 把 RFC 3339 字符串直接存进 `ts` 列（必须先解析为 int ms）
- 用 `datetime.now()`（无时区）→ 必须 `time.time()` 或 `datetime.now(timezone.utc)`
- 在特征 / 标签 / 回测代码里直接读 `ingest_ts`（除非 ADR-0002 显式批准的诊断场景）

---

## 6. 兼容性与迁移

- v1 schema 已经包含三字段（见 `schemas.py:_COMMON_IDENTITY_FIELDS`），无需 schema migration
- 历史 Parquet 文件无需重写；新增的两条 quarantine rule 仅作用于新 ETL 批次
- 阈值配置默认值（5s future grace / 10min stale）经过 #5 backfill 场景验证：手动 replay 几个小时前的 gap 仍在容差内，但若 replay 跨越数天则会被 quarantine（设计意图）

---

## 7. 风险与开放问题

1. **上游时钟前漂** — 若 Polymarket 服务端时钟比 NTP 快 > 5s，`TS_IN_FUTURE` 会误判。缓解：监控 `bigan_ingest_lag_seconds` p1 分位是否长期为负数（lag 为负即代表 ts > ingest_ts）。
2. **跨 ms 边界舍入** — `_as_int_ms("1700000000.999")` 会拒绝（当前实现要求 int 字符串）。多源接入时 reader 边界要先 round。
3. **Backfill 的 stale 误报** — 手动用 `bigan-ingest backfill` replay 一周前的 gap，所有 trade 都会被 quarantine。已知限制；运维侧若需 replay 旧数据，应临时调高 `BIGAN_TIMESTAMP_STALE_THRESHOLD_SECONDS` 环境变量。
4. **特征任务的 `ts` 单调性假设** — 模型若要求严格单调输入，需在特征侧排序（不在 ingestion 层处理乱序）。

---

## 8. 参考

- ADR-0001 §4 字段 Schema（v1）
- `src/bigan/canonical/schemas.py:_COMMON_IDENTITY_FIELDS`
- `src/bigan/canonical/validation.py:RowValidator`
- [Polymarket CLOB market channel documentation](https://docs.polymarket.com/developers/CLOB/websocket/market-channel-migration-guide)
- [Binance Spot WebSocket Streams documentation](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md)
- [Coinbase Advanced Trade WebSocket Channels documentation](https://docs.cdp.coinbase.com/coinbase-business/advanced-trade-apis/websocket/websocket-channels)
- [Chainlink Data Feeds API Reference (`latestRoundData`)](https://docs.chain.link/data-feeds/api-reference#latestrounddata)
- Issue #23、#22、#24、#5

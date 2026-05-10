# ADR-0001 · 预测市场数据源选型（Polymarket 15m BTC + 现货参考价）

- **Status**: Accepted
- **Date**: 2026-05-10 (rev 2 — HF dataset re-evaluated and adopted)
- **Issue**: [#21](https://github.com/phead198708/BiGan/issues/21)
- **Milestone**: `mvp-v1`
- **Owners**: data-ingestion

---

## 1. 决策摘要（TL;DR）

| 数据维度 | 选型 | 理由 |
|---|---|---|
| 主行情数据源 | **自建：Polymarket CLOB WebSocket（market channel）+ REST 兜底** | 唯一一手数据源；第三方接口 PredictAPI 已不可用、MarketLens 无开放 API；HuggingFace 数据集已损坏 |
| 市场元数据 | **Polymarket Gamma API**（`gamma-api.polymarket.com`） | 公开免认证，提供 `slug` / `startDate` / `endDate` / `clobTokenIds` / `outcomes` |
| BTC 现货参考价（实时特征） | **Coinbase Exchange `BTC-USD`（主）+ Kraken `XBTUSD`（备）** | Binance / Bybit 在当前部署区域被 CloudFront / Binance ToS 地理封锁；Coinbase 与 Kraken 在 mvp-v1 部署网络下均 200 OK |
| BTC 结算/锚定参考价（标签生成） | **Chainlink BTC/USD on Polygon**（feed `0xc907E116054Ad103354f2D350FD2514433D57F6f`）| 与 Polymarket UMA/Chainlink 结算路径一致，避免标签与结算口径漂移；通过公共 RPC `https://polygon.drpc.org` 即可读 `latestRoundData()` |
| 历史回测起点数据集 | **采用 HuggingFace `kaboomfox/15btc_eth`** | 990 个有效 shard / 约 61 GB / ≈ 6.9 亿行预工程化特征 + 标签；覆盖 BTC/ETH/SOL/XRP/BNB/DOGE/HYPE × 5min/15min；schema 与 issue #21 关键字段一一对应，并扩展了期货 / 跨市场特征 |

---

## 2. 背景与约束

`btc-updown-15m-{epoch}` 市场每 15 分钟新开一只，单只市场生命周期仅 15 分钟。订单簿是 **YES (Up) / NO (Down) 份额** 的概率盘，价格区间 0.0–1.0，与 BTC 现货交易所订单簿 **不可互换**。

关键约束：

- 单市场存活时间 ≈ 15 min → CLOB `prices-history` 每只市场最多回看 15 min，**没有跨市场长期历史**。任何回测都必须依赖 **我们自己累积的滚动数据**。
- 该类市场 `gamma-api` 返回 **不包含显式 `target_price` (strike) 字段**；这类市场结算规则为「`endDate` 时刻 BTC 价 vs `startDate` 时刻 BTC 价」，因此 strike = 我们在 `startDate` 锁定的 BTC 现货价。
- 部署网络对部分 CEX（Binance / Bybit）地理封锁，必须避开。

---

## 3. 候选评估

### 3.1 Polymarket 官方 CLOB（自建）— **采用**

**REST 关键端点**（base: `https://clob.polymarket.com`）：

| 端点 | 用途 | 实测 |
|---|---|---|
| `GET /book?token_id={id}` | 全量订单簿快照（bids/asks 数组） | ✅ 返回 `market`/`asset_id`/`timestamp`/`hash`/`bids`/`asks` |
| `GET /midpoint?token_id={id}` | 中间价 | ✅ `{"mid":"0.505"}` |
| `GET /price?token_id={id}&side=BUY|SELL` | 最优买/卖价 | ✅ |
| `GET /prices-history?market={id}&interval=...&fidelity=1` | 单市场价格历史 | ✅ 但单市场最多 15 min |

**WebSocket**（endpoint: `wss://ws-subscriptions-clob.polymarket.com/ws/market`）：

订阅消息：
```json
{ "assets_ids": ["<token_id_1>", "<token_id_2>"], "type": "market", "custom_feature_enabled": true }
```

事件类型：

- `book` — 全量订单簿快照（用作初始化 / 重对齐基准），含 `event_type`/`asset_id`/`market`/`bids[]`/`asks[]`/`timestamp`/`hash`
- `price_change` — 增量更新（同一价位 `size="0"` 表示删除），含 `best_bid`/`best_ask`
- `last_trade_price` — 成交，含 `price`/`side`/`size`
- `best_bid_ask` — TOB 单独推送（需 `custom_feature_enabled: true`），含 `best_bid`/`best_ask`/`spread`
- `tick_size_change`、`new_market`、`market_resolved` — 元数据/生命周期

**优势**：一手、低延迟、字段完整；客户端有官方 TS / Python / Rust SDK（无需鉴权读公共行情）。

**需自行处理**：

- `book` 与 `price_change` 的对齐（用 `hash` 校验快照后逐增量重放，hash 不一致时重 snapshot）
- 心跳 / 断线重连
- 多 market 动态订阅 / 退订（每 15 min 一批新 market）

### 3.2 PredictAPI — **不采用**

`https://www.predictapi.dev` 实测无法连接（DNS/EOF）；仓库与文档无活跃维护痕迹。结论：**不可作为依赖**。

### 3.3 MarketLens — **不采用**

未找到开放 REST/WS 文档与公开定价；不符合「mvp-v1 优先快速接入」目标。如后续需要交叉验证再评估。

### 3.4 HuggingFace `kaboomfox/15btc_eth` — **采用（冷启动训练 / 回测主数据集）**

> 注：HF dataset viewer 因首个 shard `shard_11000.parquet` 为 0 字节而报错 `Parquet file size is 0 bytes`，**这只是 viewer 的自动解析失败**；通过 `git clone` / HF Resolve URL 实测仓库内有 990 个非空 shard。

**实测规模**（API 列表 + 单 shard 解析）：

| 指标 | 实测 |
|---|---|
| 文件总数 | 999 parquet shard（`shard_11000` ~ `shard_11998`） |
| 有效 shard | **990 个非空**（仅 9 个 0 字节哨兵：11000/11002/11050/11674/11675/...） |
| 单 shard 大小 | 30–130 MB |
| 单 shard 行数 | **700,000** 行 |
| 全量行数 | ≈ **6.9 亿行** |
| 全量大小 | ≈ **61 GB**（LFS / xet 后端） |
| 时间窗口（样本 shard） | 单 shard ≈ 28 min；全集覆盖度需进一步抽样 5–10 个 shard 评估 |
| 资产 | BTC, ETH, SOL, XRP, BNB, DOGE, HYPE |
| 时间框架 | `15min`, `5min` |
| Exchange | `polymarket` |
| 单 shard 唯一 condition_id 数 | 42（即 42 只市场样本） |
| 单 market 样本量 | 中位数 6,550 行 / 最大 104,814 行 |
| 数据集激活度 | `lastModified=2026-05-10`、`downloads=1364`，活跃维护 |

**Schema（68 列，全部已实测）**：

核心订单簿（覆盖 issue #21 全部要求）：`ts`, `message_ts`, `type`, `seq`, `dt_ms`, `progress`, `outcome_up`, `outcome_down`, `side`, `price`, `size`, `best_bid`, `best_ask`, `spread`, `bid_levels`, `ask_levels`, `bid_size_total`, `ask_size_total`, `best_bid_size`, `best_ask_size`, `has_book`, `mid`, `imbalance`。

BTC 参考价（多源冗余）：`oracle_price`, `binance_price`, `coinbase_price`, `bybit_price`, `binance_aggtrade_price`, `mark_price`, `index_price`, `target_price`。

标签 / 元数据：`outcome`（0=Down 赢, 1=Up 赢，**已预计算**）, `condition_id`, `timeframe`, `asset`, `exchange`。

期货侧 / 大盘流：`futures_basis`, `funding_rate_bps`, `taker_buy_ratio`, `binance_sell_ratio` (含 `_1s` / `_10s`), `bybit_sell_ratio` (含 `_1s` / `_10s`), `liq_long_volume_cents`, `liq_short_volume_cents`, `binance_bid_depth_5`, `binance_ask_depth_5`, `open_interest_btc`, `long_short_ratio`。

Polymarket 流：`poly_trade_count`, `poly_buy_volume`, `poly_sell_volume`, `poly_buy_volume_5s/15s`, `poly_sell_volume_5s/15s`, `poly_max_trade_size`, `poly_whale_count`, `poly_yes_depth_3`, `poly_no_depth_3`, `poly_book_concentration`, `depth_bid_delta`, `depth_ask_delta`。

跨市场：`concurrent_market_count`, `concurrent_yes_pct`, `concurrent_avg_abs_delta`。

**关键约定（必须写入 ingestion / 回测代码）**：

1. **价格统一以「美元 × 100」存储**（cents-style）。例如 BTC `target_price = 7,284,303` 表示 $72,843.03。我们自建 ingestion 必须遵循同一约定，使两份数据可直接拼接。
2. **行粒度是 tick**（事件级），不是固定时间间隔。`message_ts` / `ts` / `seq` 三者共同定位一条消息；`dt_ms` 给与上一条的时间差。
3. **类别不平衡**：在样本 shard 中 BTC 按行计 Up:Down ≈ 1:2.9。注意这是「按 tick」加权（成交量大的市场行多），需在 baseline 训练前以 `condition_id` 去重做「按市场」类别分布检查。
4. **shard 命名 `shard_11000` ~ `shard_11998`**：编号语义未在 dataset card 中说明，按时间排序需抽样多个 shard 拉 `ts` min/max 来确认。

**不利点 / 风险**：

- 无 dataset card，无 license 声明（HF 页 `tags=["region:us"]` 之外无字段含义文档）→ 我们必须用解析脚本反推每列含义并固化为本仓库的「字段字典」（issue #6）。
- 与上游作者无直接联系，**未来更新节奏不可控**。需要把当前 sha=`91e5fcf21021bdc7f662a12d5da157d8f8c93394` 钉住作为 mvp-v1 的训练快照。
- 9 个 0 字节 shard 需在加载脚本里直接跳过（不要 raise）。

**采用策略**：

- 作为 **冷启动训练集 + walk-forward 回测起点**（解锁 issue #15 / #16 / #17 / #11 / #14 立即可做，无需等 7–14 天）。
- 自建 ingestion（issue #2/#3）的 schema **以 HF 68 列为蓝本**，确保「HF 离线训练 → 自建实时推理」迁移零阻抗。
- 数据完整性校验（issue #4）需新增「价格 × 100 一致性」断言，避免 schema 漂移。

### 3.5 BTC 现货价候选

实测部署环境（命令均带 8s 超时）：

| 源 | 端点 | 结果 | 备注 |
|---|---|---|---|
| Binance global | `api.binance.com/api/v3/...` | ❌ HTTP 451 风格地理封锁 | ToS Eligibility 拒绝 |
| Bybit | `api.bybit.com/...` | ❌ HTTP 403 CloudFront block | 国家级封锁 |
| **Coinbase Exchange** | `api.exchange.coinbase.com/products/BTC-USD/{ticker,candles}` | ✅ 200，1m candle 一次拉 350 行 | 主用 |
| **Kraken** | `api.kraken.com/0/public/Ticker?pair=XBTUSD` | ✅ 返回 bid/ask/last | 备用，故障切换 |
| **Chainlink BTC/USD on Polygon** | feed `0xc907E116054Ad103354f2D350FD2514433D57F6f` via `polygon.drpc.org` | ✅ `latestRoundData()` 解码出 8 位小数答案 | 用于结算锚定 |
| Ankr Polygon RPC | `rpc.ankr.com/polygon` | ❌ 现要求 API key | 仅作 Chainlink 备用 RPC，需账号 |

**最终选型**：

- **特征侧（高频）**：Coinbase 1m candle / WebSocket ticker 主，Kraken 备
- **标签侧（结算锚定）**：Chainlink Polygon feed —— 因为 Polymarket 自身 `15m updown` 市场的 UMA / 决议口径与 Chainlink 一致，使用 Chainlink 可让标签 (`y`) 与真实结算 (`r`) 几乎不漂移

---

## 4. 字段 Schema（v1）

> 命名遵循 issue #21 的「关键字段」要求，并补全到落库可用的最小完备集。所有时间戳统一 **epoch milliseconds, UTC**；所有金额/规模统一 **USDC（Polymarket 计价币）**。

### 4.1 `market_orderbook_tick`（CLOB 增量行情，主表）

| 列 | 类型 | 来源 | 说明 |
|---|---|---|---|
| `ts` | int64 ms | WS `timestamp` | 行情时间戳 |
| `slug` | text | Gamma | 例：`btc-updown-15m-1778502600` |
| `condition_id` | text | WS `market` | 0x… ConditionId |
| `asset_id` | numeric(78) | WS `asset_id` | YES/NO 的 ERC1155 token id |
| `outcome` | enum(`UP`, `DOWN`) | Gamma `outcomes[]` 映射 | 由 token_id → outcomes 推出 |
| `best_bid` | decimal(6,4) | WS `best_bid_ask` / 计算 | 0.0001–0.9999 |
| `best_ask` | decimal(6,4) | WS `best_bid_ask` / 计算 | |
| `best_bid_size` | decimal | book 第一档 size | 单位 USDC |
| `best_ask_size` | decimal | book 第一档 size | |
| `spread` | decimal(6,4) | WS `best_bid_ask.spread` 或派生 | |
| `imbalance` | decimal | `(bid_size − ask_size)/(bid_size + ask_size)` | 派生字段，落表加速训练 |
| `mid` | decimal(6,4) | `(best_bid+best_ask)/2` | 派生 |
| `book_hash` | text | WS `hash` | 用于 snapshot/delta 一致性校验 |
| `tick_size` | decimal | `tick_size_change` 跟踪 | 默认 0.01 |
| `event_type` | enum | `book` / `price_change` / `last_trade_price` / `best_bid_ask` | |
| `last_trade_price` | decimal | `last_trade_price.price` | 仅成交事件 |
| `last_trade_size` | decimal | `last_trade_price.size` | |
| `last_trade_side` | enum(`BUY`,`SELL`) | `last_trade_price.side` | |
| `ingest_ts` | int64 ms | 服务器接收时间 | 用于断流检测（issue #5） |

### 4.2 `market_orderbook_snapshot`（深度快照，可选副表）

按 `event_type='book'` 的全量 bids/asks 数组以 `JSONB` 列原样存档，便于后续做更深的微观结构特征。

### 4.3 `market_meta`（每只 15m 市场一行）

| 列 | 类型 | 来源 | 说明 |
|---|---|---|---|
| `slug` | text PK | Gamma | |
| `condition_id` | text | Gamma | |
| `asset_id_up` | numeric(78) | Gamma `clobTokenIds[0]` + `outcomes[0]` | |
| `asset_id_down` | numeric(78) | Gamma `clobTokenIds[1]` + `outcomes[1]` | |
| `start_ts` | int64 ms | Gamma `startDate` | 市场开放（= strike 锚定时刻） |
| `end_ts` | int64 ms | Gamma `endDate` | 结算时刻 |
| `target_price` | decimal | 自建：T0 现货 BTC 价 | 在 `start_ts` 锁定 Chainlink + Coinbase 双口径 |
| `progress` | decimal in [0,1] | 派生：`(now − start_ts)/(end_ts − start_ts)` | 不落表，特征任务计算 |
| `tick_size` | decimal | Gamma `orderPriceMinTickSize` | 默认 0.01 |
| `created_ts` | int64 ms | Gamma `createdAt` | |

### 4.4 `btc_spot_tick`（BTC 现货，特征侧）

| 列 | 类型 | 说明 |
|---|---|---|
| `ts` | int64 ms | |
| `source` | enum(`coinbase`, `kraken`) | 多源冗余 |
| `bid` / `ask` / `last` | decimal | |
| `binance_price` | decimal **NULLABLE** | 部署区域可达时回填，否则永远 NULL（issue #21 字段兼容） |

### 4.5 `btc_oracle_tick`（BTC 锚定，标签侧）

| 列 | 类型 | 说明 |
|---|---|---|
| `ts` | int64 ms | feed `updatedAt` × 1000 |
| `oracle_price` | decimal | Chainlink `answer / 1e8` |
| `round_id` | numeric(80) | Chainlink `roundId` |
| `block_number` | int64 | 读取时的 Polygon block |

### 4.6 与 issue #21「关键字段」的对应

| issue 字段 | 落表位置 |
|---|---|
| `ts` | 各 tick 表 `ts` |
| `best_bid` / `best_ask` | `market_orderbook_tick.best_bid/ask` |
| `best_bid_size` / `best_ask_size` | 同上 `_size` |
| `imbalance` | `market_orderbook_tick.imbalance` |
| `oracle_price` | `btc_oracle_tick.oracle_price` |
| `binance_price` | `btc_spot_tick.binance_price`（可为 NULL）+ 约定：当 NULL 时下游使用 Coinbase last 作为代理 |
| `target_price` | `market_meta.target_price` |
| `progress` | 派生：`(now − market_meta.start_ts)/(end_ts − start_ts)` |

---

## 5. 历史数据 / 回测可用方案

### 5.1 现实约束

- Polymarket CLOB 不提供「跨 market 的全市场历史 orderbook archive」。
- 单 market `prices-history` 只能回看该 market 的 ≤15 min。
- ~~HuggingFace 候选数据集不可用~~ → **已采用 `kaboomfox/15btc_eth`（详见 §3.4）**。

### 5.2 起步阶段（mvp-v1）

1. **冷启动训练 / 回测 = HuggingFace 离线快照**：钉住 sha `91e5fcf21021bdc7f662a12d5da157d8f8c93394`，下载到 `data/raw/hf_kaboomfox_15btc_eth/`（不入 git，写 `.gitignore`），先抽 5–10 个 shard 做时间分布勘察后决定全量 vs 子集训练。
2. **Day 0 同步上线 ingestion**：issue #2/#3 仍是 P0，目标是**双轨并行**——HF 训练 baseline 模型 → 自建 ingestion 提供实时推理 + 持续累积 → 周期性 re-train（fine-tune）。
3. **冷启动可立刻做的 ticket**：
   - #15 训练数据集装配脚本（直接读 HF parquet → 拼按时间分片）
   - #16 LR baseline、#17 XGBoost v1（HF 数据量足够避免欠拟合）
   - #11 纯预测指标评估、#14 walk-forward 回测（按 `ts` 切折）
4. **schema 对齐**：自建 ingestion 表必须复刻 HF 68 列子集 + 价格 ×100 cents 约定，**才能把 HF 训练的模型直接用在实时数据上**。

### 5.3 中期增强

- **持续数据累积**（最高优先级）：自建 ingestion 持续写入与 HF 同 schema 的实时数据。当本地累积量 ≥ HF 子集时，新增 `data_source` 列区分 `hf-snapshot` / `live` 做交叉验证。
- **HF dataset 增量同步**：每周 cron 检查 HF repo 新 commit，决定是否把更多 shard 加入训练集。
- **历史回填备选**（性价比降序）：
  1. 联系上游作者（kaboomfox）获取数据生成代码 / 持续更新意向。
  2. Polymarket 子图 / The Graph：`condition` 与 `orderFilled` 事件可还原成交流，但还原不出完整 LOB → 用于成交序列回测。
  3. 第三方付费数据（Kaiko / Amberdata）——出 mvp-v1 后再评估。

### 5.4 回测一致性保证

- 回测仅使用 `market_orderbook_tick` + `btc_oracle_tick` + `btc_spot_tick`，**严禁用 `last_trade_price` 之外的未来字段**（防 lookahead）。
- 标签生成（issue #9）规定使用 `btc_oracle_tick` 在 `end_ts` 之后第一笔 round 的 `oracle_price`，与 Polymarket 实际结算路径一致。
- 特征对齐采用 `as-of join`：以 `market_orderbook_tick.ts` 为基准向左 join 最近一笔 BTC spot / oracle。

---

## 6. 待办与下游 ticket 影响

本 ADR 解锁的 ticket（双轨：HF 离线 + 自建实时）：

**HF 数据轨（无依赖，可立刻并行启动）**：

- **#15** 训练数据集装配 → HF parquet 加载器 + 按 `ts` 切折 + 跳过 0 字节 shard + 价格 ×100 反推校验
- **#16** LR baseline / **#17** XGBoost v1 → HF 直接喂入
- **#11** 纯预测指标评估 / **#14** walk-forward 回测 → 全量 HF 时间分片
- **#6** 特征字典 v1 → **以 HF 68 列为蓝本**反推语义并固化为仓库内 schema 文档

**实时 ingestion 轨**：

- **#2** WebSocket 行情接入 → 订阅 `market` channel，处理 `book`/`price_change`/`best_bid_ask`/`last_trade_price`，断线重连、`hash` 校验
- **#3** 原始市场数据表 → 列对齐 HF 68 列子集（先实现 issue 提到的 8 列 + `outcome`/`condition_id`/`timeframe`/`asset`）
- **#4** 数据校验 → 关键断言：`best_bid < best_ask`、`tick_size` 对齐、`imbalance ∈ [-1,1]`、`book_hash` 匹配、**价格统一 ×100 (cents)**
- **#5** 断流检测 → WS heartbeat + `ingest_ts` gap 阈值（>5s 告警 / >30s 补数）
- **#9** 标签生成 → 与 HF `outcome` 列同口径（`endDate` 时刻 oracle_price vs target_price），仍以 Chainlink 为锚

**新增建议 ticket**：

- HF 数据集元信息分析（多 shard 抽样确认时间覆盖、数据缺口、价格 ×100 在所有资产上一致）
- 接入 **Coinbase Exchange WebSocket**（特征侧高频，避免 1m candle 延迟）
- 接入 **Kraken** 兜底
- 接入 **Polygon Chainlink reader**（轻量 RPC poll，每 30s 一次足够）
- 评估 **Binance** / **Bybit** 在生产环境是否可用，决定其字段是否长期填充（HF 内已有数据，但实时端可能拿不到 → 训练用 / 推理时置 NaN）
- 评估期货侧数据来源（funding / OI / 清算）以匹配 HF 已有的 `funding_rate_bps` / `open_interest_btc` / `liq_*_volume_cents` 列

---

## 7. 风险与开放问题

1. **Coinbase 单点风险**：若被限流需立即切 Kraken。建议两源同时跑，落表时用 `source` 列区分。
2. **Polygon 公共 RPC 限流**：`polygon.drpc.org` 公测无 key 可用，但生产建议接入自建 RPC 或托管服务（Alchemy/QuickNode）做主，公共 RPC 兜底。
3. **每 15 min 动态订阅新 market**：CLOB WS 支持 `unsubscribe`/`subscribe`；需在 ingestion 服务里实现「定期拉 Gamma → diff 当前订阅集合 → 调整」的调度循环（建议每 60s 一次）。
4. **Binance / Bybit 字段兼容**：HF 数据集这些列有值，但部署区域被封锁 → 实时推理时这些列将为 NaN。模型必须**在训练阶段也对这些列做随机 mask 训练**，避免推理崩溃；或单独训练「无 Binance/Bybit」子模型。
5. **HF schema 漂移风险**：上游若改列名/语义，会破坏「HF 训练 → 自建推理」迁移。缓解：钉 sha + 在仓库写 schema contract 单测，每次拉新数据自动校验。
6. **HF 价格 ×100 约定无文档**：必须在自建 ingestion 上线第一天就强制同样约定，否则未来训练/推理拼接会产生静默 100 倍误差。建议在 ORM 层加 `assert price < 1e10` 类型护栏。

---

## 8. 参考

- Polymarket API 总览：https://docs.polymarket.com/api-reference/introduction
- CLOB WebSocket Market Channel：https://docs.polymarket.com/developers/CLOB/websocket/market-channel
- Gamma API base：`https://gamma-api.polymarket.com`
- Chainlink Polygon BTC/USD feed：`0xc907E116054Ad103354f2D350FD2514433D57F6f`（8 decimals）
- Coinbase Exchange API：`https://docs.cdp.coinbase.com/exchange/`
- Kraken Public API：`https://docs.kraken.com/rest/`
- HuggingFace 数据集：`https://huggingface.co/datasets/kaboomfox/15btc_eth`（钉住 sha `91e5fcf21021bdc7f662a12d5da157d8f8c93394`）
- HF 数据集 API：`https://huggingface.co/api/datasets/kaboomfox/15btc_eth/tree/main?recursive=true`（用于程序化拿全量 shard 列表 + 大小，避免依赖 git-lfs）

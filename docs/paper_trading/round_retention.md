# 每个 round 的归档范围

当前 Operator 的一个 round 对应一个 MarketWindow 和一个独立 paper run。
**已实现的是按 round 保存交易审计记录，不是完整原始行情归档。**

## 保存位置和生命周期

```text
<output_dir>/
  <operator_id>/
    account_checkpoint.json       # 当前账户进度，原子替换
    operator_status.json          # 最新状态/行情投影，原子替换，不是历史
  paper-<run-id>/                  # 每个 round 独立目录，旧目录不因 rollover 删除
    operator_account_link.json    # 本窗口身份、开盘参考价、前驱 run、初始资金
    paper_run_manifest.json       # source commit、配置哈希、安全边界
    signal_events.jsonl           # 已进入 Runner 的决策及决策输入
    execution_events.jsonl        # FILLED / REJECTED / NO_ORDER
    settlement_events.jsonl       # 收到有效最终 resolution 后的结算事实
    ledger_events.jsonl           # 决策/结算后的账户账本
    position_snapshots.jsonl      # 决策/结算后的仓位观察
    pnl_snapshots.jsonl           # 决策/结算后的 PnL 观察
    paper_snapshot.json          # 本 run 最新账户快照，结算后为最终账户状态
    paper_idempotency.sqlite3    # 可从历史决策重建的幂等索引
```

数据在运行期间持续落盘，不是等 round 结束才一次性备份。Operator 使用
fsync；先持久化最终结算，再激活下一个 run。新 run 的前驱链接支持有界跨
run 查询和分页；最新 status 的切换不代表旧交易文件被清空。当前没有自动
删除旧 run 的保留期策略，也没有远程备份/独立压缩归档协议。

## 基础信息覆盖表

| 信息 | 当前持久化情况 | 限制 |
|---|---|---|
| 市场、window 起止、YES/NO token、resolution 身份 | `operator_account_link.json` 中的 market | 仅记录已激活的窗口，不保证停机期间所有市场都有 run |
| 开盘价 / Price to beat | run link 的 `reference_price_at_start` 和 `opening_reference` | TWAP 市场保存已核验值、请求/来源时间、endpoint 和 payload hash；不是原始响应完整副本或签名报告 |
| Binance.US Spot、Chainlink TWAP | `signal_events.jsonl` 中每个决策的输入 | 缺失输入字段可为 null；没有完整独立时间序列 |
| UP/YES、DOWN/NO 的 bid/ask 和 size | 每个持久化决策中的八个报价/数量字段 | 不是每条 CLOB 消息；Operator gate 前丢弃的数据不会生成 Session decision |
| OFI、波动率、概率、edge、方向、仓位建议 | 每个决策的对应字段 | 原始 Binance depth、OFI 更新事件、预热样本未归档，不能完整重放 alpha |
| 成交、拒单、手续费、现金、仓位、PnL | execution、ledger、position、pnl 文件 | HOLD/DROPPED 在 signal 中；账户观察发生在决策/结算时，不是连续 MTM 行情 |
| 最终胜负/payout、结算来源和时间 | `settlement_events.jsonl` | 要等有效 final resolution；到期、停机或断连不等于已经结算 |
| 源码版本及参数身份 | manifest 的 `source_commit`、`config_sha256`；run link 的 Operator 配置哈希 | 完整可读 TOML/策略参数没有自动复制进 run；仅靠 hash 无法还原配置，应另保管部署配置与 sealed wheel |
| 每条原始 WS/REST 消息、历史 feed health / reconnect 详情 | **未完整归档** | 部分原始响应只有 hash；status 只保留最新投影，日志/soak 采样聚合不是原始 tape |
| 精确关窗时的 Spot、TWAP、UP/DOWN 盘口 | **没有单独的完整收盘行情快照** | 最后决策可能早于关窗；resolution payout 不能代替收盘现货/盘口 |

因此：若“所有基础信息”指可以逐 tick 重放一个 round，并在历史页面查看
完整价格曲线，当前答案是 **尚未实现**。已有文件足以支持已有决策和交易
账本的审计，但不能声称能从中重建全部行情和未记录的决策。

## 查询与验证

- `OperatorReadRepository.recent_runs()` 按新到旧遍历前驱链；
  `recent_decisions()`、`recent_fills()`、`settlements()` 支持跨 run 有界查询。
  分页可使用 `before_run_id`，不应把默认页面行数当成所有历史记录。
- Dashboard 的 `market_data` 显示当前窗口投影，不是历史价格数据库。
- 三窗口及重启回归核对：旧 run 的开盘价、token、决策报价/Spot/TWAP、
  alpha 时间、结算引用可回查；新窗口与重启不能改写旧 run 文件的字节。
- 开盘 TWAP reference 的持久化和无重抓恢复另有回归覆盖。
- 若在到期前停机，run 会保留已写入的交易与未平仓位；在同一身份下恢复、
  取得权威最终 resolution 后才能补结算。不能填入猜测 payout 或把旧 FAIL
  报告改为 PASS。

完整行情归档需要另行明确采样/逐事件粒度、磁盘预算与保留期、压缩/校验、
中断与缺口标记以及历史查询接口。本次检查不隐式增加无限行情记录器。

补充：soak 报告现保留最近 180 次就绪检查和停机前的失败快照；组件状态
保留固定原因码计数及最近 32 条诊断（每次报告采样最多复制其中 8 条）。
它能区分盘口重建、心跳超时、OFI 过期和定价样本时序等问题，但仍是
有界诊断，不是上表中的完整历史 feed health 或原始行情归档。

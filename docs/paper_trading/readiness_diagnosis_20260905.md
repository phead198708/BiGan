# Live readiness failure — 2026-09-05

本轮已复现 `LIVE_INPUTS_UNAVAILABLE_DEADLINE`，并保留了关闭服务前的失败现场。
目标为 5 分钟**诊断运行**，实际测量 204.201 秒后 FAIL；不是 5 分钟完成，
也不是 30 分钟验收通过。所有交易均为模拟执行，没有真实资金。

## 证据身份

- 已验证的 sealed wheel 源码：`d01e5f52999e1179d3becef97449e4c40242817f`。
- Wheel SHA-256：`485325e823d61b2205c496979751507ae786e50fcdc4f76f4dbca198cfa55090`。
- 配置哈希：`1deb884294b69be4cf0088ee6802a1713a349f8d508a336dcff292aadbe26542`。
- 报告目录（相对于仓库）：
  `artifacts/paper_soak/readiness-5m-d01e5f52-20260905_003703`。
- 数据目录：`data/paper_readiness_d01e5f52_20260905_003703`。
- `load_completed_report()` 已验证 completion marker 和 artifact 哈希。
  旧失败报告、旧账户未被覆盖。
- 未修改交易门槛、行情时间戳或自动校时设置；运行前 NTP 估计偏差约 18 ms。

## 最后一次连续未就绪的时间线

以下为北京时间；就绪状态按约 2 秒间隔观察，不是逐毫秒全量轨迹。

| 时间 | 证据 |
| --- | --- |
| 00:41:04.0 | 最后一次 RUNNING 采样，无阻塞门槛 |
| 00:41:06.0 | OFI 过期，开始连续未就绪计时 |
| 00:41:10.0 | OFI、pricing 输入均过期 |
| 00:41:10.163 | CLOB 同步器记录 `DEPTH_INVALID_PRICE`，触发清簿/重新订阅 |
| 00:41:16.1 | Spot、Oracle、OFI 已恢复；Polymarket 仍未同步 |
| 00:41:25.175 | transport 记录 `WS_REBOOTSTRAP_REQUIRED` |
| 00:41:27.744 | 新连接收到的盘口触发 `DEPTH_EMPTY_SIDE` |
| 00:41:32.000 | 再次重连仍触发 `DEPTH_EMPTY_SIDE` |
| 00:41:36.351 | 连续未就绪 30,347 ms，超过 30,000 ms 门槛；停止 |

失败快照中：Binance 与 alpha 均 fresh，年龄 339 ms；Chainlink 与 pricing
均 fresh，年龄 2,747 ms；Session 正常。只有 Polymarket 未同步、不 fresh。
因此，本轮是多项门槛先后阻塞、始终没有同时恢复，最后卡在 CLOB 重建。
不能把全部 30 秒都说成单独的 Polymarket 故障。

`DEPTH_INVALID_PRICE` 是本地校验分类，不证明交易所发出了错误数据；空侧
盘口也可能是正常的无流动性状态。当前实现要求双方有正流动性，遇到空侧
会清簿并重连。未保留原始报文，因此不能断言具体价格值为 0、越界或其他
形式。应继续核对空侧/缺失 best quote 的协议语义及重建策略，不能伪造深度。
本轮诊断没有 `EVENT_FROM_FUTURE` 或 `WS_HEARTBEAT_TIMEOUT` 记录。

## 独立确认：定价 provider 没有按决策时点选取历史样本

失败快照累计记录 `PRICING_FUTURE_SPOT=9485`、
`PRICING_FUTURE_ORACLE=2134`，同一次决策可以同时增加两项，不能相加当作
唯一决策数。这些失败发生在 provider 的时间校验，而不是 Oracle 断线。

最小内存复现使用同一安装包：缓存 Spot/Oracle 在 1000、2000、3000、4000 ms
的样本，且已完成波动率预热。4100 ms 时 provider health 为 ready/fresh；
处理 3500 ms 的决策时，provider 只检查最后的 4000 ms 样本并返回 `None`。
虽然缓存中存在有效的 3000 ms 历史样本，却没有被选用。

建议后续修复：按 `sample.timestamp <= decision.timestamp` 分别选择 Spot、
Oracle；波动率样本也必须遵循同一 as-of 截止，不能只修价格选择而引入未来
收益。这个问题导致丢决策，但**不是最终失败快照中的剩余阻塞项**。

## 本轮交付范围

完成有界、脱敏诊断与复现；未改变 CLOB 校验/恢复策略或 provider 取样行为。
后续先处理 CLOB 空侧/缺失报价状态及 as-of 定价，再在原门槛下复测。
详细报告提供有限采样和计数，不等同于原始 feed 全量归档。

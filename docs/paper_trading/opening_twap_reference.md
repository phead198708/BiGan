# Opening reference and published-TWAP pricing

PAPER / SIMULATED — NO REAL FUNDS.

## Market identity comes first

Live discovery reads `cryptoMarketConfig` from Gamma. Asset, duration, enabled
TWAP flag and 30/60-second lookback must agree with the exact market slug and
Chainlink resolution URL. Missing or contradictory TWAP identity fails closed.
Do not infer the oracle from message frequency, market title, or a duration-only
default. Legacy/mock markets without TWAP resolution retain the prior model.

## Opening reference

The read-only Polymarket website endpoint is allowlisted explicitly:
`https://polymarket.com/api/crypto/crypto-price`. It is the site's current JSON
price service, **not a versioned/documented Gamma API or a signed oracle report**.
The integration is intentionally schema-strict; changes or unavailability cause
discovery to retry without creating a Session or substituting another price.

The request binds `symbol`, `eventStartTime`, `endDate`, `variant`,
`twapEnabled=true`, and `twapLookbackSeconds`. Only `openPrice` supplies the
strike. `closePrice` never becomes an opening price or a settlement payout.
`incomplete=true` is valid while the closing price is unavailable. Null/invalid
opening prices, pre-open requests, responses finishing at/after expiry,
future/invalid response timestamps, and conflicting Gamma prices are rejected.

An immutable `OpeningReferenceProof` records the endpoint, market/condition,
asset, opening/end times, lookback, price, request/receipt/publication times and
canonical response SHA-256. Its source timestamp is the service's publication
time, **not a claim that the price was observed at receipt time**. The opening
time is established by the exact request. The proof is persisted in the account
checkpoint/run link and bound into the Session manifest configuration hash.
Resume recovers the existing proof/strike without refreshing them from a current
quote. A hash provides audit binding, not authenticity against a malicious source.

No wallet, signing, private credentials, paid data subscription, scraping-based
price fallback, or geographical-restriction workaround is used. For stronger
oracle authenticity, a separately authorized, verified Chainlink report adapter
would be needed; this implementation trusts the public Polymarket HTTPS service.

## Real-time input

The [official TWAP documentation](https://docs.polymarket.com/market-data/chainlink-twap)
specifies `crypto_prices_twap_thirty` and `crypto_prices_twap_sixty`. The transport
subscribes to the market's explicit lookback and compact JSON symbol filter,
with an application `PING` at most five seconds apart, including busy streams.
It validates topic, symbol, `window_s`, observation and publication timestamps.
The E18 `full_accuracy_value` is decoded using decimal arithmetic before entering
the existing floating-point strategy interface; display-only `value` is not a
fallback. Observation time drives freshness, not local receipt time.

RTDS provides no TWAP history/snapshot/replay. Reconnect therefore invalidates
rolling inputs and requires warm-up again. Ordinary `crypto_prices_chainlink`
updates and the other lookback/asset cannot enter a TWAP-configured Session.
Binance remains the independent spot/OFI source, and Polymarket CLOB remains the
execution source. All three inputs must pass the existing freshness gates.

## Pricing model and limitations

`reference_model=published_twap` uses the latest published TWAP as the reference
process and its fixed opening TWAP as the strike. Annualized volatility is
estimated from returns of that **same TWAP series**, normalized by actual elapsed
time. Binance spot remains independently recorded in decision events; it is not
renamed to TWAP. The legacy event field `oracle_twap_so_far` holds the latest
published value in this mode, and `twap_weight` must be zero.

The binary probability uses a lognormal diffusion approximation for the published
TWAP over the remaining horizon, plus the existing OFI adjustment. It does not
apply the cumulative-window effective-strike formula, re-average the oracle
series, or claim to reconstruct Chainlink's unpublished sampling/rounding rules.
The lookback, reference model and volatility source are configuration identities.
Gamma's final binary outcome remains the sole settlement authority.

This is a **paper forecasting model, not a calibrated profitability claim**.
Rolling TWAP returns can be serially correlated; empirical volatility and the
diffusion approximation require out-of-sample calibration before reliance. Tail
cutoff, spread/liquidity gates, Kelly caps and all other execution controls remain.

Use a fresh deployment configuration/output directory when validating this model
revision. Do not relabel an older run's configuration/provenance or delete its
checkpoint to force a reset; an existing-account migration must be explicit.

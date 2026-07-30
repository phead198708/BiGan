# BTC-15M-MoE-confirmatory-v1 hardening report

Reconciliation passed for all five candidates. The independent recomputation consumed 1,460 OOF prediction rows and 365 fold audits over 73 markets; PnL, bootstrap intervals, largest-winner removal, chronological halves, probability metrics, cost decomposition, and every gate boolean matched exactly.

The architecture is frozen as `deterministic_regime_router_with_conditional_experts_and_global_fallback`. Candidate id remains `mixture_of_experts`; this is not represented as a pure independent-expert ensemble.

## MoE attribution

| Measure | Value |
|---|---:|
| Markets / accepted | 73 / 72 |
| Total development unit PnL | 4.372000 |
| Native expert PnL | 2.425750 |
| Global fallback PnL | 1.946250 |
| Fallback count / share | 15 / 20.5479% |
| Fallback share Q1 / Q2 / Q3 / Q4 | 52.6316% / 11.1111% / 11.1111% / 5.5556% |
| Native expert largest winner | 0.834750 |
| Global fallback largest winner | 0.734750 |

By requested route, PnL was high-vol `+3.288500`, bullish `+1.110500`, bearish `-2.050750`, and low-vol `+2.023750`. Trade-volume input was missing in 60/73 markets; depth and spread were complete in all 73. These are development-only diagnostics and are permanently ineligible for promotion evidence.

## Static artifact and portability

- Bundle: `30d180b028c83146fafd81c8b81269f51fa567b30bc5ab4d3577dd99c256dcf8`
- Artifact graph: `f0f0f09c14b85a888dd1aa2c1cfe4f88acfe9332bc6513f83e99404a089768dd`
- Model manifest: `fddd8c1993c0abdcbd45dc439ae7053ca4173f60f938097bf0e8083f0104b930`
- Global fallback: `662989a6c2f479e04ebc925339fd74e969c85821ad36752d631c8039c18e43d2`
- Expert hashes: high-vol `40f9ee88…`, bullish `4c4be84c…`, bearish `fe302bf9…`; low-vol is an explicit unavailable stub (`0958b4b9…`) because frozen support is 18, below 20.

All graph paths are repository-relative. A fresh-clone copy resolved the graph, verified all hashes, loaded the router, three available experts, low-vol unavailable stub and fallback, then reproduced both deterministic synthetic predictions. Graph and component SHA mismatches fail closed.

## Validation and residual limits

The focused hardening suite passed 33/33. The full `tests/v8` run collected 1,597 tests: 1,574 passed and 23 pre-existing unrelated tests failed because of missing legacy v6/O/HTS artifacts, pre-existing future-evaluation fixture gates, or legacy public-provider mocks that do not accept the Data API fallback. Changed-file Ruff checks passed; full-repository Ruff still reports four unrelated pre-existing findings.

The original parent report's recorded source commit remains unresolvable and is not claimed reachable. The development MoE LCB remains negative, no parent candidate was selected, and no development result is promotion evidence.

Fresh collection remains blocked: `fresh_collection_authorized=false`, `fresh_collection_started=false`, and `fresh_outcomes_opened=false`. No new outcome was opened.

BTC-15M-MoE-confirmatory-v1 hardening protocol is internally consistent.

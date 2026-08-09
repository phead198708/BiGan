# BTC-15M-MoE-confirmatory-v2 terminal failure diagnostic

## Decision

`BTC-15M-MoE-confirmatory-v2` remains terminal failed. It must not be rerun, extended, threshold-tuned, or rescued by changing its population.

A materially new development lineage is conditionally justified:

`BTC-15M-cost-aware-market-residual-v1`

This report does not authorize training or collection. The new hypothesis must predict market-relative residual or after-cost action advantage, rather than applying deterministic regime experts to raw outcome probability.

## Why the confirmatory gates failed

The failure was not caused by collection, settlement, paired-ask coverage, artifact integrity, or population reconciliation. All 800 frozen markets settled, paired executable ask coverage was 100%, and every non-statistical protocol check passed.

The economic effect did not generalize:

| Metric | Development | Confirmatory | Effect retained |
| --- | ---: | ---: | ---: |
| Candidate mean unit PnL | +0.059890 | +0.004429 | 7.40% |
| Paired-delta mean unit PnL | +0.052603 | +0.003052 | 5.80% |

The candidate produced total PnL of `+3.54325` and paired delta of `+2.44125`, but the 97.5% bootstrap LCBs were `-0.026126` and `-0.021841`. Candidate second-half PnL was `-0.97800`; first-half paired delta was `-4.16050`.

The preregistered power analysis expected approximately 95.7% and 80.3% probability of the absolute and paired LCBs crossing zero at 800 markets under the development effect. The confirmatory result therefore invalidated the effect assumption; it is not a reason to collect more markets under the same terminal lineage.

## Economic-edge diagnosis

The candidate still predicts direction, but it does not add stable information relative to executable market price:

| Probability metric | Candidate | Matched global baseline |
| --- | ---: | ---: |
| AUC | 0.819751 | 0.831196 |
| Brier score | 0.169797 | 0.165959 |
| Log loss | 0.518554 | 0.507881 |

For 795 accepted candidate markets, mean predicted net score was `+0.104788`, while realized mean unit PnL was only `+0.004457`. The optimism gap was `0.100331`, and score-to-realized-PnL correlation was only `0.0676`.

Cost/signal rose from `0.1608` in development to `0.7389` in confirmatory data. The primary change was collapse of realized gross edge, not a material increase in the frozen cost model.

## Route and time instability

| Requested route | Markets | Candidate PnL | Paired delta |
| --- | ---: | ---: | ---: |
| high_vol | 151 | -2.25475 | -2.06675 |
| bullish | 206 | +2.13850 | -0.76275 |
| bearish | 271 | -0.30800 | +3.77075 |
| low_vol | 172 | +3.96750 | +1.50000 |

The high-vol expert reversed from a development mean of `+0.12648` to a confirmatory mean of `-0.01493`. The low-vol expert was unavailable; all `+1.5` reported fallback delta came from three markets where an expert first rejected and the later decision routed to fallback. It is not evidence that the byte-identical fallback artifact directly outperformed itself.

Candidate quartile PnLs were:

`+4.72050 / -0.19925 / -6.19725 / +5.21925`

Paired-delta quartile PnLs were:

`-3.77575 / -0.38475 / +3.99825 / +2.60350`

Absolute and relative edge were therefore not stable in the same chronological segments.

## Side and population shift

Candidate DOWN PnL was `+7.61575`; UP PnL was `-4.07250`. Outcomes were only modestly DOWN-heavy—423 DOWN and 377 UP—so the concentration cannot be attributed solely to a broad down market.

Development-to-confirmatory shifts included:

- high-vol route: 31.9% to 18.9%
- low-vol/fallback route: 8.0% to 21.5%
- sideways regime: 9.7% to 23.1%
- feature-complete rows: 20.4% to 13.6%

These shifts contributed to the failure, but route-stratified reversals show that population mixture is not the complete explanation.

## Requirements for any new lineage

1. Register the opened 800-market window and prior 113-market corpus as development-only forever.
2. Use rolling-origin, market-grouped, side-symmetric training with native missingness.
3. Predict market-relative residual or `settlement payout - executable ask - frozen costs` action advantage.
4. Start with a pooled residual model; use regime as interactions until independent expert support is adequate.
5. Preregister a finite candidate budget. Do not use threshold grids, route filtering, side filtering, missingness filtering, or outlier deletion.
6. Require positive absolute and paired market-bootstrap LCBs, chronological stability, largest-winner robustness, and stable score-to-PnL ordering before freezing a candidate.
7. Use only a strictly later unopened window for future promotion evidence, with sample size determined from conservative rolling-origin OOF effects.

## Safety and evidence boundary

This is a post-terminal diagnostic. It modifies no candidate, baseline, router, threshold, statistical gate, population, or confirmatory evidence. It authorizes neither training nor collection.

Paper, live trading, wallet signing, Polymarket writes, execution handoff, promotion, and capital at risk all remain disabled.

# ADR-0003 · Cross-Batch Trade ID De-Duplication

- **Status**: Accepted
- **Date**: 2026-05-13
- **Issue**: [#27](https://github.com/phead198708/BiGan/issues/27)
- **Milestone**: `mvp-v1`
- **Owners**: data-ingestion

---

## Decision

Use **方案 C: read + compare target partition before write** for
`raw_trades.trade_id` cross-batch de-duplication.

For each clean trade row, ETL derives its target partition
`raw_trades/source=<source>/dt=<UTC date from ts>`, lazily reads existing
`trade_id` values from that partition's Parquet files, and skips any row whose
`trade_id` is already present. Accepted rows are added to the in-memory
partition cache so repeated IDs later in the same ETL run are also blocked
before they reach the warehouse.

Skipped rows are not quarantined because they are valid data already present in
the warehouse. `EtlReport.cross_batch_duplicates_skipped` records the count.

## Rationale

- The warehouse remains append-only; no compaction or historical file rewrite is
  needed.
- The lookup is partition-local, so normal backfills only read the small set of
  affected dates and sources.
- `trade_id` is already stable and source-scoped:
  `{source}-{source_symbol}-{ts}-{price}-{size}-{side}`.
- The approach composes with issue #4's in-batch `duplicate_trade_id` rule:
  validator still catches duplicates inside one ETL run, while the partition
  cache catches duplicates already written by previous runs.

## Rejected Options

### 方案 A: Persisted Bloom filter / set per partition

Rejected for now because it introduces a second metadata artifact that must stay
transactionally aligned with append-only Parquet writes. A stale index is worse
than no index because it can silently drop valid trades or admit duplicates.

### 方案 B: DuckDB anti-join compaction rewrite

Rejected for `mvp-v1` because it changes the storage contract from append-only to
rewrite/compaction. It may be a useful future maintenance command, but it should
not be required for routine ETL correctness.

## Operational Notes

`bigan-ingest etl-batch` prints `cross_batch_duplicates_skipped`. The
`quarantine-report` command also reports `raw_trade_duplicate_rows`, computed as
`COUNT(*) - COUNT(DISTINCT trade_id)` over the current warehouse.

Expected invariant after ETL:

```sql
SELECT COUNT(*) = COUNT(DISTINCT trade_id) FROM raw_trades;
```

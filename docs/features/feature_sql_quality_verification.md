# Feature SQL Quality Verification

Use this after `features_15m_v1` has been generated from a canonical warehouse.
The checks are intentionally SQL-only so they can run in DuckDB against local
Parquet evidence, CI artifacts, or a copied production warehouse.

## CLI Command

```bash
.venv/bin/bigan-ingest feature-quality-report
```

The command prints JSON and exits non-zero if any quality check fails.

## SQL Command

```bash
.venv/bin/python - <<'PY'
from pathlib import Path

from bigan.canonical.query import open_warehouse

warehouse = Path("data/warehouse")

checks = {
    "row_count": """
        select case when count(*) = 0 then 1 else 0 end as failures
        from features_15m_v1
    """,
    "duplicate_symbol_feature_ts": """
        select count(*) as failures
        from (
            select source, source_symbol, feature_ts, count(*) as c
            from features_15m_v1
            group by source, source_symbol, feature_ts
            having c > 1
        )
    """,
    "minute_alignment": """
        select count(*) as failures
        from features_15m_v1
        where feature_ts % 60000 != 0 or ts != feature_ts or message_ts != feature_ts
    """,
    "required_identity_not_null": """
        select count(*) as failures
        from features_15m_v1
        where source is null
           or source_symbol is null
           or symbol is null
           or feature_version is null
    """,
    "quality_score_bounds": """
        select count(*) as failures
        from features_15m_v1
        where completeness_score < 0
           or completeness_score > 1
           or completeness_score is null
    """,
    "gap_flag_consistency": """
        select count(*) as failures
        from features_15m_v1
        where data_gap_flag is null
           or quality_filter_pass is null
           or (data_gap_flag and quality_filter_pass)
    """,
    "training_filter_has_rows": """
        select case when count(*) = 0 then 1 else 0 end as failures
        from features_15m_v1
        where quality_filter_pass
    """,
}

with open_warehouse(warehouse) as con:
    failed = []
    for name, sql in checks.items():
        failures = con.execute(sql).fetchone()[0]
        print(f"{name}: {failures}")
        if failures:
            failed.append(name)

if failed:
    raise SystemExit(f"feature SQL quality failed: {failed}")
PY
```

## Expected Result

Every check should print `0` failures. `training_filter_has_rows` is inverted:
it prints `0` when there is at least one row that passes the training filter.

The most useful ad-hoc inspection query is:

```sql
select
  symbol,
  count(*) as rows,
  min(completeness_score) as min_score,
  avg(completeness_score) as avg_score,
  sum(case when data_gap_flag then 1 else 0 end) as gap_rows,
  sum(case when quality_filter_pass then 1 else 0 end) as trainable_rows
from features_15m_v1
group by symbol
order by rows desc;
```

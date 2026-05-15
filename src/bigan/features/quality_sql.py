"""DuckDB quality checks for generated feature tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb

from bigan.canonical.query import open_warehouse


@dataclass(frozen=True, slots=True)
class FeatureQualitySqlCheck:
    """One SQL quality check result."""

    name: str
    failures: int

    @property
    def passed(self) -> bool:
        return self.failures == 0

    def to_dict(self) -> dict[str, bool | int | str]:
        out = asdict(self)
        out["passed"] = self.passed
        return out


@dataclass(frozen=True, slots=True)
class FeatureQualitySqlReport:
    """Aggregated SQL quality report for ``features_15m_v1``."""

    checks: tuple[FeatureQualitySqlCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }


FEATURE_QUALITY_SQL_CHECKS: tuple[tuple[str, str], ...] = (
    (
        "row_count",
        """
        select case when count(*) = 0 then 1 else 0 end as failures
        from features_15m_v1
        """,
    ),
    (
        "duplicate_symbol_feature_ts",
        """
        select count(*) as failures
        from (
            select source, source_symbol, feature_ts, count(*) as c
            from features_15m_v1
            group by source, source_symbol, feature_ts
            having c > 1
        )
        """,
    ),
    (
        "minute_alignment",
        """
        select count(*) as failures
        from features_15m_v1
        where feature_ts % 60000 != 0 or ts != feature_ts or message_ts != feature_ts
        """,
    ),
    (
        "required_identity_not_null",
        """
        select count(*) as failures
        from features_15m_v1
        where source is null
           or source_symbol is null
           or symbol is null
           or feature_version is null
        """,
    ),
    (
        "quality_score_bounds",
        """
        select count(*) as failures
        from features_15m_v1
        where completeness_score < 0
           or completeness_score > 1
           or completeness_score is null
        """,
    ),
    (
        "gap_flag_consistency",
        """
        select count(*) as failures
        from features_15m_v1
        where data_gap_flag is null
           or quality_filter_pass is null
           or (data_gap_flag and quality_filter_pass)
        """,
    ),
    (
        "training_filter_has_rows",
        """
        select case when count(*) = 0 then 1 else 0 end as failures
        from features_15m_v1
        where quality_filter_pass
        """,
    ),
)


def run_feature_quality_sql_checks(warehouse_dir: Path | str) -> FeatureQualitySqlReport:
    """Run SQL acceptance checks against ``features_15m_v1``."""

    checks: list[FeatureQualitySqlCheck] = []
    with open_warehouse(warehouse_dir) as conn:
        for name, sql in FEATURE_QUALITY_SQL_CHECKS:
            checks.append(FeatureQualitySqlCheck(name=name, failures=_failure_count(conn, sql)))
    return FeatureQualitySqlReport(checks=tuple(checks))


def _failure_count(conn: duckdb.DuckDBPyConnection, sql: str) -> int:
    try:
        row = conn.execute(sql).fetchone()
    except (duckdb.CatalogException, duckdb.IOException):
        return 1
    if row is None:
        return 1
    return int(row[0] or 0)

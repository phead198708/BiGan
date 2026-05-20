"""Training dataset assembly for model pipelines (issue #15)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from bigan.canonical.query import open_warehouse
from bigan.canonical.schemas import SCHEMAS

DATASET_VERSION = "bigan-training-15m-v1.0.0"

FEATURE_TABLE = "features_15m_v1"
LABEL_TABLE = "labels_15m_v1"
SPLITS: tuple[str, ...] = ("train", "val", "test")
NON_MODEL_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "ts",
        "message_ts",
        "feature_ts",
        "ingest_ts",
        "source",
        "source_symbol",
        "source_market",
        "canonical_symbol",
        "symbol",
        "feature_version",
        "completeness_score",
        "data_gap_flag",
        "quality_filter_pass",
        "quote_age_ms",
        "depth_age_ms",
        "trade_age_ms",
    }
)


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Deterministic time split fractions for assembled training samples."""

    train_fraction: float = 0.60
    val_fraction: float = 0.20

    def __post_init__(self) -> None:
        if not 0.0 <= self.train_fraction <= 1.0:
            raise ValueError("train_fraction must be in [0, 1]")
        if not 0.0 <= self.val_fraction <= 1.0:
            raise ValueError("val_fraction must be in [0, 1]")
        if self.train_fraction + self.val_fraction > 1.0:
            raise ValueError("train_fraction + val_fraction must be <= 1")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SplitStats:
    """Row and class-balance stats for one dataset split."""

    row_count: int
    positive_count: int
    negative_count: int
    positive_rate: float | None
    start_ts: int | None
    end_ts: int | None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DatasetAssemblyReport:
    """Reproducible manifest for one training dataset assembly run."""

    dataset_version: str
    rows_joined: int
    rows_written: int
    rows_filtered_quality: int
    rows_missing_label: int
    min_completeness_score: float
    split_config: SplitConfig
    splits: dict[str, SplitStats]
    feature_columns: tuple[str, ...]
    feature_versions: tuple[str, ...]
    label_versions: tuple[str, ...]
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "rows_joined": self.rows_joined,
            "rows_written": self.rows_written,
            "rows_filtered_quality": self.rows_filtered_quality,
            "rows_missing_label": self.rows_missing_label,
            "min_completeness_score": self.min_completeness_score,
            "split_config": self.split_config.to_dict(),
            "splits": {
                name: stats.to_dict()
                for name, stats in sorted(self.splits.items())
            },
            "feature_columns": list(self.feature_columns),
            "feature_versions": list(self.feature_versions),
            "label_versions": list(self.label_versions),
            "output_dir": self.output_dir,
        }


def assemble_training_dataset(
    warehouse_dir: Path | str,
    output_dir: Path | str,
    *,
    split_config: SplitConfig | None = None,
    min_completeness_score: float = 0.80,
) -> DatasetAssemblyReport:
    """Join feature and label tables, filter quality, and write train/val/test.

    The join is point-in-time safe by construction: features and labels meet on
    ``(source, source_symbol, feature_ts)`` and the assembler rejects any joined
    label whose ``target_ts`` is not strictly after ``feature_ts``.
    """

    if not 0.0 <= min_completeness_score <= 1.0:
        raise ValueError("min_completeness_score must be in [0, 1]")

    config = split_config or SplitConfig()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    feature_columns = _feature_columns()

    with open_warehouse(warehouse_dir) as conn:
        _require_source_tables(conn)
        rows_joined = _count_joined(conn)
        rows_missing_label = _count_missing_labels(conn)
        leakage_count = _count_leakage_rows(conn)
        if leakage_count:
            raise ValueError(
                "future information leakage detected: "
                f"{leakage_count} joined label rows are not after feature_ts"
            )
        table = _fetch_training_samples(
            conn,
            feature_columns=feature_columns,
            min_completeness_score=min_completeness_score,
        )

    rows_written = table.num_rows
    rows_filtered_quality = rows_joined - rows_written
    split_tables = _split_table(table, config)
    for name, split_table in split_tables.items():
        pq.write_table(split_table, target / f"{name}.parquet")

    report = DatasetAssemblyReport(
        dataset_version=DATASET_VERSION,
        rows_joined=rows_joined,
        rows_written=rows_written,
        rows_filtered_quality=rows_filtered_quality,
        rows_missing_label=rows_missing_label,
        min_completeness_score=min_completeness_score,
        split_config=config,
        splits={
            name: _split_stats(split_table)
            for name, split_table in split_tables.items()
        },
        feature_columns=feature_columns,
        feature_versions=_unique_strings(table, "feature_version"),
        label_versions=_unique_strings(table, "label_version"),
        output_dir=str(target),
    )
    (target / "manifest.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _feature_columns() -> tuple[str, ...]:
    schema = SCHEMAS[FEATURE_TABLE]
    return tuple(
        field.name
        for field in schema
        if field.name not in NON_MODEL_FEATURE_COLUMNS
    )


def _require_source_tables(conn: duckdb.DuckDBPyConnection) -> None:
    missing = []
    for table_name in (FEATURE_TABLE, LABEL_TABLE):
        try:
            conn.execute(f"select 1 from {table_name} limit 1")
        except (duckdb.CatalogException, duckdb.IOException):
            missing.append(table_name)
    if missing:
        raise ValueError(f"warehouse is missing required tables: {', '.join(missing)}")


def _count_joined(conn: duckdb.DuckDBPyConnection) -> int:
    return int(
        conn.execute(
            """
            select count(*)
            from features_15m_v1 f
            inner join labels_15m_v1 l
              on f.source = l.source
             and f.source_symbol = l.source_symbol
             and f.feature_ts = l.feature_ts
            """
        ).fetchone()[0]
    )


def _count_missing_labels(conn: duckdb.DuckDBPyConnection) -> int:
    return int(
        conn.execute(
            """
            select count(*)
            from features_15m_v1 f
            left join labels_15m_v1 l
              on f.source = l.source
             and f.source_symbol = l.source_symbol
             and f.feature_ts = l.feature_ts
            where l.feature_ts is null
            """
        ).fetchone()[0]
    )


def _count_leakage_rows(conn: duckdb.DuckDBPyConnection) -> int:
    return int(
        conn.execute(
            """
            select count(*)
            from features_15m_v1 f
            inner join labels_15m_v1 l
              on f.source = l.source
             and f.source_symbol = l.source_symbol
             and f.feature_ts = l.feature_ts
            where l.target_ts <= f.feature_ts
               or l.round_start_ts > f.feature_ts
            """
        ).fetchone()[0]
    )


def _fetch_training_samples(
    conn: duckdb.DuckDBPyConnection,
    *,
    feature_columns: tuple[str, ...],
    min_completeness_score: float,
) -> pa.Table:
    feature_sql = ",\n                   ".join(f"f.{_quote_identifier(name)}" for name in feature_columns)
    query = f"""
        select
            f.source,
            f.source_symbol,
            f.source_market,
            f.canonical_symbol,
            f.symbol,
            f.feature_ts,
            f.feature_version,
            l.label_version,
            l.target_ts,
            l.round_start_ts,
            l.round_end_ts,
            l.start_price,
            l.target_price,
            l.label_up_15m,
            f.completeness_score,
            f.data_gap_flag,
            f.quality_filter_pass,
            {feature_sql}
        from features_15m_v1 f
        inner join labels_15m_v1 l
          on f.source = l.source
         and f.source_symbol = l.source_symbol
         and f.feature_ts = l.feature_ts
        where not f.data_gap_flag
          and f.quality_filter_pass
          and f.completeness_score >= ?
        order by f.feature_ts, f.source, f.source_symbol
    """
    return conn.execute(query, [min_completeness_score]).to_arrow_table()


def _split_table(table: pa.Table, config: SplitConfig) -> dict[str, pa.Table]:
    total = table.num_rows
    train_count = int(total * config.train_fraction)
    val_count = int(total * config.val_fraction)
    test_count = total - train_count - val_count
    return {
        "train": table.slice(0, train_count),
        "val": table.slice(train_count, val_count),
        "test": table.slice(train_count + val_count, test_count),
    }


def _split_stats(table: pa.Table) -> SplitStats:
    labels = table.column("label_up_15m").to_pylist() if table.num_rows else []
    positive_count = sum(1 for value in labels if bool(value))
    row_count = table.num_rows
    negative_count = row_count - positive_count
    feature_ts = table.column("feature_ts").to_pylist() if row_count else []
    return SplitStats(
        row_count=row_count,
        positive_count=positive_count,
        negative_count=negative_count,
        positive_rate=None if row_count == 0 else positive_count / row_count,
        start_ts=None if not feature_ts else int(feature_ts[0]),
        end_ts=None if not feature_ts else int(feature_ts[-1]),
    )


def _unique_strings(table: pa.Table, column: str) -> tuple[str, ...]:
    if table.num_rows == 0:
        return ()
    return tuple(sorted({str(value) for value in table.column(column).to_pylist() if value is not None}))


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

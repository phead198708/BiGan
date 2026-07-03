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
from bigan.modeling.families import market_family_from_symbol

DATASET_VERSION = "bigan-training-15m-profitability-v1.0.0"

FEATURE_TABLE = "features_15m_v1"
LABEL_TABLE = "labels_15m_v1"
SPLITS: tuple[str, ...] = ("train", "val", "test")
OUTCOME_SIDES: frozenset[str] = frozenset({"UP", "DOWN", "ANY"})
V6_OPTIONAL_LABEL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("settlement_margin", "DOUBLE"),
    ("settlement_abs_margin", "DOUBLE"),
    ("settlement_neutral_margin", "DOUBLE"),
    ("label_settlement_3way", "VARCHAR"),
    ("max_exit_gain_up", "DOUBLE"),
    ("max_exit_gain_down", "DOUBLE"),
    ("max_exit_return_per_usdc_up", "DOUBLE"),
    ("max_exit_return_per_usdc_down", "DOUBLE"),
    ("time_to_best_exit_up", "DOUBLE"),
    ("time_to_best_exit_down", "DOUBLE"),
    ("best_exit_price_up", "DOUBLE"),
    ("best_exit_price_down", "DOUBLE"),
    ("label_volatility_up", "BOOLEAN"),
    ("label_volatility_down", "BOOLEAN"),
    ("volatility_path_validity_up", "VARCHAR"),
    ("volatility_path_validity_down", "VARCHAR"),
)
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
    family_splits: dict[str, dict[str, SplitStats]]
    v6_label_diagnostics: dict[str, Any]
    output_dir: str
    outcome_side: str

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
            "outcome_side": self.outcome_side,
            "family_splits": {
                family: {
                    name: stats.to_dict()
                    for name, stats in sorted(split_stats.items())
                }
                for family, split_stats in sorted(self.family_splits.items())
            },
            "v6_label_diagnostics": self.v6_label_diagnostics,
            "output_dir": self.output_dir,
        }


def assemble_training_dataset(
    warehouse_dir: Path | str,
    output_dir: Path | str,
    *,
    split_config: SplitConfig | None = None,
    min_completeness_score: float = 0.80,
    outcome_side: str = "UP",
) -> DatasetAssemblyReport:
    """Join feature and label tables, filter quality, and write train/val/test.

    The join is point-in-time safe by construction: features and labels meet on
    ``(source, source_symbol, feature_ts)`` and the assembler rejects any joined
    label whose settlement ``target_ts`` is not strictly after ``feature_ts``.
    """

    if not 0.0 <= min_completeness_score <= 1.0:
        raise ValueError("min_completeness_score must be in [0, 1]")

    config = split_config or SplitConfig()
    normalised_outcome_side = _normalise_outcome_side(outcome_side)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    feature_columns = _feature_columns()

    with open_warehouse(warehouse_dir) as conn:
        _require_source_tables(conn)
        outcome_filter_sql = _outcome_filter_sql(normalised_outcome_side)
        rows_joined = _count_joined(conn, outcome_filter_sql=outcome_filter_sql)
        rows_missing_label = _count_missing_labels(conn, outcome_filter_sql=outcome_filter_sql)
        leakage_count = _count_leakage_rows(conn, outcome_filter_sql=outcome_filter_sql)
        if leakage_count:
            raise ValueError(
                "future information leakage detected: "
                f"{leakage_count} joined label rows are not after feature_ts"
            )
        table = _fetch_training_samples(
            conn,
            feature_columns=feature_columns,
            min_completeness_score=min_completeness_score,
            outcome_filter_sql=outcome_filter_sql,
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
        family_splits=_family_split_stats(split_tables),
        v6_label_diagnostics=_v6_label_diagnostics(split_tables),
        output_dir=str(target),
        outcome_side=normalised_outcome_side,
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


def _normalise_outcome_side(value: str) -> str:
    text = str(value).strip().upper()
    if text not in OUTCOME_SIDES:
        raise ValueError("outcome_side must be UP, DOWN, or ANY")
    return text


def _outcome_filter_sql(outcome_side: str) -> str:
    if outcome_side == "ANY":
        return "true"
    return f"""
(
  upper(coalesce(f.canonical_symbol, f.symbol, '')) like '%:{outcome_side}'
  or upper(coalesce(f.canonical_symbol, f.symbol, '')) like '%-{outcome_side}-15M'
)
"""


def _count_joined(conn: duckdb.DuckDBPyConnection, *, outcome_filter_sql: str) -> int:
    return int(
        conn.execute(
            f"""
            select count(*)
            from features_15m_v1 f
            inner join labels_15m_v1 l
              on f.source = l.source
             and f.source_symbol = l.source_symbol
             and f.feature_ts = l.feature_ts
            where {outcome_filter_sql}
            """
        ).fetchone()[0]
    )


def _count_missing_labels(conn: duckdb.DuckDBPyConnection, *, outcome_filter_sql: str) -> int:
    return int(
        conn.execute(
            f"""
            select count(*)
            from features_15m_v1 f
            left join labels_15m_v1 l
              on f.source = l.source
             and f.source_symbol = l.source_symbol
             and f.feature_ts = l.feature_ts
            where {outcome_filter_sql}
              and l.feature_ts is null
            """
        ).fetchone()[0]
    )


def _count_leakage_rows(conn: duckdb.DuckDBPyConnection, *, outcome_filter_sql: str) -> int:
    return int(
        conn.execute(
            f"""
            select count(*)
            from features_15m_v1 f
            inner join labels_15m_v1 l
              on f.source = l.source
             and f.source_symbol = l.source_symbol
             and f.feature_ts = l.feature_ts
            where {outcome_filter_sql}
              and l.target_ts <= f.feature_ts
            """
        ).fetchone()[0]
    )


def _fetch_training_samples(
    conn: duckdb.DuckDBPyConnection,
    *,
    feature_columns: tuple[str, ...],
    min_completeness_score: float,
    outcome_filter_sql: str,
) -> pa.Table:
    feature_sql = ",\n                   ".join(f"f.{_quote_identifier(name)}" for name in feature_columns)
    label_columns = _table_columns(conn, LABEL_TABLE)
    v6_label_sql = ",\n            ".join(
        _optional_label_select_sql(label_columns, name, sql_type)
        for name, sql_type in V6_OPTIONAL_LABEL_COLUMNS
    )
    query = f"""
        select
            concat(
                coalesce(l.round_slug, f.source_market, f.source_symbol),
                ':',
                cast(f.feature_ts as varchar)
            ) as event_id,
            coalesce(f.source_market, l.source_market) as market_id,
            regexp_replace(
                split_part(coalesce(f.canonical_symbol, f.symbol), ':', 1),
                '-(UP|DOWN)-',
                '-'
            ) as family,
            split_part(
                regexp_replace(
                    split_part(coalesce(f.canonical_symbol, f.symbol), ':', 1),
                    '-(UP|DOWN)-',
                    '-'
                ),
                '-',
                2
            ) as horizon,
            f.feature_ts as decision_ts,
            f.source,
            f.source_symbol,
            f.source_market,
            f.canonical_symbol,
            f.symbol,
            f.feature_ts,
            f.feature_version,
            l.label_version,
            l.label_kind,
            l.target_ts,
            l.round_start_ts,
            l.round_end_ts,
            l.start_price,
            l.target_price,
            l.direction_up_15m,
            l.entry_ask_price,
            l.settlement_price,
            l.entry_fee,
            l.entry_cost,
            l.realized_return,
            l.fee_bps,
            {v6_label_sql},
            l.label_profit_up_15m,
            l.label_profit_down_15m,
            l.label_up_15m,
            l.label_down_15m,
            f.completeness_score,
            f.data_gap_flag,
            f.quality_filter_pass,
            {feature_sql}
        from features_15m_v1 f
        inner join labels_15m_v1 l
         on f.source = l.source
         and f.source_symbol = l.source_symbol
         and f.feature_ts = l.feature_ts
        where {outcome_filter_sql}
          and not f.data_gap_flag
          and f.quality_filter_pass
          and f.completeness_score >= ?
        order by f.feature_ts, f.source, f.source_symbol
    """
    return conn.execute(query, [min_completeness_score]).to_arrow_table()


def _table_columns(conn: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    result = conn.execute(f"select * from {_quote_identifier(table_name)} limit 0")
    return {str(column[0]) for column in (result.description or [])}


def _optional_label_select_sql(
    label_columns: set[str],
    name: str,
    sql_type: str,
) -> str:
    quoted = _quote_identifier(name)
    if name in label_columns:
        return f"l.{quoted} as {quoted}"
    return f"cast(null as {sql_type}) as {quoted}"


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
    return _split_stats_from_rows(table.to_pylist())


def _split_stats_from_rows(rows: list[dict[str, Any]]) -> SplitStats:
    labels = [_label_value(row) for row in rows]
    positive_count = sum(1 for value in labels if bool(value))
    row_count = len(rows)
    negative_count = row_count - positive_count
    feature_ts = [int(row["feature_ts"]) for row in rows] if row_count else []
    return SplitStats(
        row_count=row_count,
        positive_count=positive_count,
        negative_count=negative_count,
        positive_rate=None if row_count == 0 else positive_count / row_count,
        start_ts=None if not feature_ts else int(feature_ts[0]),
        end_ts=None if not feature_ts else int(feature_ts[-1]),
    )


def _family_split_stats(split_tables: dict[str, pa.Table]) -> dict[str, dict[str, SplitStats]]:
    rows_by_split = {
        split: table.to_pylist()
        for split, table in split_tables.items()
    }
    families = sorted(
        {
            market_family_from_symbol(row.get("canonical_symbol") or row.get("symbol"))
            for rows in rows_by_split.values()
            for row in rows
        }
    )
    return {
        family: {
            split: _split_stats_from_rows(
                [
                    row
                    for row in rows
                    if market_family_from_symbol(row.get("canonical_symbol") or row.get("symbol"))
                    == family
                ]
            )
            for split, rows in rows_by_split.items()
        }
        for family in families
    }


def _v6_label_diagnostics(split_tables: dict[str, pa.Table]) -> dict[str, Any]:
    rows_by_split = {
        split: table.to_pylist()
        for split, table in split_tables.items()
    }
    families = sorted(
        {
            market_family_from_symbol(row.get("canonical_symbol") or row.get("symbol"))
            for rows in rows_by_split.values()
            for row in rows
        }
    )
    return {
        "settlement_3way_class_balance": {
            split: _class_balance(rows, "label_settlement_3way")
            for split, rows in rows_by_split.items()
        },
        "family_settlement_3way_class_balance": {
            family: {
                split: _class_balance(
                    [
                        row
                        for row in rows
                        if market_family_from_symbol(row.get("canonical_symbol") or row.get("symbol"))
                        == family
                    ],
                    "label_settlement_3way",
                )
                for split, rows in rows_by_split.items()
            }
            for family in families
        },
        "volatility_label_rates": {
            split: {
                "up": _volatility_stats(rows, "up"),
                "down": _volatility_stats(rows, "down"),
            }
            for split, rows in rows_by_split.items()
        },
        "family_volatility_label_rates": {
            family: {
                split: {
                    "up": _volatility_stats(
                        [
                            row
                            for row in rows
                            if market_family_from_symbol(row.get("canonical_symbol") or row.get("symbol"))
                            == family
                        ],
                        "up",
                    ),
                    "down": _volatility_stats(
                        [
                            row
                            for row in rows
                            if market_family_from_symbol(row.get("canonical_symbol") or row.get("symbol"))
                            == family
                        ],
                        "down",
                    ),
                }
                for split, rows in rows_by_split.items()
            }
            for family in families
        },
    }


def _class_balance(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    counts: dict[str, int] = {"UP": 0, "DOWN": 0, "NEUTRAL": 0}
    rows_with_label = 0
    for row in rows:
        value = row.get(column)
        if value is None:
            continue
        rows_with_label += 1
        key = str(value).upper()
        counts[key] = counts.get(key, 0) + 1
    counts["rows"] = len(rows)
    counts["rows_with_label"] = rows_with_label
    return counts


def _volatility_stats(rows: list[dict[str, Any]], side: str) -> dict[str, float | int | None]:
    label_column = f"label_volatility_{side}"
    path_column = f"volatility_path_validity_{side}"
    row_count = len(rows)
    known_labels = [row.get(label_column) for row in rows if row.get(label_column) is not None]
    positive_count = sum(1 for value in known_labels if bool(value))
    valid_paths = sum(1 for row in rows if row.get(path_column) == "valid")
    return {
        "rows": row_count,
        "known_label_count": len(known_labels),
        "valid_path_count": valid_paths,
        "price_path_coverage_rate": _safe_ratio(len(known_labels), row_count),
        "valid_path_rate": _safe_ratio(valid_paths, row_count),
        "positive_count": positive_count,
        "positive_rate": _safe_ratio(positive_count, len(known_labels)),
    }


def _label_value(row: dict[str, Any]) -> bool:
    if str(row.get("label_kind") or "").strip().lower() == "down_token_profitability":
        value = row.get("label_profit_down_15m")
        if value is None:
            value = row.get("label_down_15m")
        return bool(value)
    value = row.get("label_profit_up_15m")
    if value is None:
        value = row.get("label_up_15m")
    return bool(value)


def _unique_strings(table: pa.Table, column: str) -> tuple[str, ...]:
    if table.num_rows == 0:
        return ()
    return tuple(sorted({str(value) for value in table.column(column).to_pylist() if value is not None}))


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _primary_label_column(table: pa.Table) -> str:
    names = set(table.schema.names)
    if "label_profit_up_15m" in names:
        values = table.column("label_profit_up_15m").to_pylist() if table.num_rows else []
        if any(value is not None for value in values):
            return "label_profit_up_15m"
    return "label_up_15m"


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator

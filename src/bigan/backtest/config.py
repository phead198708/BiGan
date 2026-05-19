"""Backtest configuration contract (issue #10)."""

from __future__ import annotations

import json
import secrets
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

BACKTEST_CONFIG_SCHEMA_VERSION = "backtest_config_v1"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BacktestStrategyConfig(_StrictModel):
    """Trading decision thresholds."""

    long_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Minimum predicted UP probability required to enter a long-UP trade.",
    )


class BacktestCostConfig(_StrictModel):
    """Explicit trading costs used by the simulator."""

    fee_bps: float = Field(..., ge=0.0, description="Fee charged per trade, in basis points.")
    slippage_bps: float = Field(
        ...,
        ge=0.0,
        description="Expected execution slippage per trade, in basis points.",
    )


class BacktestExecutionConfig(_StrictModel):
    """Execution timing assumptions."""

    latency_ms: int = Field(
        ...,
        ge=0,
        description="Delay between signal timestamp and simulated order placement.",
    )


class BacktestDatasetConfig(_StrictModel):
    """Feature/label dataset identity."""

    dataset_version: str = Field(..., min_length=1)
    feature_table: str = Field(default="features_15m_v1", min_length=1)
    label_table: str = Field(default="labels_15m_v1", min_length=1)


class BacktestModelConfig(_StrictModel):
    """Model artifact identity."""

    model_version: str = Field(..., min_length=1)
    model_uri: str | None = Field(
        default=None,
        description="Optional local path or registry URI for the model artifact.",
    )


class BacktestOutputConfig(_StrictModel):
    """Output destination and unique run identity."""

    output_dir: str = Field(default="data/backtests", min_length=1)
    run_id: str = Field(default_factory=lambda: generate_run_id())


class BacktestConfig(_StrictModel):
    """Fixed v1 backtest configuration file format."""

    schema_version: str = Field(default=BACKTEST_CONFIG_SCHEMA_VERSION)
    strategy: BacktestStrategyConfig
    costs: BacktestCostConfig
    execution: BacktestExecutionConfig
    dataset: BacktestDatasetConfig
    model: BacktestModelConfig
    output: BacktestOutputConfig = Field(default_factory=BacktestOutputConfig)

    @field_validator("schema_version")
    @classmethod
    def _schema_version_is_supported(cls, value: str) -> str:
        if value != BACKTEST_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported backtest config schema_version: {value!r}; "
                f"expected {BACKTEST_CONFIG_SCHEMA_VERSION!r}"
            )
        return value

    def with_new_run_id(self) -> BacktestConfig:
        """Return a copy with a freshly generated run id."""

        return self.model_copy(
            update={"output": self.output.model_copy(update={"run_id": generate_run_id()})}
        )

    def to_script_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict for shell scripts and batch jobs."""

        return self.model_dump(mode="json")


def generate_run_id(*, now: datetime | None = None) -> str:
    """Generate a compact unique backtest run id."""

    dt = now or datetime.now(tz=UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt = dt.astimezone(UTC)
    return f"bt-{dt.strftime('%Y%m%dT%H%M%S%fZ')}-{secrets.token_hex(4)}"


def load_backtest_config(path: Path | str, *, new_run_id: bool = True) -> BacktestConfig:
    """Load a fixed-format YAML or JSON backtest config.

    ``new_run_id=True`` is the normal backtest-run path: it materializes a fresh
    run id even if the file was copied from an older run. Pass
    ``new_run_id=False`` only when intentionally replaying a known run id.
    """

    data = _load_mapping(Path(path))
    config = BacktestConfig.model_validate(data)
    return config.with_new_run_id() if new_run_id else config


def _load_mapping(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        data = _load_yaml(path)
    else:
        raise ValueError(f"unsupported backtest config file type: {path}")
    if not isinstance(data, Mapping):
        raise ValueError("backtest config must be a mapping")
    return data


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        return _load_yaml_subset(path.read_text(encoding="utf-8"))

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("backtest YAML config must be a mapping")
    return data


def _load_yaml_subset(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by the fixed v1 config.

    This fallback keeps YAML configs usable without adding a runtime PyYAML
    dependency. It supports nested mappings with two-space indentation and
    scalar string/number/bool/null values.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError("YAML indentation must use multiples of two spaces")
        key, sep, value = line.strip().partition(":")
        if not sep or not key:
            raise ValueError(f"unsupported YAML line: {raw_line!r}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_yaml_scalar(value.strip())
    return root


def _parse_yaml_scalar(value: str) -> Any:
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if value[:1] in {'"', "'"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value.strip("'\"")
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value

"""Model promotion rules and reports (issue #19)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PromotionRules:
    """Thresholds for promoting a candidate model over the baseline."""

    min_roc_auc_delta: float = 0.0
    max_brier_delta: float = 0.0
    min_backtest_net_pnl: float = 0.0
    require_calibration_improved: bool = True

    def to_dict(self) -> dict[str, bool | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PromotionCheck:
    """One pass/fail promotion checklist item."""

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, bool | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PromotionReport:
    """Serializable model-promotion decision report."""

    passed: bool
    decision: str
    baseline_model_version: str
    candidate_model_version: str
    dataset_version: str | None
    checks: tuple[PromotionCheck, ...]
    baseline_test_metrics: dict[str, float | int | None]
    candidate_test_metrics: dict[str, float | int | None]
    calibration: dict[str, Any] | None
    backtest: dict[str, Any] | None
    rules: PromotionRules
    artifact_paths: dict[str, str]
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "decision": self.decision,
            "baseline_model_version": self.baseline_model_version,
            "candidate_model_version": self.candidate_model_version,
            "dataset_version": self.dataset_version,
            "checks": [check.to_dict() for check in self.checks],
            "baseline_test_metrics": self.baseline_test_metrics,
            "candidate_test_metrics": self.candidate_test_metrics,
            "calibration": self.calibration,
            "backtest": self.backtest,
            "rules": self.rules.to_dict(),
            "artifact_paths": self.artifact_paths,
            "output_dir": self.output_dir,
        }


def evaluate_model_promotion(
    baseline_dir: Path | str,
    candidate_dir: Path | str,
    calibration_dir: Path | str,
    backtest_summary_path: Path | str,
    output_dir: Path | str,
    *,
    rules: PromotionRules | None = None,
) -> PromotionReport:
    """Compare baseline/candidate metrics and write a promotion decision."""

    active_rules = rules or PromotionRules()
    baseline_root = Path(baseline_dir)
    candidate_root = Path(candidate_dir)
    calibration_root = Path(calibration_dir)
    backtest_path = Path(backtest_summary_path)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    baseline_manifest = _read_json(baseline_root / "manifest.json")
    candidate_manifest = _read_json(candidate_root / "manifest.json")
    baseline_metrics = _read_json(baseline_root / "metrics.json")
    candidate_metrics = _read_json(candidate_root / "metrics.json")
    calibration = _read_optional_json(calibration_root / "calibration_report.json")
    backtest = _summarize_backtest(_read_optional_json(backtest_path))

    baseline_test = _split_metrics(baseline_metrics, "test")
    candidate_test = _split_metrics(candidate_metrics, "test")
    baseline_model_version = str(baseline_manifest.get("model_version") or "unknown")
    candidate_model_version = str(candidate_manifest.get("model_version") or "unknown")
    dataset_version = _dataset_version(baseline_manifest, candidate_manifest)

    checks = tuple(
        _build_checks(
            baseline_test=baseline_test,
            candidate_test=candidate_test,
            calibration=calibration,
            backtest=backtest,
            dataset_version=dataset_version,
            baseline_model_version=baseline_model_version,
            candidate_model_version=candidate_model_version,
            rules=active_rules,
        )
    )
    passed = all(check.passed for check in checks)
    report = PromotionReport(
        passed=passed,
        decision="promote" if passed else "reject",
        baseline_model_version=baseline_model_version,
        candidate_model_version=candidate_model_version,
        dataset_version=dataset_version,
        checks=checks,
        baseline_test_metrics=baseline_test,
        candidate_test_metrics=candidate_test,
        calibration=calibration,
        backtest=backtest,
        rules=active_rules,
        artifact_paths={
            "baseline_dir": str(baseline_root),
            "candidate_dir": str(candidate_root),
            "calibration_dir": str(calibration_root),
            "backtest_summary_path": str(backtest_path),
        },
        output_dir=str(target),
    )
    (target / "promotion_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (target / "promotion_checklist.md").write_text(
        _checklist_markdown(report),
        encoding="utf-8",
    )
    return report


def _build_checks(
    *,
    baseline_test: dict[str, float | int | None],
    candidate_test: dict[str, float | int | None],
    calibration: dict[str, Any] | None,
    backtest: dict[str, Any] | None,
    dataset_version: str | None,
    baseline_model_version: str,
    candidate_model_version: str,
    rules: PromotionRules,
) -> list[PromotionCheck]:
    baseline_auc = _metric(baseline_test, "roc_auc")
    candidate_auc = _metric(candidate_test, "roc_auc")
    baseline_brier = _metric(baseline_test, "brier_score")
    candidate_brier = _metric(candidate_test, "brier_score")
    backtest_net_pnl = None if backtest is None else _metric(backtest, "net_pnl")
    calibration_improved = bool(calibration and calibration.get("improved"))
    checks = [
        PromotionCheck(
            name="traceable_versions",
            passed=bool(dataset_version and baseline_model_version and candidate_model_version),
            detail=(
                f"dataset={dataset_version}, baseline={baseline_model_version}, "
                f"candidate={candidate_model_version}"
            ),
        ),
        PromotionCheck(
            name="test_roc_auc_vs_baseline",
            passed=(
                baseline_auc is not None
                and candidate_auc is not None
                and candidate_auc >= baseline_auc + rules.min_roc_auc_delta
            ),
            detail=(
                f"candidate ROC AUC {candidate_auc} vs baseline {baseline_auc}; "
                f"required delta >= {rules.min_roc_auc_delta}"
            ),
        ),
        PromotionCheck(
            name="test_brier_vs_baseline",
            passed=(
                baseline_brier is not None
                and candidate_brier is not None
                and candidate_brier <= baseline_brier + rules.max_brier_delta
            ),
            detail=(
                f"candidate Brier {candidate_brier} vs baseline {baseline_brier}; "
                f"allowed delta <= {rules.max_brier_delta}"
            ),
        ),
        PromotionCheck(
            name="calibration_improved",
            passed=(calibration_improved or not rules.require_calibration_improved),
            detail=f"calibration improved={calibration_improved}",
        ),
        PromotionCheck(
            name="backtest_net_pnl",
            passed=(
                backtest_net_pnl is not None
                and backtest_net_pnl >= rules.min_backtest_net_pnl
            ),
            detail=(
                f"best/backtest net_pnl={backtest_net_pnl}; "
                f"required >= {rules.min_backtest_net_pnl}"
            ),
        ),
    ]
    return checks


def _checklist_markdown(report: PromotionReport) -> str:
    lines = [
        "# Model Promotion Checklist",
        "",
        f"Decision: **{report.decision}**",
        "",
        f"- Baseline: `{report.baseline_model_version}`",
        f"- Candidate: `{report.candidate_model_version}`",
        f"- Dataset: `{report.dataset_version}`",
        "",
        "## Checks",
    ]
    for check in report.checks:
        mark = "x" if check.passed else " "
        lines.append(f"- [{mark}] {check.name}: {check.detail}")
    lines.extend(
        [
            "",
            "## Metrics",
            f"- Baseline test ROC AUC: {report.baseline_test_metrics.get('roc_auc')}",
            f"- Candidate test ROC AUC: {report.candidate_test_metrics.get('roc_auc')}",
            f"- Baseline test Brier: {report.baseline_test_metrics.get('brier_score')}",
            f"- Candidate test Brier: {report.candidate_test_metrics.get('brier_score')}",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _split_metrics(metrics: dict[str, Any], split: str) -> dict[str, float | int | None]:
    row = metrics.get(split)
    if not isinstance(row, dict):
        raise ValueError(f"metrics missing {split!r} split")
    return row


def _dataset_version(
    baseline_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
) -> str | None:
    baseline_version = baseline_manifest.get("dataset_version")
    candidate_version = candidate_manifest.get("dataset_version")
    if baseline_version and candidate_version and baseline_version != candidate_version:
        return None
    return None if candidate_version is None else str(candidate_version)


def _summarize_backtest(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        rows = [row for row in raw if isinstance(row, dict)]
        if not rows:
            return None
        return max(rows, key=lambda row: float(row.get("net_pnl") or float("-inf")))
    if isinstance(raw, dict):
        return raw
    return None


def _metric(row: dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    if value is None:
        return None
    return float(value)

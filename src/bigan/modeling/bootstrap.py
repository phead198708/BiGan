"""First-champion bootstrap decision reports."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW"]
BootstrapPromotionAction = Literal["first_champion", "replace_champion"]


@dataclass(frozen=True, slots=True)
class BootstrapRules:
    """Conservative gates for creating the first production champion."""

    min_roc_auc_delta: float = 0.01
    max_brier_delta: float = 0.0
    min_backtest_net_pnl: float = 0.0
    require_cost_adjusted_backtest: bool = True
    require_shadow_evaluation: bool = False
    # Allow lower Sharpe than baseline when probability quality is materially better
    # and trading utility still clears positive Sharpe plus positive net-PnL delta.
    allow_lower_sharpe_if_brier_gap: float = 0.05
    max_global_ece: float | None = None
    max_execution_subset_ece: float | None = None
    min_high_up_realized_up_rate: float | None = None
    min_high_down_realized_down_rate: float | None = None
    require_positive_avg_return_by_family: bool = False

    def to_dict(self) -> dict[str, bool | float | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BootstrapCandidateInput:
    """Artifact locations for one initial champion candidate."""

    candidate_dir: Path | str
    calibration_dir: Path | str | None = None
    candidate_backtest_summary_path: Path | str | None = None
    serving_readiness_path: Path | str | None = None
    feature_schema_path: Path | str | None = None
    model_complexity_notes_path: Path | str | None = None
    shadow_evaluation_path: Path | str | None = None


@dataclass(frozen=True, slots=True)
class BootstrapGateResult:
    """Hard-gate result for one candidate."""

    model_version: str
    passed: bool
    reason: str

    def to_dict(self) -> dict[str, bool | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BootstrapComparisonRow:
    """Baseline or candidate comparison row for the strict Markdown table."""

    model: str
    offline: str
    calibration: str
    backtest: str
    production_readiness: str
    simplicity: str
    verdict: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BootstrapChecklist:
    """Promotion checklist for the best available candidate."""

    beats_baseline: bool
    calibration_acceptable: bool
    backtest_acceptable: bool
    serving_readiness_acceptable: bool
    rollback_fallback_available: bool
    schema_stable: bool
    simple_enough: bool

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BootstrapCandidateAssessment:
    """Internal assessment for a single candidate."""

    model_version: str
    gate: BootstrapGateResult
    row: BootstrapComparisonRow
    checklist: BootstrapChecklist
    score: float
    missing_or_weak_evidence: tuple[str, ...]
    risks: tuple[str, ...]
    next_actions: tuple[str, ...]
    explicit_unacceptable: bool


@dataclass(frozen=True, slots=True)
class BootstrapChampionReport:
    """Serializable first-champion bootstrap decision report."""

    recommended_action: str
    confidence_level: ConfidenceLevel
    baseline_name: str
    baseline_type: str
    baseline_identification: str
    missing_or_weak_evidence: tuple[str, ...]
    hard_gate_results: tuple[BootstrapGateResult, ...]
    comparison_rows: tuple[BootstrapComparisonRow, ...]
    promotion_checklist: BootstrapChecklist
    risks: tuple[str, ...]
    next_actions: tuple[str, ...]
    bootstrap_rationale: str
    artifact_paths: dict[str, str | None]
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_action": self.recommended_action,
            "confidence_level": self.confidence_level,
            "baseline_identified": {
                "baseline_name": self.baseline_name,
                "baseline_type": self.baseline_type,
                "whether_explicit_or_inferred": self.baseline_identification,
            },
            "missing_or_weak_evidence": list(self.missing_or_weak_evidence),
            "hard_gate_results": [gate.to_dict() for gate in self.hard_gate_results],
            "baseline_vs_candidate_table": [row.to_dict() for row in self.comparison_rows],
            "bootstrap_promotion_checklist": self.promotion_checklist.to_dict(),
            "risks": list(self.risks),
            "next_actions": list(self.next_actions),
            "bootstrap_rationale": self.bootstrap_rationale,
            "artifact_paths": self.artifact_paths,
            "output_dir": self.output_dir,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Bootstrap Champion Decision",
            "",
            "## Recommended Action",
            self.recommended_action,
            "",
            "## Confidence Level",
            self.confidence_level,
            "",
            "## Baseline Identified",
            f"- {self.baseline_name}",
            f"- {self.baseline_type}",
            f"- {self.baseline_identification}",
            "",
            "## Missing or Weak Evidence",
        ]
        if self.missing_or_weak_evidence:
            lines.extend(f"- {item}" for item in self.missing_or_weak_evidence)
        else:
            lines.append("None")
        lines.extend(["", "## Hard Gate Results", "For each candidate:"])
        for gate in self.hard_gate_results:
            if gate.passed:
                lines.append(f"- {gate.model_version}: PASS")
            else:
                lines.append(f"- {gate.model_version}: FAIL - {gate.reason}")
        lines.extend(
            [
                "",
                "## Baseline vs Candidate Table",
                "| Model | Offline | Calibration | Backtest | Production Readiness | Simplicity | Verdict |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        lines.extend(
            "| "
            + " | ".join(
                (
                    _escape_table(row.model),
                    _escape_table(row.offline),
                    _escape_table(row.calibration),
                    _escape_table(row.backtest),
                    _escape_table(row.production_readiness),
                    _escape_table(row.simplicity),
                    _escape_table(row.verdict),
                )
            )
            + " |"
            for row in self.comparison_rows
        )
        checklist = self.promotion_checklist
        lines.extend(
            [
                "",
                "## Bootstrap Promotion Checklist",
                _checklist_line(
                    checklist.beats_baseline,
                    "beats or justifies replacing baseline",
                ),
                _checklist_line(checklist.calibration_acceptable, "calibration acceptable"),
                _checklist_line(checklist.backtest_acceptable, "backtest acceptable"),
                _checklist_line(
                    checklist.serving_readiness_acceptable,
                    "serving readiness acceptable",
                ),
                _checklist_line(
                    checklist.rollback_fallback_available,
                    "rollback/fallback available",
                ),
                _checklist_line(checklist.schema_stable, "schema stable"),
                _checklist_line(
                    checklist.simple_enough,
                    "simple enough for v1 production",
                ),
                "",
                "## Risks",
            ]
        )
        lines.extend(f"- {risk}" for risk in self.risks[:5])
        lines.extend(["", "## Next Actions"])
        lines.extend(f"- {action}" for action in self.next_actions[:5])
        lines.extend(["", "## Bootstrap Rationale", self.bootstrap_rationale])
        return "\n".join(lines) + "\n"


def evaluate_bootstrap_champion(
    *,
    baseline_dir: Path | str | None,
    candidates: tuple[BootstrapCandidateInput, ...],
    output_dir: Path | str,
    baseline_type: str = "logistic regression baseline",
    baseline_explicit: bool = True,
    baseline_backtest_summary_path: Path | str | None = None,
    rollback_runbook_path: Path | str | None = Path("docs/runbooks/model_rollback.md"),
    rules: BootstrapRules | None = None,
    promotion_action: BootstrapPromotionAction = "first_champion",
) -> BootstrapChampionReport:
    """Evaluate whether any initial candidate is ready to be the first champion."""

    active_rules = rules or BootstrapRules()
    if promotion_action not in {"first_champion", "replace_champion"}:
        raise ValueError("promotion_action must be 'first_champion' or 'replace_champion'")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    baseline = _load_model_run(baseline_dir, fallback_name="inferred_naive_prior_15m")
    baseline_backtest = _load_backtest_summary(baseline_backtest_summary_path)
    rollback_ready = _rollback_ready(rollback_runbook_path, baseline)
    baseline_artifacts = _bootstrap_artifact_paths(
        baseline_dir=baseline_dir,
        baseline_backtest_summary_path=baseline_backtest_summary_path,
        rollback_runbook_path=rollback_runbook_path,
    )
    baseline_row = _baseline_row(
        baseline=baseline,
        baseline_backtest=baseline_backtest,
        rollback_ready=rollback_ready,
    )

    assessments = tuple(
        _assess_candidate(
            baseline=baseline,
            baseline_backtest=baseline_backtest,
            candidate_input=candidate,
            rollback_ready=rollback_ready,
            rules=active_rules,
        )
        for candidate in candidates
    )

    if not assessments:
        gate_results = (
            BootstrapGateResult(
                model_version="candidate_unspecified",
                passed=False,
                reason="no candidate model/version or evaluation evidence provided",
            ),
        )
        checklist = BootstrapChecklist(False, False, False, False, rollback_ready, False, False)
        report = BootstrapChampionReport(
            recommended_action="CONTINUE_BOOTSTRAP_EXPERIMENTATION",
            confidence_level="HIGH",
            baseline_name=baseline.model_version,
            baseline_type=baseline_type,
            baseline_identification=_identification_label(baseline_explicit),
            missing_or_weak_evidence=(
                "No candidate model versions provided",
                "No candidate evaluation evidence provided",
            ),
            hard_gate_results=gate_results,
            comparison_rows=(baseline_row,),
            promotion_checklist=checklist,
            risks=("Promoting now would be unsupported by candidate evidence.",),
            next_actions=(
                "Train at least one simple candidate and save metrics, calibration, and backtest artifacts.",
                "Add serving readiness, schema validation, and rollback evidence before promotion review.",
            ),
            bootstrap_rationale=(
                "No first champion can be selected without a concrete candidate and evidence bundle."
            ),
            artifact_paths=baseline_artifacts,
            output_dir=str(target),
        )
        return _write_bootstrap_report(report, target)

    passing = [assessment for assessment in assessments if assessment.gate.passed]
    if passing:
        best = max(passing, key=lambda assessment: assessment.score)
        recommended_action = (
            "PROMOTE_CHAMPION"
            if promotion_action == "replace_champion"
            else f"PROMOTE_FIRST_CHAMPION:{best.model_version}"
        )
        confidence: ConfidenceLevel = "HIGH"
        rationale = (
            f"{best.model_version} clears the hard bootstrap gates and is good enough for "
            "a v1 champion. This does not mean it is globally best; it means the evidence "
            "supports a simple, monitorable production starting point."
        )
    else:
        best = max(assessments, key=lambda assessment: assessment.score)
        if best.explicit_unacceptable:
            recommended_action = "KEEP_BASELINE_TEMPORARILY"
            confidence = "HIGH"
            rationale = (
                "The safest decision is to keep the baseline temporarily because the best "
                "candidate has explicit hard-gate failures. The candidate can remain an "
                "experiment, but it is not safe enough to become the first champion."
            )
        else:
            recommended_action = "CONTINUE_BOOTSTRAP_EXPERIMENTATION"
            confidence = "MEDIUM"
            rationale = (
                "The candidate signal is promising, but critical promotion evidence is "
                "incomplete. Continue bootstrap experimentation before establishing the "
                "first production champion."
            )

    missing = _dedupe(
        _baseline_missing_evidence(
            baseline=baseline,
            baseline_backtest=baseline_backtest,
            baseline_explicit=baseline_explicit,
        )
        + tuple(item for assessment in assessments for item in assessment.missing_or_weak_evidence)
    )
    risks = _dedupe(tuple(risk for assessment in assessments for risk in assessment.risks))[:5]
    next_actions = _dedupe(
        tuple(action for assessment in assessments for action in assessment.next_actions)
    )[:5]
    report = BootstrapChampionReport(
        recommended_action=recommended_action,
        confidence_level=confidence,
        baseline_name=baseline.model_version,
        baseline_type=baseline_type,
        baseline_identification=_identification_label(baseline_explicit),
        missing_or_weak_evidence=missing,
        hard_gate_results=tuple(assessment.gate for assessment in assessments),
        comparison_rows=(baseline_row, *(assessment.row for assessment in assessments)),
        promotion_checklist=best.checklist,
        risks=risks or ("Residual model risk remains until online monitoring accumulates outcomes.",),
        next_actions=next_actions
        or ("Monitor prediction quality, calibration, latency, drift, and rollback readiness.",),
        bootstrap_rationale=rationale,
        artifact_paths=_bootstrap_artifact_paths(
            baseline_dir=baseline_dir,
            baseline_backtest_summary_path=baseline_backtest_summary_path,
            rollback_runbook_path=rollback_runbook_path,
            candidate_input=_candidate_input_by_version(candidates, best.model_version),
        ),
        output_dir=str(target),
    )
    return _write_bootstrap_report(report, target)


@dataclass(frozen=True, slots=True)
class _ModelRun:
    model_version: str
    model_dir: Path | None
    manifest: dict[str, Any] | None
    metrics: dict[str, Any] | None


def _assess_candidate(
    *,
    baseline: _ModelRun,
    baseline_backtest: dict[str, Any] | None,
    candidate_input: BootstrapCandidateInput,
    rollback_ready: bool,
    rules: BootstrapRules,
) -> BootstrapCandidateAssessment:
    candidate = _load_model_run(candidate_input.candidate_dir, fallback_name="candidate_unknown")
    test_metrics = _split_metrics(candidate.metrics, "test")
    baseline_test_metrics = _split_metrics(baseline.metrics, "test")
    offline = _offline_assessment(baseline_test_metrics, test_metrics, rules)
    calibration = _calibration_assessment(candidate_input.calibration_dir, rules)
    candidate_backtest = _backtest_assessment(
        candidate_input.candidate_backtest_summary_path,
        baseline_backtest=baseline_backtest,
        rules=rules,
        brier_improvement=_metric(offline, "brier_improvement"),
    )
    serving = _serving_assessment(candidate_input.serving_readiness_path)
    shadow = _shadow_assessment(candidate_input.shadow_evaluation_path, rules)
    schema = _schema_assessment(candidate_input, candidate)
    simplicity = _simplicity_assessment(candidate, candidate_input.model_complexity_notes_path)

    checklist = BootstrapChecklist(
        beats_baseline=offline["passed"],
        calibration_acceptable=calibration["passed"],
        backtest_acceptable=candidate_backtest["passed"],
        serving_readiness_acceptable=serving["passed"] and shadow["passed"],
        rollback_fallback_available=rollback_ready,
        schema_stable=schema["passed"],
        simple_enough=simplicity["passed"],
    )
    failed_reasons = [
        str(item["reason"])
        for item in (offline, calibration, candidate_backtest, serving, shadow, schema)
        if not bool(item["passed"])
    ]
    gate_passed = (
        checklist.beats_baseline
        and checklist.calibration_acceptable
        and checklist.backtest_acceptable
        and checklist.serving_readiness_acceptable
        and checklist.rollback_fallback_available
        and checklist.schema_stable
    )
    gate = BootstrapGateResult(
        model_version=candidate.model_version,
        passed=gate_passed,
        reason="all hard bootstrap gates passed" if gate_passed else "; ".join(failed_reasons),
    )
    missing_or_weak = tuple(
        item
        for assessment in (
            offline,
            calibration,
            candidate_backtest,
            serving,
            shadow,
            schema,
            simplicity,
        )
        for item in assessment["missing_or_weak"]
    )
    row = BootstrapComparisonRow(
        model=candidate.model_version,
        offline=str(offline["summary"]),
        calibration=str(calibration["summary"]),
        backtest=str(candidate_backtest["summary"]),
        production_readiness=_join_summaries(serving["summary"], shadow["summary"]),
        simplicity=str(simplicity["summary"]),
        verdict="Eligible for v1 champion" if gate_passed else "Not eligible for promotion",
    )
    risks = tuple(
        item
        for assessment in (
            offline,
            calibration,
            candidate_backtest,
            serving,
            shadow,
            schema,
            simplicity,
        )
        for item in assessment["risks"]
    )
    next_actions = tuple(
        item
        for assessment in (offline, calibration, candidate_backtest, serving, shadow, schema)
        for item in assessment["next_actions"]
    )
    score = _weighted_score(
        offline=offline,
        backtest=candidate_backtest,
        serving=serving,
        simplicity=simplicity,
    )
    explicit_unacceptable = any(
        bool(assessment["explicit_unacceptable"])
        for assessment in (offline, calibration, candidate_backtest, serving, shadow, schema)
    )
    return BootstrapCandidateAssessment(
        model_version=candidate.model_version,
        gate=gate,
        row=row,
        checklist=checklist,
        score=score,
        missing_or_weak_evidence=missing_or_weak,
        risks=risks,
        next_actions=next_actions,
        explicit_unacceptable=explicit_unacceptable,
    )


def _candidate_input_by_version(
    candidates: tuple[BootstrapCandidateInput, ...],
    model_version: str,
) -> BootstrapCandidateInput | None:
    for candidate_input in candidates:
        candidate = _load_model_run(candidate_input.candidate_dir, fallback_name="candidate_unspecified")
        if candidate.model_version == model_version:
            return candidate_input
    return None


def _bootstrap_artifact_paths(
    *,
    baseline_dir: Path | str | None,
    baseline_backtest_summary_path: Path | str | None,
    rollback_runbook_path: Path | str | None,
    candidate_input: BootstrapCandidateInput | None = None,
) -> dict[str, str | None]:
    return {
        "baseline_dir": _path_str(baseline_dir),
        "baseline_eval_dir": _path_str(baseline_dir),
        "baseline_backtest_summary_path": _path_str(baseline_backtest_summary_path),
        "candidate_dir": _path_str(None if candidate_input is None else candidate_input.candidate_dir),
        "candidate_eval_dir": _path_str(
            None if candidate_input is None else candidate_input.candidate_dir
        ),
        "calibration_dir": _path_str(None if candidate_input is None else candidate_input.calibration_dir),
        "candidate_backtest_summary_path": _path_str(
            None if candidate_input is None else candidate_input.candidate_backtest_summary_path
        ),
        "serving_readiness_path": _path_str(
            None if candidate_input is None else candidate_input.serving_readiness_path
        ),
        "feature_schema_path": _path_str(
            None if candidate_input is None else candidate_input.feature_schema_path
        ),
        "model_complexity_notes_path": _path_str(
            None if candidate_input is None else candidate_input.model_complexity_notes_path
        ),
        "shadow_evaluation_path": _path_str(
            None if candidate_input is None else candidate_input.shadow_evaluation_path
        ),
        "rollback_runbook_path": _path_str(rollback_runbook_path),
    }


def _path_str(path: Path | str | None) -> str | None:
    return None if path is None else str(path)


def _offline_assessment(
    baseline_test: dict[str, Any] | None,
    candidate_test: dict[str, Any] | None,
    rules: BootstrapRules,
) -> dict[str, Any]:
    missing: list[str] = []
    risks: list[str] = []
    next_actions: list[str] = []
    if baseline_test is None or candidate_test is None:
        missing.append("Offline test metrics missing for baseline or candidate")
        next_actions.append("Write baseline and candidate test metrics with Brier, ROC AUC, PR AUC, and sample count.")
        return _assessment(False, "Offline metrics missing", missing, risks, next_actions)

    baseline_auc = _metric(baseline_test, "roc_auc")
    candidate_auc = _metric(candidate_test, "roc_auc")
    baseline_brier = _metric(baseline_test, "brier_score")
    candidate_brier = _metric(candidate_test, "brier_score")
    candidate_pr_auc = _metric(candidate_test, "pr_auc")
    sample_count = _metric(candidate_test, "sample_count")
    for metric_name, value in (
        ("baseline test ROC AUC", baseline_auc),
        ("candidate test ROC AUC", candidate_auc),
        ("baseline test Brier", baseline_brier),
        ("candidate test Brier", candidate_brier),
        ("candidate test PR AUC", candidate_pr_auc),
    ):
        if value is None:
            missing.append(f"{metric_name} missing")
    if sample_count is None or sample_count <= 0:
        missing.append("candidate test sample count missing or zero")
    if missing:
        next_actions.append("Regenerate offline evaluation with all required predictive metrics.")
        return _assessment(False, "Offline metrics incomplete", missing, risks, next_actions)

    auc_delta = float(candidate_auc) - float(baseline_auc)
    brier_delta = float(candidate_brier) - float(baseline_brier)
    brier_improvement = float(baseline_brier) - float(candidate_brier)
    passed = auc_delta >= rules.min_roc_auc_delta and brier_delta <= rules.max_brier_delta
    summary = (
        f"AUC {candidate_auc:.4f} vs {baseline_auc:.4f}; "
        f"Brier {candidate_brier:.4f} vs {baseline_brier:.4f}"
    )
    if not passed:
        risks.append("Candidate offline lift is not strong enough to justify replacing the baseline.")
        next_actions.append("Try a simpler feature or calibration variant and require clear test-set lift over baseline.")
    assessment = _assessment(
        passed,
        summary,
        missing,
        risks,
        next_actions,
        explicit_unacceptable=not passed,
        quality_score=max(0.0, min(1.0, 0.5 + auc_delta * 5.0 - max(0.0, brier_delta) * 5.0)),
    )
    return {
        **assessment,
        "auc_delta": auc_delta,
        "brier_delta": brier_delta,
        "brier_improvement": brier_improvement,
    }


def _calibration_assessment(
    calibration_dir: Path | str | None,
    rules: BootstrapRules,
) -> dict[str, Any]:
    missing: list[str] = []
    risks: list[str] = []
    next_actions: list[str] = []
    report_path = None if calibration_dir is None else Path(calibration_dir) / "calibration_report.json"
    report = _read_optional_json(report_path)
    if report is None:
        missing.append("Calibration report missing")
        risks.append("Probability quality is unknown without calibration evidence.")
        next_actions.append("Fit and save calibration_report.json with raw and calibrated Brier/ECE.")
        return _assessment(False, "Calibration unknown", missing, risks, next_actions)

    calibrated = report.get("calibrated_metrics") if isinstance(report, dict) else None
    ece = _metric(calibrated, "ece") if isinstance(calibrated, dict) else None
    brier = _metric(calibrated, "brier_score") if isinstance(calibrated, dict) else None
    improved = bool(report.get("improved")) if isinstance(report, dict) else False
    if ece is None:
        missing.append("Calibrated ECE missing")
    if brier is None:
        missing.append("Calibrated Brier missing")
    gate_missing = _calibration_gate_missing(report, rules)
    gate_failures = _bucket_level_calibration_gate_failures(report, rules)
    passed = improved and ece is not None and brier is not None and not gate_missing and not gate_failures
    summary_parts = []
    if ece is not None and brier is not None:
        summary_parts.append(f"{report.get('method', 'unknown')} ECE {ece:.4f}, Brier {brier:.4f}")
    execution_ece = _execution_subset_ece(report)
    if execution_ece is not None:
        summary_parts.append(f"execution ECE {execution_ece:.4f}")
    if report.get("family_metrics"):
        summary_parts.append(f"families {len(report.get('family_metrics') or {})}")
    if gate_failures:
        summary_parts.append("bucket/family gate FAIL")
    summary = ", ".join(summary_parts) if summary_parts else "Calibration incomplete or not improved"
    if not passed:
        risks.append("Candidate calibration is poor or unknown.")
        risks.extend(gate_failures)
        next_actions.append("Recalibrate on validation data and verify calibrated Brier/ECE on holdout data.")
    return _assessment(
        passed,
        summary,
        missing + gate_missing,
        risks,
        next_actions,
        explicit_unacceptable=not passed and not missing,
        quality_score=1.0 if passed else 0.0,
    )


def _calibration_gate_missing(report: dict[str, Any], rules: BootstrapRules) -> list[str]:
    missing: list[str] = []
    bucket_metrics = report.get("bucket_metrics")
    family_metrics = report.get("family_metrics")
    if (
        rules.min_high_up_realized_up_rate is not None
        and _bucket_metric(bucket_metrics, "high_up", ("realized_up_rate", "positive_rate", "up_rate")) is None
    ):
        missing.append("high_up bucket realized up rate missing")
    if rules.min_high_down_realized_down_rate is not None and _high_down_realized_rate(bucket_metrics) is None:
        missing.append("high_down bucket realized down rate missing")
    if rules.max_execution_subset_ece is not None and _execution_subset_ece(report) is None:
        missing.append("execution subset ECE missing")
    if rules.require_positive_avg_return_by_family and not isinstance(family_metrics, dict):
        missing.append("family calibration metrics missing")
    elif rules.require_positive_avg_return_by_family and not family_metrics:
        missing.append("family calibration metrics empty")
    return missing


def _bucket_level_calibration_gate_failures(
    report: dict[str, Any],
    rules: BootstrapRules,
) -> list[str]:
    failures: list[str] = []
    calibrated = report.get("calibrated_metrics") if isinstance(report, dict) else None
    ece = _metric(calibrated, "ece") if isinstance(calibrated, dict) else None
    if rules.max_global_ece is not None and ece is not None and ece >= rules.max_global_ece:
        failures.append(f"Global ECE {ece:.4f} does not beat gate {rules.max_global_ece:.4f}.")
    execution_ece = _execution_subset_ece(report)
    if (
        rules.max_execution_subset_ece is not None
        and execution_ece is not None
        and execution_ece >= rules.max_execution_subset_ece
    ):
        failures.append(
            f"Execution subset ECE {execution_ece:.4f} does not beat gate "
            f"{rules.max_execution_subset_ece:.4f}."
        )
    bucket_metrics = report.get("bucket_metrics")
    high_up = _bucket_metric(bucket_metrics, "high_up", ("realized_up_rate", "positive_rate", "up_rate"))
    if (
        rules.min_high_up_realized_up_rate is not None
        and high_up is not None
        and high_up <= rules.min_high_up_realized_up_rate
    ):
        failures.append(
            f"high_up realized up rate {high_up:.4f} <= "
            f"{rules.min_high_up_realized_up_rate:.4f}."
        )
    high_down = _high_down_realized_rate(bucket_metrics)
    if (
        rules.min_high_down_realized_down_rate is not None
        and high_down is not None
        and high_down <= rules.min_high_down_realized_down_rate
    ):
        failures.append(
            f"high_down realized down rate {high_down:.4f} <= "
            f"{rules.min_high_down_realized_down_rate:.4f}."
        )
    family_metrics = report.get("family_metrics")
    if rules.require_positive_avg_return_by_family and isinstance(family_metrics, dict):
        bad_families = [
            family
            for family, metrics in family_metrics.items()
            if _metric(metrics, "avg_realized_return") is not None
            and float(_metric(metrics, "avg_realized_return")) <= 0.0
        ]
        missing_returns = [
            family
            for family, metrics in family_metrics.items()
            if _metric(metrics, "avg_realized_return") is None
        ]
        if bad_families:
            failures.append(
                "Family avg realized return is non-positive for "
                + ", ".join(sorted(map(str, bad_families)))
                + "."
            )
        if missing_returns:
            failures.append(
                "Family avg realized return missing for "
                + ", ".join(sorted(map(str, missing_returns)))
                + "."
            )
    return failures


def _execution_subset_ece(report: dict[str, Any]) -> float | None:
    metrics = report.get("execution_subset_metrics")
    if isinstance(metrics, dict):
        calibrated = metrics.get("calibrated_metrics")
        if isinstance(calibrated, dict):
            return _metric(calibrated, "ece")
    family_metrics = report.get("family_metrics")
    if isinstance(family_metrics, dict):
        subset = family_metrics.get("_execution_subset")
        if isinstance(subset, dict):
            calibrated = subset.get("calibrated_metrics")
            if isinstance(calibrated, dict):
                return _metric(calibrated, "ece")
    return None


def _high_down_realized_rate(bucket_metrics: Any) -> float | None:
    realized_down = _bucket_metric(
        bucket_metrics,
        "high_down",
        ("realized_down_rate", "negative_rate", "down_rate"),
    )
    if realized_down is not None:
        return realized_down
    realized_up = _bucket_metric(
        bucket_metrics,
        "high_down",
        ("realized_up_rate", "positive_rate", "up_rate"),
    )
    return None if realized_up is None else 1.0 - realized_up


def _bucket_metric(
    bucket_metrics: Any,
    bucket_name: str,
    metric_names: tuple[str, ...],
) -> float | None:
    bucket = None
    if isinstance(bucket_metrics, dict):
        bucket = bucket_metrics.get(bucket_name)
    elif isinstance(bucket_metrics, list):
        bucket = next(
            (
                row
                for row in bucket_metrics
                if isinstance(row, dict) and row.get("bucket") == bucket_name
            ),
            None,
        )
    if not isinstance(bucket, dict):
        return None
    return _first_metric(bucket, metric_names)


def _backtest_assessment(
    backtest_summary_path: Path | str | None,
    *,
    baseline_backtest: dict[str, Any] | None,
    rules: BootstrapRules,
    brier_improvement: float | None = None,
) -> dict[str, Any]:
    missing: list[str] = []
    risks: list[str] = []
    next_actions: list[str] = []
    row = _load_backtest_summary(backtest_summary_path)
    if row is None:
        missing.append("Candidate cost-adjusted backtest summary missing")
        risks.append("Trading utility is unknown without a backtest.")
        next_actions.append("Run candidate and baseline threshold backtests after realistic fees, slippage, and latency.")
        return _assessment(False, "Backtest missing", missing, risks, next_actions)
    if baseline_backtest is None:
        missing.append("Baseline cost-adjusted backtest missing")

    net_pnl = _metric(row, "net_pnl")
    trade_count = _metric(row, "trade_count")
    baseline_net_pnl = _metric(baseline_backtest, "net_pnl")
    net_pnl_delta = (
        None
        if net_pnl is None or baseline_net_pnl is None
        else net_pnl - baseline_net_pnl
    )
    max_drawdown = _first_metric(row, ("max_drawdown", "max_drawdown_pct"))
    sharpe = _first_metric(row, ("sharpe_ratio", "sharpe"))
    baseline_sharpe = _first_metric(baseline_backtest, ("sharpe_ratio", "sharpe"))
    sortino = _first_metric(row, ("sortino_ratio", "sortino"))
    turnover = _first_metric(
        row,
        (
            "turnover",
            "turnover_trades_per_signal",
            "turnover_trades_per_1000_signals",
            "trades_per_1000_signals",
            "trades_per_day",
        ),
    )
    settings = row.get("settings") if isinstance(row, dict) else None
    fee_bps = _metric(settings, "fee_bps") if isinstance(settings, dict) else None
    slippage_bps = _metric(settings, "slippage_bps") if isinstance(settings, dict) else None
    if net_pnl is None:
        missing.append("Candidate backtest net_pnl missing")
    if trade_count is None:
        missing.append("Candidate backtest trade_count missing")
    if baseline_backtest is not None and baseline_net_pnl is None:
        missing.append("Baseline backtest net_pnl missing")
    if max_drawdown is None:
        missing.append("Backtest max drawdown missing")
    if sharpe is None and sortino is None:
        missing.append("Backtest Sharpe/Sortino missing")
    if turnover is None:
        missing.append("Backtest turnover missing")
    if not _concentration_available(row):
        missing.append("Backtest concentration missing")
    cost_adjusted = (
        fee_bps is not None
        and slippage_bps is not None
        and (fee_bps > 0.0 or slippage_bps > 0.0)
    )
    if rules.require_cost_adjusted_backtest and not cost_adjusted:
        missing.append("Backtest is not cost-adjusted with nonzero fee_bps or slippage_bps")
    if missing:
        next_actions.append("Rerun threshold backtests with candidate and baseline, realistic costs, drawdown, Sharpe/Sortino, and turnover.")

    explicit_bad = (
        (net_pnl is not None and net_pnl < rules.min_backtest_net_pnl)
        or (trade_count is not None and trade_count <= 0)
        or (net_pnl_delta is not None and net_pnl_delta < 0.0)
    )
    lower_sharpe_allowed = _lower_sharpe_allowed(
        candidate_sharpe=sharpe,
        baseline_sharpe=baseline_sharpe,
        net_pnl_delta=net_pnl_delta,
        brier_improvement=brier_improvement,
        rules=rules,
    )
    lower_sharpe_unjustified = (
        sharpe is not None
        and baseline_sharpe is not None
        and sharpe < baseline_sharpe
        and not lower_sharpe_allowed
    )
    explicit_bad = explicit_bad or lower_sharpe_unjustified
    if explicit_bad:
        if net_pnl_delta is not None and net_pnl_delta < 0.0:
            risks.append("Candidate cost-adjusted backtest underperforms the baseline.")
        elif lower_sharpe_unjustified:
            risks.append("Candidate Sharpe underperforms the baseline without enough Brier/net-PnL justification.")
        else:
            risks.append("Candidate backtest utility is unacceptable.")
    passed = not missing and not explicit_bad
    summary_parts = []
    if net_pnl is not None and trade_count is not None:
        summary_parts.append(f"net_pnl {net_pnl:.4f}, trades {int(trade_count)}")
    if net_pnl_delta is not None:
        summary_parts.append(f"delta_vs_baseline {net_pnl_delta:.4f}")
    if max_drawdown is not None:
        summary_parts.append(f"max_dd {max_drawdown:.4f}")
    if sharpe is not None:
        summary_parts.append(f"sharpe {sharpe:.4f}")
        if baseline_sharpe is not None:
            summary_parts.append(f"sharpe_delta_vs_baseline {sharpe - baseline_sharpe:.4f}")
    elif sortino is not None:
        summary_parts.append(f"sortino {sortino:.4f}")
    if (
        lower_sharpe_allowed
        and brier_improvement is not None
        and sharpe is not None
        and baseline_sharpe is not None
        and sharpe < baseline_sharpe
    ):
        summary_parts.append(f"lower_sharpe_allowed_brier_gap {brier_improvement:.4f}")
    summary = ", ".join(summary_parts) if summary_parts else "Backtest incomplete"
    return _assessment(
        passed,
        summary,
        missing,
        risks,
        next_actions,
        explicit_unacceptable=explicit_bad,
        quality_score=1.0 if passed else 0.0,
    )


def _lower_sharpe_allowed(
    *,
    candidate_sharpe: float | None,
    baseline_sharpe: float | None,
    net_pnl_delta: float | None,
    brier_improvement: float | None,
    rules: BootstrapRules,
) -> bool:
    if candidate_sharpe is None or baseline_sharpe is None:
        return False
    if candidate_sharpe >= baseline_sharpe:
        return True
    return (
        brier_improvement is not None
        and brier_improvement >= rules.allow_lower_sharpe_if_brier_gap
        and candidate_sharpe > 0.0
        and net_pnl_delta is not None
        and net_pnl_delta > 0.0
    )


def _serving_assessment(serving_readiness_path: Path | str | None) -> dict[str, Any]:
    missing: list[str] = []
    risks: list[str] = []
    next_actions: list[str] = []
    report = _read_optional_json(None if serving_readiness_path is None else Path(serving_readiness_path))
    if report is None:
        missing.append("Serving latency/error readiness report missing")
        risks.append("Serving stability is unverified.")
        next_actions.append("Run a serving readiness check and save latency, error-rate, and health evidence.")
        return _assessment(False, "Serving readiness unknown", missing, risks, next_actions)

    p95_latency = _metric(report, "p95_latency_ms") or _metric(report, "latency_p95_ms")
    latency_sla = _metric(report, "latency_sla_ms") or _metric(report, "max_p95_latency_ms")
    error_rate = _metric(report, "error_rate")
    max_error_rate = _metric(report, "max_error_rate")
    explicit_ready = bool(report.get("ready") or report.get("serving_ready"))
    status = str(report.get("status", "")).lower()
    status_ready = status in {"ok", "ready", "healthy", "passed", "pass"}
    latency_ok = p95_latency is not None and latency_sla is not None and p95_latency <= latency_sla
    error_ok = error_rate is not None and max_error_rate is not None and error_rate <= max_error_rate
    has_required_metrics = p95_latency is not None and error_rate is not None
    passed = has_required_metrics and (explicit_ready or status_ready or (latency_ok and error_ok))
    if p95_latency is None:
        missing.append("Serving p95 latency missing")
    if error_rate is None:
        missing.append("Serving error rate missing")
    if not passed:
        risks.append("Serving path is not proven deployment-ready.")
        next_actions.append("Fix serving health, latency, or error-rate failures before promotion.")
    summary = (
        f"p95 {p95_latency}ms, error_rate {error_rate}"
        if p95_latency is not None or error_rate is not None
        else "Serving readiness incomplete"
    )
    return _assessment(
        passed,
        summary,
        missing if not passed else [],
        risks,
        next_actions,
        explicit_unacceptable=bool(report) and not passed,
        quality_score=1.0 if passed else 0.0,
    )


def _shadow_assessment(
    shadow_evaluation_path: Path | str | None,
    rules: BootstrapRules,
) -> dict[str, Any]:
    missing: list[str] = []
    risks: list[str] = []
    next_actions: list[str] = []
    if shadow_evaluation_path is None:
        if not rules.require_shadow_evaluation:
            return _assessment(
                True,
                "Shadow evaluation not supplied",
                [],
                [],
                [],
                quality_score=0.5,
            )
        missing.append("Shadow evaluation report missing")
        risks.append("Live shadow behavior is unknown.")
        next_actions.append("Run shadow mode and generate shadow evaluation JSON before promotion.")
        return _assessment(False, "Shadow evaluation missing", missing, risks, next_actions)

    report = _read_optional_json(Path(shadow_evaluation_path))
    if report is None:
        missing.append("Shadow evaluation report missing or unreadable")
        risks.append("Live shadow behavior is unknown.")
        next_actions.append("Regenerate shadow evaluation JSON from the shadow run.")
        return _assessment(False, "Shadow evaluation missing", missing, risks, next_actions)
    if not isinstance(report, dict):
        missing.append("Shadow evaluation report is not a JSON object")
        return _assessment(False, "Shadow evaluation malformed", missing, risks, next_actions)

    overall_passed = bool(report.get("overall_passed") or report.get("passed"))
    checks = report.get("checks") if isinstance(report.get("checks"), dict) else {}
    failed_checks = sorted(
        name
        for name, check in checks.items()
        if isinstance(check, dict) and not bool(check.get("passed"))
    )
    challenger_trigger_rate = _metric(report, "challenger_edge_trigger_rate")
    schema_error_rate = _metric(report, "schema_error_rate")
    latency = report.get("latency_ms") if isinstance(report.get("latency_ms"), dict) else {}
    latency_summary = _shadow_latency_summary(latency, str(report.get("challenger_model_version", "")))
    summary_parts = [f"shadow {'PASS' if overall_passed else 'FAIL'}"]
    if challenger_trigger_rate is not None:
        summary_parts.append(f"edge_trigger_rate {challenger_trigger_rate:.4f}")
    if schema_error_rate is not None:
        summary_parts.append(f"schema_error_rate {schema_error_rate:.4f}")
    if latency_summary:
        summary_parts.append(latency_summary)
    if not overall_passed:
        risks.append(
            "Shadow evaluation failed promotion criteria"
            + (f": {', '.join(failed_checks)}" if failed_checks else ".")
        )
        next_actions.append("Fix or rerun shadow evaluation until all pass/fail criteria clear.")
    return _assessment(
        overall_passed,
        ", ".join(summary_parts),
        missing,
        risks,
        next_actions,
        explicit_unacceptable=not overall_passed,
        quality_score=1.0 if overall_passed else 0.0,
    )


def _schema_assessment(
    candidate_input: BootstrapCandidateInput,
    candidate: _ModelRun,
) -> dict[str, Any]:
    missing: list[str] = []
    risks: list[str] = []
    next_actions: list[str] = []
    default_path = None if candidate.model_dir is None else candidate.model_dir / "feature_schema.json"
    schema_path = Path(candidate_input.feature_schema_path) if candidate_input.feature_schema_path else default_path
    schema = _read_optional_json(schema_path)
    if schema is None:
        missing.append("Feature schema artifact missing")
        risks.append("Candidate input schema expectations are unclear.")
        next_actions.append("Write feature_schema.json with ordered feature columns, types, and schema hash.")
        return _assessment(False, "Schema artifact missing", missing, risks, next_actions)
    columns = schema.get("feature_columns") if isinstance(schema, dict) else None
    schema_hash = schema.get("schema_hash") if isinstance(schema, dict) else None
    passed = isinstance(columns, list) and bool(columns) and bool(schema_hash)
    if not passed:
        missing.append("Feature schema artifact missing columns or schema_hash")
        risks.append("Schema artifact is incomplete.")
        next_actions.append("Regenerate feature_schema.json from the training feature order.")
    return _assessment(
        passed,
        "Feature schema artifact present" if passed else "Schema artifact incomplete",
        missing,
        risks,
        next_actions,
        explicit_unacceptable=not passed and not missing,
        quality_score=1.0 if passed else 0.0,
    )


def _simplicity_assessment(
    candidate: _ModelRun,
    complexity_notes_path: Path | str | None,
) -> dict[str, Any]:
    missing: list[str] = []
    manifest = candidate.manifest or {}
    best_params = manifest.get("best_params") if isinstance(manifest.get("best_params"), dict) else {}
    model_version = candidate.model_version.lower()
    notes_path = _complexity_notes_path(candidate, complexity_notes_path)
    notes = _read_text_optional(notes_path)
    if "logreg" in model_version or "baseline" in model_version:
        return _assessment(True, "Simple/interpretable", [], [], [], quality_score=1.0)
    if "xgboost" in model_version or "xgb" in model_version:
        max_depth = _metric(best_params, "max_depth")
        rounds = _metric(best_params, "rounds")
        shallow = max_depth is not None and max_depth <= 3
        compact = rounds is not None and rounds <= 100
        if shallow and compact:
            return _assessment(True, f"Shallow XGBoost depth {int(max_depth)}, rounds {int(rounds)}", [], [], [], quality_score=0.7)
        if notes is not None:
            missing_sections = _missing_complexity_note_sections(notes)
            if not missing_sections:
                return _assessment(
                    True,
                    "Model card complexity notes present",
                    [],
                    [],
                    [],
                    quality_score=0.8,
                )
            missing.extend(
                f"Model complexity notes missing {section}" for section in missing_sections
            )
    if notes is None:
        missing.append("Model complexity notes missing")
    return _assessment(
        False,
        "Complexity not documented",
        missing,
        ["Candidate maintainability is not documented."],
        ["Add model complexity notes covering dependencies, retraining, interpretability, and feature stability."],
        quality_score=0.3,
    )


def _complexity_notes_path(
    candidate: _ModelRun,
    complexity_notes_path: Path | str | None,
) -> Path | None:
    if complexity_notes_path is not None:
        return Path(complexity_notes_path)
    candidates = []
    if candidate.model_dir is not None:
        candidates.extend(
            [
                candidate.model_dir / "model_card.md",
                candidate.model_dir / "model_complexity.md",
            ]
        )
    candidates.append(Path("docs") / "models" / f"{candidate.model_version}.md")
    for path in candidates:
        if path.exists():
            return path
    return None


def _missing_complexity_note_sections(notes: str) -> tuple[str, ...]:
    lower = notes.lower()
    requirements = {
        "dependencies": ("dependencies", "dependency"),
        "training cost": ("training cost", "training time", "retraining cost"),
        "retraining": ("retraining", "retrain"),
        "interpretability": ("interpretability", "feature importance", "contribution"),
        "feature stability": ("feature stability", "schema stability", "stable features"),
        "monitoring": ("monitoring", "monitor"),
    }
    return tuple(
        section
        for section, keywords in requirements.items()
        if not any(keyword in lower for keyword in keywords)
    )


def _shadow_latency_summary(latency: dict[str, Any], challenger_model_version: str) -> str:
    if not latency:
        return ""
    summary = latency.get(challenger_model_version) if challenger_model_version else None
    if not isinstance(summary, dict):
        candidates = [value for value in latency.values() if isinstance(value, dict)]
        summary = candidates[-1] if candidates else None
    if not isinstance(summary, dict):
        return ""
    p95 = _metric(summary, "p95")
    return "" if p95 is None else f"shadow_p95 {p95:.4f}ms"


def _join_summaries(*items: object) -> str:
    return "; ".join(str(item) for item in items if str(item))


def _baseline_row(
    *,
    baseline: _ModelRun,
    baseline_backtest: dict[str, Any] | None,
    rollback_ready: bool,
) -> BootstrapComparisonRow:
    test = _split_metrics(baseline.metrics, "test")
    if test is None:
        offline = "Missing"
    else:
        auc = _metric(test, "roc_auc")
        brier = _metric(test, "brier_score")
        offline = f"AUC {auc:.4f}; Brier {brier:.4f}" if auc is not None and brier is not None else "Incomplete"
    backtest = "Missing" if baseline_backtest is None else f"net_pnl {_metric(baseline_backtest, 'net_pnl')}"
    return BootstrapComparisonRow(
        model=baseline.model_version,
        offline=offline,
        calibration="Not required for temporary baseline",
        backtest=backtest,
        production_readiness="Fallback available" if rollback_ready else "Fallback unverified",
        simplicity="Simple",
        verdict="Temporary baseline/fallback",
    )


def _load_model_run(path: Path | str | None, *, fallback_name: str) -> _ModelRun:
    if path is None:
        return _ModelRun(fallback_name, None, None, None)
    root = Path(path)
    manifest = _read_optional_json(root / "manifest.json")
    metrics = _read_optional_json(root / "metrics.json")
    model_version = fallback_name
    if isinstance(manifest, dict):
        model_version = str(manifest.get("model_version") or fallback_name)
    return _ModelRun(model_version, root, manifest if isinstance(manifest, dict) else None, metrics if isinstance(metrics, dict) else None)


def _load_backtest_summary(path: Path | str | None) -> dict[str, Any] | None:
    raw = _read_optional_json(None if path is None else Path(path))
    if raw is None:
        return None
    if isinstance(raw, list):
        rows = [row for row in raw if isinstance(row, dict)]
        if not rows:
            return None
        return max(rows, key=_backtest_sort_value)
    return raw if isinstance(raw, dict) else None


def _backtest_sort_value(row: dict[str, Any]) -> float:
    net_pnl = _metric(row, "net_pnl")
    return net_pnl if net_pnl is not None else float("-inf")


def _split_metrics(metrics: dict[str, Any] | None, split: str) -> dict[str, Any] | None:
    row = None if metrics is None else metrics.get(split)
    return row if isinstance(row, dict) else None


def _metric(row: dict[str, Any] | None, name: str) -> float | None:
    if not isinstance(row, dict):
        return None
    value = row.get(name)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _first_metric(row: dict[str, Any] | None, names: tuple[str, ...]) -> float | None:
    for name in names:
        value = _metric(row, name)
        if value is not None:
            return value
    return None


def _concentration_available(row: dict[str, Any] | None) -> bool:
    if not isinstance(row, dict):
        return False
    if _metric(row, "top1_market_abs_net_pnl_share") is not None:
        return True
    concentration = row.get("concentration")
    if not isinstance(concentration, dict):
        return False
    return _metric(concentration, "top1_abs_net_pnl_share") is not None


def _read_optional_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text_optional(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _rollback_ready(path: Path | str | None, baseline: _ModelRun) -> bool:
    runbook_ready = path is not None and Path(path).exists()
    fallback_ready = baseline.model_dir is not None and (baseline.model_dir / "model.json").exists()
    return runbook_ready and fallback_ready


def _baseline_missing_evidence(
    *,
    baseline: _ModelRun,
    baseline_backtest: dict[str, Any] | None,
    baseline_explicit: bool,
) -> tuple[str, ...]:
    missing = []
    if not baseline_explicit:
        missing.append("Baseline is inferred, not explicitly selected")
    if baseline.metrics is None:
        missing.append("Baseline offline metrics missing")
    if baseline_backtest is None:
        missing.append("Baseline cost-adjusted backtest missing")
    return tuple(missing)


def _weighted_score(
    *,
    offline: dict[str, Any],
    backtest: dict[str, Any],
    serving: dict[str, Any],
    simplicity: dict[str, Any],
) -> float:
    return (
        float(offline["quality_score"]) * 0.35
        + float(backtest["quality_score"]) * 0.25
        + float(serving["quality_score"]) * 0.25
        + float(simplicity["quality_score"]) * 0.15
    )


def _assessment(
    passed: bool,
    summary: str,
    missing_or_weak: list[str],
    risks: list[str],
    next_actions: list[str],
    *,
    explicit_unacceptable: bool = False,
    quality_score: float = 0.0,
) -> dict[str, Any]:
    return {
        "passed": passed,
        "summary": summary,
        "reason": summary,
        "missing_or_weak": tuple(missing_or_weak),
        "risks": tuple(risks),
        "next_actions": tuple(next_actions),
        "explicit_unacceptable": explicit_unacceptable,
        "quality_score": quality_score,
    }


def _write_bootstrap_report(
    report: BootstrapChampionReport,
    output_dir: Path,
) -> BootstrapChampionReport:
    (output_dir / "bootstrap_decision.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "bootstrap_decision.md").write_text(
        report.to_markdown(),
        encoding="utf-8",
    )
    return report


def _identification_label(explicit: bool) -> str:
    return "explicit" if explicit else "inferred"


def _checklist_line(passed: bool, label: str) -> str:
    return f"- [{'x' if passed else ' '}] {label}"


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _dedupe(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)

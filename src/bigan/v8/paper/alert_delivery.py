"""GitHub-oriented delivery for v8 paper observability reports."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from bigan.v8.paper.contracts import json_ready

PAPER_ALERT_DELIVERY_PHASE = "paper_alert_delivery"
DEFAULT_ALERT_DELIVERY_CREATED_AT = "2026-06-22T05:00:00Z"

GitHubCommentPostMode = Literal["dry_run", "gh_command", "direct_comment"]

GITHUB_COMMENT_OUTPUT_FILENAMES: tuple[str, ...] = (
    "github_paper_comment_payload.json",
    "github_paper_comment.md",
    "github_paper_comment_gh_command.sh",
)

_REQUIRED_OBSERVABILITY_ARTIFACTS: dict[str, str] = {
    "observability_report": "paper_observability_report.json",
    "operator_summary": "paper_operator_summary.md",
    "alerts": "paper_alerts.jsonl",
    "dashboard_summary": "paper_dashboard_summary.json",
}
_OPTIONAL_OBSERVABILITY_ARTIFACTS: dict[str, str] = {
    "periodic_metrics_csv": "paper_periodic_metrics.csv",
    "run_comparison_json": "paper_run_comparison.json",
    "run_comparison_md": "paper_run_comparison.md",
}


class PaperAlertDeliveryError(RuntimeError):
    """Raised when paper alert delivery cannot safely produce a comment."""


@dataclass(frozen=True, slots=True)
class GitHubCommentDeliveryConfig:
    """Configuration for deterministic GitHub paper comment delivery."""

    repo_full_name: str
    issue_number: int
    output_dir: Path | str
    post_mode: GitHubCommentPostMode = "dry_run"
    created_at: str = DEFAULT_ALERT_DELIVERY_CREATED_AT
    overwrite_existing: bool = False
    max_alerts_to_inline: int = 20
    include_hashes: bool = True
    include_artifact_paths: bool = True

    def __post_init__(self) -> None:
        if not self.repo_full_name.strip() or "/" not in self.repo_full_name:
            raise ValueError("repo_full_name must be owner/repo")
        if self.issue_number <= 0:
            raise ValueError("issue_number must be positive")
        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.post_mode not in ("dry_run", "gh_command", "direct_comment"):
            raise ValueError("post_mode must be dry_run, gh_command, or direct_comment")
        if not self.created_at:
            raise ValueError("created_at is required")
        if self.max_alerts_to_inline < 0:
            raise ValueError("max_alerts_to_inline must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


@dataclass(frozen=True, slots=True)
class GitHubPaperCommentPayload:
    """Complete deterministic GitHub issue comment payload."""

    phase: str
    run_id: str
    source_observability_dir: str
    observability_report_sha256: str
    operator_summary_sha256: str
    alerts_sha256: str
    dashboard_summary_sha256: str
    issue_number: int
    repo_full_name: str
    comment_title: str
    comment_body: str
    alert_count: int
    critical_alert_count: int
    warning_alert_count: int
    operator_recommendation: str
    phase6_deployment_status: str
    feed_mode: str
    provider_name: str | None
    instrument_id: str | None
    feed_health_status: str
    safety_status: str
    paper_only: bool
    capital_at_risk: bool
    broker_exchange_write_enabled: bool
    live_exchange_write_enabled: bool
    polymarket_write_enabled: bool
    wallet_signing_enabled: bool
    polymarket_realized_trade_pnl: float
    polymarket_settlement_pnl: float
    polymarket_total_pnl: float
    source_artifact_hashes: dict[str, str]
    input_artifact_hashes: dict[str, str]
    input_artifact_paths: dict[str, str]
    post_mode: GitHubCommentPostMode
    gh_command: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GitHubPaperCommentDeliveryResult:
    """Delivery result and generated output paths."""

    payload: GitHubPaperCommentPayload
    output_dir: Path
    artifact_paths: dict[str, Path]
    delivery_receipt: dict[str, Any] | None = None


def build_github_paper_comment_payload(
    *,
    observability_dir: Path | str,
    config: GitHubCommentDeliveryConfig,
) -> GitHubPaperCommentPayload:
    """Build a deterministic GitHub paper comment payload without writing files."""

    loaded = _load_observability_artifacts(Path(observability_dir))
    return _build_payload(loaded, config=config)


def deliver_github_paper_comment(
    *,
    observability_dir: Path | str,
    config: GitHubCommentDeliveryConfig,
) -> GitHubPaperCommentDeliveryResult:
    """Write GitHub comment artifacts and optionally post the comment."""

    source_path = Path(observability_dir).expanduser().resolve()
    output_path = Path(config.output_dir).expanduser().resolve()
    _assert_output_dir_safe(source_path=source_path, output_path=output_path)
    loaded = _load_observability_artifacts(source_path)

    if output_path.exists():
        if not config.overwrite_existing:
            raise FileExistsError(
                f"paper comment output_dir already exists: {output_path}; "
                "set overwrite_existing=True to replace it"
            )
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    payload = _build_payload(loaded, config=config)
    artifact_paths = {
        "payload": output_path / "github_paper_comment_payload.json",
        "comment_body": output_path / "github_paper_comment.md",
    }
    _write_json(artifact_paths["payload"], payload.to_dict())
    _write_text(artifact_paths["comment_body"], payload.comment_body)

    if config.post_mode in {"gh_command", "direct_comment"}:
        artifact_paths["gh_command"] = (
            output_path / "github_paper_comment_gh_command.sh"
        )
        _write_text(
            artifact_paths["gh_command"],
            (payload.gh_command or _gh_command(config, artifact_paths["comment_body"]))
            + "\n",
        )

    receipt = None
    if config.post_mode == "direct_comment":
        receipt = _post_direct_comment(
            config=config,
            comment_body_path=artifact_paths["comment_body"],
        )
        artifact_paths["delivery_receipt"] = (
            output_path / "github_paper_comment_delivery_receipt.json"
        )
        _write_json(artifact_paths["delivery_receipt"], receipt)

    return GitHubPaperCommentDeliveryResult(
        payload=payload,
        output_dir=output_path,
        artifact_paths=artifact_paths,
        delivery_receipt=receipt,
    )


def _load_observability_artifacts(observability_dir: Path) -> dict[str, Any]:
    source_dir = observability_dir.expanduser().resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise PaperAlertDeliveryError(
            f"observability_dir does not exist: {source_dir}"
        )
    paths = {
        key: source_dir / filename
        for key, filename in _REQUIRED_OBSERVABILITY_ARTIFACTS.items()
    }
    missing = [path.name for path in paths.values() if not path.exists()]
    if missing:
        raise PaperAlertDeliveryError(
            "missing required observability artifacts: "
            + ", ".join(sorted(missing))
        )
    for key, filename in _OPTIONAL_OBSERVABILITY_ARTIFACTS.items():
        path = source_dir / filename
        if path.exists():
            paths[key] = path
    return {
        "source_dir": source_dir,
        "paths": paths,
        "observability_report": _read_json(paths["observability_report"]),
        "operator_summary": paths["operator_summary"].read_text(encoding="utf-8"),
        "alerts": _read_jsonl(paths["alerts"]),
        "dashboard_summary": _read_json(paths["dashboard_summary"]),
        "hashes": {key: _sha256_file(path) for key, path in paths.items()},
    }


def _build_payload(
    loaded: dict[str, Any],
    *,
    config: GitHubCommentDeliveryConfig,
) -> GitHubPaperCommentPayload:
    report = loaded["observability_report"]
    hashes = loaded["hashes"]
    source_artifact_hashes = dict(report.get("source_artifact_hashes", {}))
    input_paths = {
        key: str(path)
        for key, path in sorted(loaded["paths"].items())
        if config.include_artifact_paths
    }
    comment_title = f"v8 Paper Observability Summary: {report['run_id']}"
    comment_body = _comment_body(
        report=report,
        alerts=loaded["alerts"],
        hashes=hashes,
        source_artifact_hashes=source_artifact_hashes,
        input_artifact_paths=input_paths,
        source_observability_dir=str(loaded["source_dir"]),
        config=config,
        comment_title=comment_title,
    )
    command = None
    if config.post_mode in {"gh_command", "direct_comment"}:
        command = _gh_command(config, _comment_body_path_for_command(config))
    return GitHubPaperCommentPayload(
        phase=PAPER_ALERT_DELIVERY_PHASE,
        run_id=str(report["run_id"]),
        source_observability_dir=str(loaded["source_dir"]),
        observability_report_sha256=hashes["observability_report"],
        operator_summary_sha256=hashes["operator_summary"],
        alerts_sha256=hashes["alerts"],
        dashboard_summary_sha256=hashes["dashboard_summary"],
        issue_number=config.issue_number,
        repo_full_name=config.repo_full_name,
        comment_title=comment_title,
        comment_body=comment_body,
        alert_count=int(report["alert_count"]),
        critical_alert_count=int(report["alert_severity_counts"]["critical"]),
        warning_alert_count=int(report["alert_severity_counts"]["warning"]),
        operator_recommendation=str(report["operator_recommendation"]),
        phase6_deployment_status=str(report["phase6_status"]),
        feed_mode=str(report.get("feed_metrics", {}).get("feed_mode")),
        provider_name=report.get("feed_metrics", {}).get("provider_name"),
        instrument_id=report.get("feed_metrics", {}).get("instrument_id"),
        feed_health_status=str(report["feed_health_status"]),
        safety_status=str(report["safety_status"]),
        paper_only=report.get("paper_only") is True,
        capital_at_risk=report.get("capital_at_risk") is True,
        broker_exchange_write_enabled=report.get(
            "broker_exchange_write_enabled"
        )
        is True,
        live_exchange_write_enabled=report.get("live_exchange_write_enabled") is True,
        polymarket_write_enabled=report.get("polymarket_write_enabled") is True,
        wallet_signing_enabled=report.get("wallet_signing_enabled") is True,
        polymarket_realized_trade_pnl=float(
            report.get("performance_metrics", {}).get(
                "polymarket_realized_trade_pnl",
                0.0,
            )
            or 0.0
        ),
        polymarket_settlement_pnl=float(
            report.get("performance_metrics", {}).get(
                "polymarket_settlement_pnl",
                0.0,
            )
            or 0.0
        ),
        polymarket_total_pnl=float(
            report.get("performance_metrics", {}).get("polymarket_total_pnl", 0.0)
            or 0.0
        ),
        source_artifact_hashes=source_artifact_hashes
        if config.include_hashes
        else {},
        input_artifact_hashes=dict(sorted(hashes.items()))
        if config.include_hashes
        else {},
        input_artifact_paths=input_paths,
        post_mode=config.post_mode,
        gh_command=command,
        created_at=config.created_at,
    )


def _comment_body(
    *,
    report: dict[str, Any],
    alerts: list[dict[str, Any]],
    hashes: dict[str, str],
    source_artifact_hashes: dict[str, str],
    input_artifact_paths: dict[str, str],
    source_observability_dir: str,
    config: GitHubCommentDeliveryConfig,
    comment_title: str,
) -> str:
    critical_count = int(report["alert_severity_counts"]["critical"])
    warning_count = int(report["alert_severity_counts"]["warning"])
    recommendation = str(report["operator_recommendation"])
    phase6_status = str(report["phase6_status"])
    do_not_promote = (
        recommendation in {"blocked_fail_closed", "stop_paper_run"}
        or critical_count > 0
        or phase6_status != "approved_for_staged_live"
    )
    lines = [
        f"## {comment_title}",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| run_id | `{report['run_id']}` |",
        f"| feed_mode | `{report['feed_metrics']['feed_mode']}` |",
        f"| real_live_data | `{str(report['feed_metrics']['real_live_data']).lower()}` |",
        f"| deterministic_replay | `{str(report['feed_metrics']['deterministic_replay']).lower()}` |",
        f"| provider_name | `{report['feed_metrics']['provider_name']}` |",
        f"| instrument_id | `{report['feed_metrics']['instrument_id']}` |",
        f"| operator_recommendation | `{recommendation}` |",
        f"| phase6_deployment_status | `{phase6_status}` |",
        f"| feed_health_status | `{report['feed_health_status']}` |",
        f"| safety_status | `{report['safety_status']}` |",
        f"| alert_count | `{report['alert_count']}` |",
        f"| critical_alert_count | `{critical_count}` |",
        f"| warning_alert_count | `{warning_count}` |",
        f"| paper_only | `{str(report['paper_only']).lower()}` |",
        f"| capital_at_risk | `{str(report['capital_at_risk']).lower()}` |",
        "| broker_exchange_write_enabled | "
        f"`{str(report['broker_exchange_write_enabled']).lower()}` |",
        "| live_exchange_write_enabled | "
        f"`{str(report['live_exchange_write_enabled']).lower()}` |",
        "| polymarket_write_enabled | "
        f"`{str(report.get('polymarket_write_enabled', False)).lower()}` |",
        "| wallet_signing_enabled | "
        f"`{str(report.get('wallet_signing_enabled', False)).lower()}` |",
        "",
        "**Operator Action**",
        "",
        _operator_action(recommendation, do_not_promote),
        "",
        "**Alert Summary**",
        "",
    ]
    lines.extend(_alert_summary_lines(alerts, config.max_alerts_to_inline))
    lines.extend(
        [
            "",
            "**Key Metrics**",
            "",
            f"- cumulative_net_return: `{report['performance_metrics']['cumulative_net_return']}`",
            "- polymarket_realized_trade_pnl: "
            f"`{report['performance_metrics'].get('polymarket_realized_trade_pnl', 0.0)}`",
            "- polymarket_settlement_pnl: "
            f"`{report['performance_metrics'].get('polymarket_settlement_pnl', 0.0)}`",
            "- polymarket_total_pnl: "
            f"`{report['performance_metrics'].get('polymarket_total_pnl', 0.0)}`",
            f"- max_drawdown: `{report['risk_metrics']['max_drawdown']}`",
            f"- cost_drift_ratio: `{report['risk_metrics']['cost_drift_ratio']}`",
            f"- feed_gap_count: `{report['feed_metrics']['feed_gap_count']}`",
            f"- feed_late_event_count: `{report['feed_metrics']['feed_late_event_count']}`",
            f"- feed_out_of_order_count: `{report['feed_metrics']['feed_out_of_order_count']}`",
            f"- stale_event_count: `{report['feed_metrics']['stale_event_count']}`",
            f"- provider_disconnect_count: `{report['feed_metrics']['provider_disconnect_count']}`",
            f"- provider_reconnect_count: `{report['feed_metrics']['provider_reconnect_count']}`",
            f"- provider_error_count: `{report['feed_metrics']['provider_error_count']}`",
            f"- empty_response_count: `{report['feed_metrics']['empty_response_count']}`",
            f"- rate_limit_count: `{report['feed_metrics']['rate_limit_count']}`",
            "",
            "**Observed Paper Safety Flags**",
            "",
            _expected_flag_line("paper_only", report["paper_only"], True),
            _expected_flag_line("capital_at_risk", report["capital_at_risk"], False),
            _expected_flag_line(
                "broker_exchange_write_enabled",
                report["broker_exchange_write_enabled"],
                False,
            ),
            _expected_flag_line(
                "live_exchange_write_enabled",
                report["live_exchange_write_enabled"],
                False,
            ),
            _expected_flag_line(
                "polymarket_write_enabled",
                report.get("polymarket_write_enabled", False),
                False,
            ),
            _expected_flag_line(
                "wallet_signing_enabled",
                report.get("wallet_signing_enabled", False),
                False,
            ),
            "- automatic_deployment_promotion: `false` expected `false`",
            "",
        ]
    )
    if config.include_hashes:
        lines.extend(_hash_section(hashes, source_artifact_hashes))
    if config.include_artifact_paths:
        lines.extend(
            [
                "**Observability Artifacts**",
                "",
                f"- source_observability_dir: `{source_observability_dir}`",
                *[
                    f"- {name}: `{path}`"
                    for name, path in sorted(input_artifact_paths.items())
                ],
                "",
            ]
        )
    lines.append(f"_created_at: `{config.created_at}`_")
    lines.append("")
    return "\n".join(lines)


def _alert_summary_lines(
    alerts: list[dict[str, Any]],
    max_noncritical_alerts: int,
) -> list[str]:
    if not alerts:
        return ["- none"]

    critical_alerts = [
        alert for alert in alerts if str(alert.get("severity")) == "critical"
    ]
    noncritical_alerts = [
        alert for alert in alerts if str(alert.get("severity")) != "critical"
    ]

    lines: list[str] = []
    if critical_alerts:
        lines.append("- critical alerts:")
        lines.extend(_format_alert_line(alert) for alert in critical_alerts)
    else:
        lines.append("- critical alerts: none")

    inline_noncritical = noncritical_alerts[:max_noncritical_alerts]
    if inline_noncritical:
        lines.append("- warning/info alerts:")
        lines.extend(_format_alert_line(alert) for alert in inline_noncritical)

    omitted_noncritical = len(noncritical_alerts) - len(inline_noncritical)
    if omitted_noncritical > 0:
        lines.append(
            f"- `{omitted_noncritical}` non-critical alerts omitted from inline summary"
        )
    return lines


def _format_alert_line(alert: dict[str, Any]) -> str:
    return (
        f"  - `{alert['severity']}` `{alert['category']}` "
        f"`{alert['code']}`: {alert['message']}"
    )


def _expected_flag_line(name: str, actual: Any, expected: bool) -> str:
    return f"- {name}: `{str(actual).lower()}` expected `{str(expected).lower()}`"


def _operator_action(recommendation: str, do_not_promote: bool) -> str:
    if do_not_promote:
        return (
            "Do not promote to live trading. Keep the run paper-only and inspect "
            f"`{recommendation}` evidence before continuing."
        )
    return "Recommendation: continue_paper_run. Keep monitoring paper-only evidence."


def _hash_section(
    hashes: dict[str, str],
    source_artifact_hashes: dict[str, str],
) -> list[str]:
    lines = ["**Observability Artifact Hashes**", ""]
    for name, digest in sorted(hashes.items()):
        lines.append(f"- {name}: `{digest}`")
    lines.extend(["", "**Source Artifact Hashes**", ""])
    for name, digest in sorted(source_artifact_hashes.items()):
        lines.append(f"- {name}: `{digest}`")
    lines.append("")
    return lines


def _assert_output_dir_safe(*, source_path: Path, output_path: Path) -> None:
    if output_path == source_path:
        raise PaperAlertDeliveryError("output_dir must not equal observability_dir")
    if source_path in output_path.parents:
        raise PaperAlertDeliveryError("output_dir must not be inside observability_dir")
    if output_path in source_path.parents:
        raise PaperAlertDeliveryError("output_dir must not contain observability_dir")


def _gh_command(config: GitHubCommentDeliveryConfig, body_file: Path) -> str:
    return " ".join(
        [
            "gh",
            "issue",
            "comment",
            str(config.issue_number),
            "--repo",
            shlex.quote(config.repo_full_name),
            "--body-file",
            shlex.quote(str(body_file)),
        ]
    )


def _comment_body_path_for_command(config: GitHubCommentDeliveryConfig) -> Path:
    return Path(config.output_dir).expanduser().resolve() / "github_paper_comment.md"


def _post_direct_comment(
    *,
    config: GitHubCommentDeliveryConfig,
    comment_body_path: Path,
) -> dict[str, Any]:
    command = [
        "gh",
        "issue",
        "comment",
        str(config.issue_number),
        "--repo",
        config.repo_full_name,
        "--body-file",
        str(comment_body_path),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "post_mode": config.post_mode,
        "repo_full_name": config.repo_full_name,
        "issue_number": config.issue_number,
        "comment_url": completed.stdout.strip(),
        "created_at": config.created_at,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _write_text(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

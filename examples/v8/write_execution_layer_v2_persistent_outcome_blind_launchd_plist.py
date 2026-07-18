"""Write a launchd descriptor for the persistent outcome-blind collector."""

from __future__ import annotations

import argparse
import hashlib
import json
import plistlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "examples/v8/run_execution_layer_v2_persistent_outcome_blind_collector.py"
DEFAULT_PROTOCOL = ROOT / (
    "examples/v8/polymarket_configs/execution_layer_v2_persistent_outcome_blind_collector_v1.json"
)


def write_launchd_plist(
    *,
    output_path: Path | str,
    label: str,
    service_root: Path | str,
    protocol_path: Path | str,
    protocol_sha256: str,
    batch_round_count: int,
    python_executable: Path | str,
) -> dict:
    """Write a KeepAlive service descriptor without starting the service."""

    if not label.strip():
        raise ValueError("launchd label is required")
    if batch_round_count <= 0:
        raise ValueError("batch_round_count must be positive")
    service_root = Path(service_root).expanduser().resolve()
    direct_training_root = Path("/Volumes/PHILIPS/v8")
    if service_root == direct_training_root or direct_training_root in service_root.parents:
        raise ValueError("raw collector service_root cannot use direct training corpus root")
    protocol_path = Path(protocol_path).expanduser().resolve()
    if _sha256(protocol_path) != protocol_sha256.lower():
        raise ValueError("persistent collector protocol SHA-256 mismatch")
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    service_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "Label": label,
        "ProgramArguments": [
            str(Path(python_executable).expanduser().resolve()),
            str(RUNNER),
            "--service-root",
            str(service_root),
            "--protocol",
            str(protocol_path),
            "--protocol-sha256",
            protocol_sha256.lower(),
            "--batch-round-count",
            str(batch_round_count),
            "--max-batches",
            "0",
        ],
        "WorkingDirectory": str(ROOT),
        "EnvironmentVariables": {
            "PYTHONPATH": "src:.",
        },
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 30,
        "ProcessType": "Background",
        "StandardOutPath": str(service_root / "persistent_collector_stdout.log"),
        "StandardErrorPath": str(service_root / "persistent_collector_stderr.log"),
    }
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with temporary.open("wb") as handle:
        plistlib.dump(payload, handle, sort_keys=True)
        handle.flush()
    temporary.replace(output_path)
    return {
        "launchd_plist_path": str(output_path),
        "launchd_plist_sha256": _sha256(output_path),
        "launchd_label": label,
        "service_root": str(service_root),
        "continuous_collection": True,
        "restart_supervision_enabled": True,
        "outcome_blind_collection_only": True,
        "settlement_finalizer_started": False,
        "resolution_provider_called": False,
        "training_corpus_export_attempted": False,
        "labels_outcomes_or_pnl_opened": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--label",
        default="com.bigan.v8.persistent-outcome-blind-collector",
    )
    parser.add_argument("--service-root", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--batch-round-count", type=int, default=12)
    parser.add_argument("--python-executable", default=sys.executable)
    args = parser.parse_args(argv)
    report = write_launchd_plist(
        output_path=args.output_path,
        label=args.label,
        service_root=args.service_root,
        protocol_path=args.protocol,
        protocol_sha256=args.protocol_sha256,
        batch_round_count=args.batch_round_count,
        python_executable=args.python_executable,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

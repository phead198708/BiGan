#!/usr/bin/env python3
"""Diagnose prior evidence and freeze the bounded remediation candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_pre_promotion_remediation import (
    diagnose_and_select_remediation_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-dir", required=True, type=Path)
    args = parser.parse_args()
    result = diagnose_and_select_remediation_candidate(goal_dir=args.goal_dir)
    print(f"selected_candidate_name={result.selected_candidate_name}")
    print(f"previous_candidate_diagnosis={result.diagnosis_path}")
    print(f"development_evidence_manifest={result.development_manifest_path}")
    print(f"candidate_search_protocol={result.candidate_protocol_path}")
    print(
        "candidate_search_protocol_sha256="
        f"{result.candidate_protocol_sha256_path.read_text(encoding='utf-8').strip()}"
    )
    print(f"candidate_development_report={result.candidate_report_path}")
    print(f"selected_candidate_contract={result.selected_contract_path}")
    print(
        "selected_candidate_contract_sha256="
        f"{result.selected_contract_sha256_path.read_text(encoding='utf-8').strip()}"
    )
    print("uses_fresh_validation_for_selection=false")
    print("promotion_evidence_eligible=false")
    print("v8_execution_handoff_allowed=false")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only Polymarket CLOB auth and balance preflight.

This script intentionally never places orders. It derives CLOB L2 credentials
from POLYMARKET_PRIVATE_KEY at runtime so Builder/Relayer keys from the web UI
cannot be mistaken for SDK ApiCreds.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="Optional shell-style env file to load")
    parser.add_argument(
        "--signature-type",
        default=os.getenv("POLYMARKET_SIGNATURE_TYPE", "POLY_PROXY"),
        help="Polymarket signature type name or integer. Default: POLY_PROXY",
    )
    parser.add_argument(
        "--require-positive-balance",
        action="store_true",
        help="Exit non-zero when the collateral balance is zero.",
    )
    args = parser.parse_args()

    if args.env_file:
        _load_env_file(args.env_file)

    payload: dict[str, Any] = {
        "checked_at": datetime.now(UTC).isoformat(),
        "host": os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com"),
        "signature_type": str(args.signature_type),
        "auth_mode": "derive_clob_l2_from_private_key",
        "env_present": {
            key: bool(os.getenv(key))
            for key in (
                "POLYMARKET_PRIVATE_KEY",
                "POLYMARKET_FUNDER",
                "POLYMARKET_API_KEY",
                "POLYMARKET_API_SECRET",
                "POLYMARKET_API_PASSPHRASE",
                "POLYMARKET_CLOB_AUTH_MODE",
                "RELAYER_API_KEY",
                "RELAYER_API_KEY_ADDRESS",
            )
        },
        "masked": {
            "funder": _mask(os.getenv("POLYMARKET_FUNDER")),
            "api_key": _mask(os.getenv("POLYMARKET_API_KEY")),
            "relayer_key": _mask(os.getenv("RELAYER_API_KEY")),
            "relayer_key_address": _mask(os.getenv("RELAYER_API_KEY_ADDRESS")),
        },
    }

    try:
        balance_allowance = _read_balance_allowance(
            host=payload["host"],
            private_key=_require_env("POLYMARKET_PRIVATE_KEY"),
            funder=os.getenv("POLYMARKET_FUNDER"),
            signature_type_name=str(args.signature_type),
        )
    except Exception as exc:  # noqa: BLE001 - CLI should report third-party failures.
        payload["ok"] = False
        payload["error_type"] = type(exc).__name__
        payload["error"] = str(exc)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    payload["ok"] = True
    payload["balance_allowance"] = balance_allowance
    payload["balance_usdc"] = _collateral_balance_to_usdc(balance_allowance.get("balance"))
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.require_positive_balance and Decimal(str(payload["balance_usdc"])) <= 0:
        return 3
    return 0


def _read_balance_allowance(
    *,
    host: str,
    private_key: str,
    funder: str | None,
    signature_type_name: str,
) -> dict[str, Any]:
    from py_clob_client_v2 import ClobClient, SignatureTypeV2  # type: ignore[import-not-found]
    from py_clob_client_v2.clob_types import (  # type: ignore[import-not-found]
        AssetType,
        BalanceAllowanceParams,
    )

    signature_type = _resolve_signature_type(SignatureTypeV2, signature_type_name)
    client = ClobClient(
        host,
        key=private_key,
        chain_id=137,
        signature_type=signature_type,
        funder=funder,
    )
    creds = client.create_or_derive_api_key()
    client.set_api_creds(creds)
    return client.get_balance_allowance(
        BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL,
            signature_type=signature_type,
        )
    )


def _resolve_signature_type(signature_type_enum: Any, value: str) -> Any:
    text = value.strip()
    if text.isdigit():
        return signature_type_enum(int(text))
    return getattr(signature_type_enum, text.upper())


def _collateral_balance_to_usdc(raw_balance: Any) -> str:
    try:
        raw = Decimal(str(raw_balance or "0"))
    except InvalidOperation:
        return "0"
    if raw == 0:
        return "0"
    return str((raw / Decimal("1000000")).quantize(Decimal("0.000001")).normalize())


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _mask(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 10:
        return "***"
    return f"{value[:6]}...{value[-4:]}"


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json

import pytest

from bigan.paper_trading.dashboard.__main__ import main
from bigan.paper_trading.dashboard.server import SECURITY_HEADERS, loopback_host

ROUTES = ["/", "/healthz", "/readyz", "/api/v1/dashboard", "/api/v1/status", "/api/v1/account",
          "/api/v1/runs", "/api/v1/decisions", "/api/v1/fills", "/api/v1/settlements",
          "/static/app.js", "/static/styles.css"]


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("method", ["GET", "HEAD"])
async def test_routes_headers_and_head(client, route, method):
    response = await client.request(method, route, headers={"Origin": "https://untrusted.example"})
    assert response.status == 200
    assert all(response.headers[key] == value for key, value in SECURITY_HEADERS.items())
    assert "Access-Control-Allow-Origin" not in response.headers
    if route.startswith(("/api", "/health", "/ready")):
        assert response.content_type == "application/json"
    if method == "HEAD":
        assert await response.read() == b""


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
@pytest.mark.parametrize("route", ROUTES)
async def test_all_routes_reject_write_methods(client, route, method):
    response = await client.request(method, route)
    assert response.status == 405
    assert response.headers["Allow"] == "GET, HEAD"
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("query", ["limit=0", "limit=-1", "limit=1.5", "limit=word", "limit=9999999", "limit=", "limit=1&limit=2", "unknown=true", "before_run_id=..", "before_run_id=/private", "before_run_id=", "before_run_id=paper-123", "before_run_id=" + "a" * 200, "before_run_id=paper-" + "a" * 24 + "&before_run_id=paper-" + "a" * 24])
async def test_invalid_history_queries(client, query):
    response = await client.get("/api/v1/dashboard?" + query)
    assert response.status == 400
    assert (await response.json())["message"] == "Invalid request"


@pytest.mark.parametrize("limit", [1, 500])
async def test_valid_bounds(client, limit):
    response = await client.get(f"/api/v1/fills?limit={limit}")
    assert response.status == 200
    assert len((await response.json())["items"]) <= limit


@pytest.mark.parametrize("route", ["/api/v1/account", "/api/v1/status", "/readyz", "/healthz", "/"])
async def test_non_history_routes_reject_queries(client, route):
    assert (await client.get(route + "?limit=1")).status == 400


async def test_unavailable_status_liveness_readiness_and_sanitized_response(client, bundle):
    bundle.reader.status_path.write_text("/private/TOKEN_SECRET bad json")
    assert (await client.get("/healthz")).status == 200
    for route in ("/readyz", "/api/v1/dashboard", "/api/v1/status"):
        response = await client.get(route)
        assert response.status == 503
        body = await response.text()
        assert "TOKEN_SECRET" not in body and "/private" not in body and "Traceback" not in body
        assert response.headers["Cache-Control"] == "no-store"


async def test_combined_partial_failure_vs_standalone_503(client, bundle):
    (bundle.operator.session.store.run_dir / "paper_idempotency.sqlite3").unlink()
    response = await client.get("/api/v1/dashboard")
    assert response.status == 200
    view = await response.json()
    assert view["schema_version"] == 1 and view["account"] is not None
    assert view["recent"]["fills"] is None
    assert (await client.get("/api/v1/fills")).status == 503
    assert (await client.get("/api/v1/decisions")).status == 200


async def test_rebinding_host_and_unknown_route_fail_safely(client):
    assert (await client.get("/api/v1/dashboard", headers={"Host": "untrusted.example"})).status == 400
    response = await client.get("/not-a-route")
    assert response.status == 404
    assert response.headers["X-Frame-Options"] == "DENY"


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.1", "example.com", "localhost", "127.0.0.1.example.com"])
def test_cli_rejects_non_literal_loopback_before_config_io(host, capsys):
    with pytest.raises(ValueError):
        loopback_host(host)
    with pytest.raises(SystemExit) as exit:
        main(["--config", "/private/no-such-secret.toml", "--host", host])
    assert exit.value.code == 2
    assert "/private/no-such-secret.toml" not in capsys.readouterr().err


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1"])
def test_loopback_hosts(host):
    assert loopback_host(host) == host


async def test_endpoint_cursor_excludes_the_named_run(client, bundle):
    response = await client.get("/api/v1/runs?before_run_id=" + bundle.operator.run_id)
    assert response.status == 200
    assert (await response.json())["items"] == []
    response = await client.get("/api/v1/dashboard")
    assert "NaN" not in json.dumps(await response.json(), allow_nan=False)

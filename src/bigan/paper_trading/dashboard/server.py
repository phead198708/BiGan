"""Loopback-only aiohttp service; fixed routes and packaged static assets."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
from collections.abc import Awaitable, Callable
from importlib.resources import files
from urllib.parse import urlsplit

from aiohttp import web

from .reader import SECTIONS, DashboardReader, DashboardUnavailable, validate_cursor

SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
}


def loopback_host(host: str) -> str:
    try:
        if ipaddress.ip_address(host).is_loopback:
            return host
    except ValueError:
        pass
    raise ValueError("Dashboard requires a literal loopback address")


def _json(payload: object, *, status: int = 200) -> web.Response:
    return web.Response(
        text=json.dumps(payload, allow_nan=False, ensure_ascii=False),
        status=status, content_type="application/json",
    )


@web.middleware
async def security(request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]) -> web.StreamResponse:
    try:
        # Bind checks in the CLI plus peer/Host checks here prevent accidental
        # non-loopback embedding and DNS rebinding. No CORS or proxy trust.
        if request.remote is not None:
            loopback_host(request.remote)
        host = urlsplit("//" + request.host).hostname
        if host != "localhost":
            loopback_host(host or "")
        if request.method not in {"GET", "HEAD"}:
            response = _json({"error": "METHOD_NOT_ALLOWED", "message": "Read-only service"}, status=405)
            response.headers["Allow"] = "GET, HEAD"
        else:
            response = await handler(request)
    except DashboardUnavailable as exc:
        response = _json({"error": "UNAVAILABLE", "message": str(exc)}, status=503)
    except ValueError:
        response = _json({"error": "BAD_REQUEST", "message": "Invalid request"}, status=400)
    except web.HTTPException as exc:
        response = _json({"error": "HTTP_ERROR", "message": "Request unavailable"}, status=exc.status)
    except Exception:
        response = _json({"error": "UNAVAILABLE", "message": "Dashboard data is temporarily unavailable"}, status=503)
    response.headers.update(SECURITY_HEADERS)
    return response


def _query(request: web.Request, reader: DashboardReader, *, history: bool) -> tuple[int | None, str | None]:
    allowed = {"limit", "before_run_id"} if history else set()
    for key in request.query:
        if key not in allowed or len(request.query.getall(key)) != 1:
            raise ValueError("Invalid query parameters")
    raw = request.query.get("limit")
    limit = None
    if raw is not None:
        if len(raw) > 9 or not re.fullmatch(r"[0-9]+", raw):
            raise ValueError("Invalid limit")
        limit = int(raw)
        if not 1 <= limit <= reader.config.recent_query_max:
            raise ValueError("Invalid limit")
    cursor = request.query.get("before_run_id")
    validate_cursor(cursor)
    return limit, cursor


def create_app(reader: DashboardReader) -> web.Application:
    app = web.Application(middlewares=[security], client_max_size=1024)
    slots = asyncio.Semaphore(4)

    async def live(request: web.Request) -> web.Response:
        _query(request, reader, history=False)
        return _json({"alive": True, "paper_only": True, "read_only": True})

    async def status(request: web.Request) -> web.Response:
        _query(request, reader, history=False)
        async with slots:
            view = await asyncio.to_thread(reader.read_status)
        if request.path == "/readyz":
            return _json({"ready": True, "stale": view["stale"]})
        return _json({key: view[key] for key in (
            "schema_version", "generated_at_ms", "status", "status_age_ms", "stale", "warnings",
        )})

    async def data(request: web.Request) -> web.Response:
        section = request.path.rsplit("/", 1)[-1]
        limit, cursor = _query(request, reader, history=section != "account")
        async with slots:
            view = await asyncio.to_thread(reader.read, limit=limit, before_run_id=cursor)
        if section == "dashboard":
            return _json(view)
        if section == "account":
            result = {key: view[key] for key in ("account", "positions", "warnings")}
            available = view["account"] is not None
        else:
            result = {"items": view["recent"][section], "warnings": view["warnings"], "before_run_id": cursor}
            available = result["items"] is not None
        return _json({"schema_version": 1, "run_id": view["status"]["run_id"], **result}, status=200 if available else 503)

    async def asset(request: web.Request) -> web.Response:
        _query(request, reader, history=False)
        name = {"/": "index.html", "/static/app.js": "app.js", "/static/styles.css": "styles.css"}[request.path]
        content_type = {"index.html": "text/html", "app.js": "application/javascript", "styles.css": "text/css"}[name]
        return web.Response(body=files(__package__).joinpath("static", name).read_bytes(), content_type=content_type)

    app.router.add_get("/healthz", live)
    app.router.add_get("/readyz", status)
    app.router.add_get("/api/v1/status", status)
    for section in ("dashboard", "account", *SECTIONS):
        app.router.add_get(f"/api/v1/{section}", data)
    for path in ("/", "/static/app.js", "/static/styles.css"):
        app.router.add_get(path, asset)
    return app

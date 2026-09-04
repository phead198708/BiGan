from __future__ import annotations

import ast
from importlib.resources import files


def test_assets_are_packaged_and_required_sections_exist():
    root = files("bigan.paper_trading.dashboard")
    html = root.joinpath("static/index.html").read_text()
    assert "PAPER / SIMULATED — NO REAL FUNDS" in html
    for section in ("account", "market", "operator", "positions", "feeds", "alpha", "inputs", "decisions", "fills", "settlements", "runs"):
        assert f'id="{section}-title"' in html
    assert '<script src="/static/app.js" defer>' in html
    assert "Load older" in html and "UTC" in html
    css = root.joinpath("static/styles.css").read_text()
    assert "@media" in css and ":focus-visible" in css and "overflow-x: auto" in css


def test_javascript_uses_text_nodes_bounded_pages_and_preserves_last_view():
    script = files("bigan.paper_trading.dashboard").joinpath("static/app.js").read_text()
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert forbidden not in script
    assert "textContent" in script and 'el("th"' in script
    assert 'th.scope = "col"' in script
    assert "setTimeout(refresh, 2000)" in script
    assert "Keeping the last successful view" in script
    assert "before_run_id=" in script
    assert "Number.isFinite" in script
    assert "latest_decision" not in script  # correct operator field: last_decision


def test_dashboard_does_not_import_operator_session_runner_or_legacy_dashboard():
    root = files("bigan.paper_trading.dashboard")
    for module in ("reader.py", "server.py", "__main__.py"):
        tree = ast.parse(root.joinpath(module).read_text())
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any(name.endswith(("operator.runtime", "session", "strategy_runner", "monitoring.dashboard")) for name in imports)

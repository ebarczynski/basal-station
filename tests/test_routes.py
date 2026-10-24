"""Route wiring regression tests — catch decorator/signature mistakes cheaply."""

import inspect

from basal_station.api.app import create_app
from basal_station.api.schemas import DecideRequest


def _route(app, path: str, method: str):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route
    raise AssertionError(f"route not found: {method} {path}")


def test_decide_route_signature_not_mangled():
    """The /decide endpoint must take a DecideRequest body, not raw helper args."""
    app = create_app()
    endpoint = _route(app, "/api/decide", "POST").endpoint
    params = inspect.signature(endpoint).parameters
    assert "body" in params
    assert params["body"].annotation is DecideRequest
    # the shared helper's args must never leak into the endpoint
    for leaked in ("state_text", "raw_questions", "app_state"):
        assert leaked not in params


def test_core_routes_registered():
    app = create_app()
    expected = [
        ("/api/system", "GET"),
        ("/api/decide", "POST"),
        ("/api/decide/raw", "POST"),
        ("/api/eval", "POST"),
        ("/api/recipes", "GET"),
        ("/api/recipes/{key}", "GET"),
        ("/api/recipes/{key}/run", "POST"),
        ("/api/engine/start", "POST"),
        ("/api/engine/stop", "POST"),
        ("/api/models/download", "POST"),
        ("/api/knowledge/search", "POST"),
    ]
    for path, method in expected:
        _route(app, path, method)

"""Route wiring regression tests — catch decorator/signature mistakes cheaply.

FastAPI (recent versions) wraps included routers lazily in app.routes, so the
tests inspect the APIRouter itself rather than the app's route list.
"""

import inspect

from basal_station.api.routes import router
from basal_station.api.schemas import DecideRequest


def _route(path: str, method: str):
    for route in router.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route
    raise AssertionError(f"route not found: {method} {path}")


def test_decide_route_signature_not_mangled():
    """The /decide endpoint must take a DecideRequest body, not raw helper args."""
    endpoint = _route("/decide", "POST").endpoint
    params = inspect.signature(endpoint).parameters
    assert "body" in params
    annotation = params["body"].annotation
    assert annotation is DecideRequest or DecideRequest in getattr(annotation, "__args__", ())
    # the shared helper's args must never leak into the endpoint
    for leaked in ("state_text", "raw_questions", "app_state"):
        assert leaked not in params


def test_core_routes_registered():
    expected = [
        ("/system", "GET"),
        ("/decide", "POST"),
        ("/decide/raw", "POST"),
        ("/eval", "POST"),
        ("/recipes", "GET"),
        ("/recipes/{key}", "GET"),
        ("/recipes/{key}/run", "POST"),
        ("/engine/start", "POST"),
        ("/engine/stop", "POST"),
        ("/models/download", "POST"),
        ("/knowledge/search", "POST"),
    ]
    for path, method in expected:
        _route(path, method)

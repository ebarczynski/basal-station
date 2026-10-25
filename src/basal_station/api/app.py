"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..backends import EngineManager
from ..config import AppSettings, load_settings
from ..hardware import detect
from ..knowledge import KnowledgeService
from .routes import router

log = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"


def create_app(settings: AppSettings | None = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        detect()  # warm the hardware cache
        app.state.settings = settings
        app.state.engine = EngineManager(settings)
        app.state.knowledge = KnowledgeService(settings.knowledge)
        await app.state.knowledge.start()
        log.info("basal-station ready (UI at http://%s:%d)", settings.server.host, settings.server.port)
        yield
        await app.state.knowledge.close()
        await app.state.engine.stop()

    app = FastAPI(title="basal-station", version="0.1.0", lifespan=lifespan)
    app.include_router(router)

    if UI_DIR.is_dir():
        # The console is a local tool with no asset versioning: serve
        # everything no-cache so a restarted station is never masked by a
        # stale cached app.js (that class of bug looks like "the app is
        # broken" from the outside).
        class NoCacheStaticFiles(StaticFiles):
            def file_response(self, *args, **kwargs):
                response = super().file_response(*args, **kwargs)
                response.headers["Cache-Control"] = "no-cache"
                return response

        app.mount("/ui", NoCacheStaticFiles(directory=UI_DIR, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        async def index():
            return FileResponse(UI_DIR / "index.html",
                                headers={"Cache-Control": "no-cache"})

    return app

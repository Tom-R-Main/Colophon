"""Colophon web UI — FastAPI app factory."""

from __future__ import annotations

from pathlib import Path


def create_app(*, db_url: str | None = None):
    """Create the FastAPI application."""
    from fastapi import FastAPI
    from fastapi.responses import FileResponse

    from colophon.web.api import create_router

    app = FastAPI(title="Colophon", description="Stylometric analysis web UI")

    # API routes FIRST (before static mount)
    router = create_router(db_url=db_url)
    app.include_router(router)

    # Serve static frontend at root — use a catch-all route instead of mount
    # so API routes take precedence
    static_dir = Path(__file__).parent / "static"

    @app.get("/")
    async def serve_index():
        return FileResponse(str(static_dir / "index.html"))

    return app

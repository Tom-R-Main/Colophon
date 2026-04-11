"""Colophon web UI — FastAPI app factory."""

from __future__ import annotations

from pathlib import Path


def create_app(*, db_url: str | None = None):
    """Create the FastAPI application."""
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    from colophon.web.api import create_router

    app = FastAPI(title="Colophon", description="Stylometric analysis web UI")

    # API routes
    router = create_router(db_url=db_url)
    app.include_router(router)

    # Serve static frontend
    static_dir = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
